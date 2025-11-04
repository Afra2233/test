"""
Full stAdv implementation (spatially transformed adversarial examples) for Food101 + CLIP
"""

import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm


def make_text_features(model, classes, device):
    template = "a photo of a {}"
    texts = [template.format(c.replace('_', ' ')) for c in classes]
    with torch.no_grad():
        tokenized = clip.tokenize(texts).to(device)
        text_features = model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.to(torch.float32)
    return text_features


def normalize_for_clip(x, device):
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
    return (x - clip_mean) / clip_std


def tv_norm_dense(flow):
    dh = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
    dw = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1]).mean()
    return dh + dw


def upsample_control_grid(control, out_h, out_w):
    return F.interpolate(control, size=(out_h, out_w), mode='bicubic', align_corners=True)


def make_sampling_grid(N, H, W, device):
    yy, xx = torch.meshgrid(
        torch.linspace(-1,1,H, device=device), torch.linspace(-1,1,W, device=device), indexing='ij')
    base_grid = torch.stack((xx, yy), dim=-1)
    base_grid = base_grid.unsqueeze(0).repeat(N,1,1,1)
    return base_grid


def dense_flow_to_grid(flow, device):
    N,C,H,W = flow.shape
    flow_x = flow[:,1:2,:,:]
    flow_y = flow[:,0:1,:,:]
    flow_xn = flow_x / ((W-1)/2.0)
    flow_yn = flow_y / ((H-1)/2.0)
    flow_norm = torch.cat([flow_xn.permute(0,2,3,1), flow_yn.permute(0,2,3,1)], dim=-1)
    return flow_norm


def warp_image_with_flow(img, dense_flow, device):
    N,C,H,W = img.shape
    base_grid = make_sampling_grid(N,H,W,device)
    flow_norm = dense_flow_to_grid(dense_flow, device)
    sampling_grid = base_grid + flow_norm
    warped = F.grid_sample(img, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped


class StAdvAttack:
    def __init__(self, model, text_features, device, grid_size=8, max_pixel=8.0, l2_lambda=0.01, tv_lambda=0.2):
        self.model = model
        self.text_features = text_features
        self.device = device
        self.grid_size = grid_size
        self.max_pixel = max_pixel
        self.l2_lambda = l2_lambda
        self.tv_lambda = tv_lambda

    def attack_batch(self, images, labels, steps=200, lr=0.1):
        N, C, H, W = images.shape
        Gh = Gw = self.grid_size
        control = torch.zeros((N, 2, Gh, Gw), device=self.device, requires_grad=True)
        control.data += torch.randn_like(control) * 0.01
        optimizer = torch.optim.Adam([control], lr=lr)

        best_warp = images.clone()
        best_score = torch.full((N,), -1e9, device=self.device, dtype=torch.float32)

        # attack in full precision to avoid AMP dtype mismatch
        for _ in range(steps):
            optimizer.zero_grad()
            dense_flow = upsample_control_grid(control, H, W)
            dense_flow_clamped = dense_flow.clamp(-self.max_pixel, self.max_pixel)
            warped = warp_image_with_flow(images.float(), dense_flow_clamped, self.device)
            inp = normalize_for_clip(warped, self.device)

            # run CLIP forward in float32 (disable autocast)
            with torch.cuda.amp.autocast(enabled=False):
                image_features = self.model.encode_image(inp)
                image_features = image_features.float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = 100.0 * (image_features @ self.text_features.t())
                loss_ce = F.cross_entropy(logits, labels, reduction='none')
                loss_mean = loss_ce.mean()

            loss_l2 = (dense_flow_clamped.norm(p=2, dim=1).pow(2).mean())
            loss_tv = tv_norm_dense(dense_flow_clamped)
            opt_loss = -loss_mean + self.l2_lambda * loss_l2 + self.tv_lambda * loss_tv
            opt_loss.backward()
            optimizer.step()

            with torch.no_grad():
                control.clamp_(-self.max_pixel, self.max_pixel)
                per = loss_ce.to(torch.float32)  # force dtype match
                update_mask = per > best_score
                if update_mask.any():
                    best_score[update_mask] = per[update_mask]
                    best_warp[update_mask] = warped[update_mask]

        return best_warp.detach().to(torch.float32)

def evaluate_clip_on_loader(model, text_features, loader, device='cuda', attack=None, attack_obj=None):
    model.eval()
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc='Eval'):
        images = images.to(device)
        labels = labels.to(device)
        if attack is not None and attack_obj is not None:
            images = attack_obj.attack_batch(images, labels, steps=attack['steps'], lr=attack['lr'])
        inp = normalize_for_clip(images, device)
        with torch.no_grad():
            image_features = model.encode_image(inp)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * (image_features @ text_features.t())
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--grid-size', type=int, default=8)
    parser.add_argument('--max-pixel', type=float, default=8.0)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--tv', type=float, default=0.2)
    parser.add_argument('--save-attacked', action='store_true')
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    print('Loading CLIP model...')
    model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
    model = model.eval().to(device)
    model.float()
    print('Loading Food101 test set...')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    valset = datasets.Food101(root='./data', split='test', download=True, transform=transform)
    loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    classes = valset.classes
    print(f'Number of classes: {len(classes)}')
    print('Constructing text features...')
    text_features = make_text_features(model, classes, device)
    print('Evaluating clean accuracy...')
    clean_acc = evaluate_clip_on_loader(model, text_features, loader, device=device, attack=None, attack_obj=None)
    print(f'CLIP zero-shot accuracy on clean Food101 test: {clean_acc*100:.2f}%')
    print('Running full stAdv attack and evaluating attacked accuracy...')
    attack_obj = StAdvAttack(model, text_features, device, grid_size=args.grid_size, max_pixel=args.max_pixel, l2_lambda=args.l2, tv_lambda=args.tv)
    attack_cfg = {'steps': args.steps, 'lr': args.lr}
    attacked_acc = evaluate_clip_on_loader(model, text_features, loader, device=device, attack=attack_cfg, attack_obj=attack_obj)
    print(f'CLIP zero-shot accuracy on stAdv-attacked images: {attacked_acc*100:.2f}%')
    print('Done.')


if __name__ == '__main__':
    main()

