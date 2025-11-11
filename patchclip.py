#!/usr/bin/env python3
"""
eval_clip_advpatch_sampled.py
加速版：每个数据集抽样用于 craft patch，批量化应用patch并评估 CLIP。
依赖: torch, torchvision, clip (OpenAI), adversarial-robustness-toolbox (ART), tqdm, numpy

用法示例:
    python eval_clip_advpatch_sampled.py --device cuda --batch_size 64 --max_iter 100 --sample_per_ds 200
    python eval_clip_advpatch_sampled.py --dataset CIFAR100 --device cuda --batch_size 128 --max_iter 80 --sample_per_ds 100
"""
import argparse, os, time
from tqdm import tqdm
import torch, clip
import torchvision, numpy as np
from torchvision import transforms, datasets
import torch.nn as nn
from art.attacks.evasion import AdversarialPatch
from art.estimators.classification import PyTorchClassifier

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--patch_size', type=int, default=80)
    p.add_argument('--max_iter', type=int, default=100, help='ART patch max iterations (craft)')
    p.add_argument('--sample_per_ds', type=int, default=200, help='用于craft patch的每个数据集样本数（抽样）')
    p.add_argument('--max_samples_global', type=int, default=None, help='若设置则为全局采样上限（覆盖 sample_per_ds）')
    p.add_argument('--data_root', default='./data')
    p.add_argument('--output', default='results_sampled.txt')
    p.add_argument('--random_placement', action='store_true', help='在应用patch时随机放置（会慢一些）')
    p.add_argument('--dataset', default=None, help='只跑单个数据集（如 CIFAR100），否则跑全部列表')
    return p.parse_args()

# ---------------- dataset loaders ----------------
def make_loaders(name, data_root, batch_size):
    print("make_loaders")
    raw_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    norm_tf = transforms.Compose([transforms.Resize((224,224)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.48145466,0.4578275,0.40821073),
                                                       (0.26862954,0.26130258,0.27577711))])
    n = name.lower()
    if n == 'cifar100':
        ds_raw = datasets.CIFAR100(root=data_root, train=False, download=True, transform=raw_tf)
        ds_norm = datasets.CIFAR100(root=data_root, train=False, download=False, transform=norm_tf)
        classes = ds_norm.classes
    elif n == 'food101':
        ds_raw = datasets.Food101(root=data_root, split='test', download=True, transform=raw_tf)
        ds_norm = datasets.Food101(root=data_root, split='test', download=False, transform=norm_tf)
        classes = ds_norm.classes
    elif n in ('oxfordpet', 'oxfordiiitpet'):
        ds_raw = datasets.OxfordIIITPet(root=data_root, download=True, transform=raw_tf, target_types='category')
        ds_norm = datasets.OxfordIIITPet(root=data_root, download=False, transform=norm_tf, target_types='category')
        try:
            classes = ds_norm.classes
        except:
            classes = [str(i) for i in range(37)]
    elif n == 'dtd':
        ds_raw = datasets.DTD(root=data_root, transform=raw_tf, download=True)
        ds_norm = datasets.DTD(root=data_root, transform=norm_tf, download=False)
        classes = ds_norm.classes
    elif n == 'stl10':
        ds_raw = datasets.STL10(root=data_root, split='test', download=True, transform=raw_tf)
        ds_norm = datasets.STL10(root=data_root, split='test', download=False, transform=norm_tf)
        classes = ds_norm.classes
    else:
        raise ValueError('Unknown dataset: ' + name)

    raw_loader = torch.utils.data.DataLoader(ds_raw, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    norm_loader = torch.utils.data.DataLoader(ds_norm, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return raw_loader, norm_loader, classes

# ---------------- utility: collect sample numpy arrays ----------------
def collect_samples(raw_loader, max_samples):
    X_list = []
    y_list = []
    cnt = 0
    for imgs, targets in raw_loader:
        imgs_np = imgs.numpy()
        X_list.append(imgs_np)
        y_list.append(np.array(targets))
        cnt += imgs_np.shape[0]
        if cnt >= max_samples:
            break
    if len(X_list) == 0:
        return np.zeros((0,3,224,224), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.concatenate(X_list, axis=0)[:max_samples]
    Y = np.concatenate(y_list, axis=0)[:max_samples]
    return X, Y

# ----------------- vectorized patch application (GPU) -----------------
def apply_patch_batch_tensor(imgs_tensor, patch_tensor, scale=0.3, random_place=False):
    print("apply_patch_batch_tensor")
    # imgs_tensor: (B,3,224,224) in 0-1 on device
    B, C, H, W = imgs_tensor.shape
    # resize patch once
    ph = max(1, int(patch_tensor.shape[1]*scale))
    pw = max(1, int(patch_tensor.shape[2]*scale))
    patch_resized = torch.nn.functional.interpolate(patch_tensor.unsqueeze(0), size=(ph,pw), mode='bilinear', align_corners=False)[0]
    imgs_p = imgs_tensor.clone()
    if not random_place:
        y0 = H - ph
        x0 = W - pw
        imgs_p[:, :, y0:y0+ph, x0:x0+pw] = patch_resized.unsqueeze(0).expand(B, -1, -1, -1)
    else:
        # random placement per image (vectorized-ish by computing indices)
        ys = torch.randint(0, H-ph+1, (B,), device=imgs_tensor.device)
        xs = torch.randint(0, W-pw+1, (B,), device=imgs_tensor.device)
        for i in range(B):
            y0 = int(ys[i].item()); x0 = int(xs[i].item())
            imgs_p[i, :, y0:y0+ph, x0:x0+pw] = patch_resized
    return imgs_p

# ---------------- main evaluation ----------------
def evaluate_dataset(ds_name, clip_model, clip_device, args):
    print("evaluate1")
    print(f'\n=== Dataset: {ds_name} ===')
    raw_loader, norm_loader, classes = make_loaders(ds_name, args.data_root, args.batch_size)
    # make text features
    text_prompts = [f"a photo of a {c}" for c in classes]
    tokenized = clip.tokenize(text_prompts).to(clip_device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(tokenized)
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

    # clean eval
    correct = total = 0
    clip_model.eval()
    with torch.no_grad():
        for imgs, targets in tqdm(norm_loader, desc=f'{ds_name} clean'):
            imgs = imgs.to(clip_device)
            feats = clip_model.encode_image(imgs)
            feats = feats / feats.norm(dim=1, keepdim=True)
            sims = (100.0 * feats @ text_feats.T)
            preds = sims.argmax(dim=1).cpu().numpy()
            targets_np = targets.numpy() if hasattr(targets, 'numpy') else np.array(targets)
            total += len(preds); correct += (preds == targets_np).sum()
    clean_acc = 100.0 * correct / total
    print(f'Clean acc: {clean_acc:.2f}%')

    # ---------- craft patch using surrogate + ART on sampled subset ----------
    raw_for_sample_loader = torch.utils.data.DataLoader(raw_loader.dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    sample_n = args.sample_per_ds if args.max_samples_global is None else min(args.sample_per_ds, args.max_samples_global)
    X_samples, Y_samples = collect_samples(raw_for_sample_loader, sample_n)
    if X_samples.shape[0] == 0:
        raise RuntimeError('No samples collected for patch crafting.')
    print(f'Collected {X_samples.shape[0]} samples for crafting patch for {ds_name} (max_iter={args.max_iter})')

    # surrogate model (pretrained resnet18)
    surrogate = torchvision.models.resnet18(pretrained=True).to(clip_device)
    surrogate.eval()
    sur_loss = nn.CrossEntropyLoss()
    sur_opt = torch.optim.SGD(surrogate.parameters(), lr=1e-3)
    sur_classifier = PyTorchClassifier(model=surrogate, loss=sur_loss, optimizer=sur_opt,
                                       input_shape=(3,224,224), nb_classes=1000, clip_values=(0.0,1.0),
                                       device_type='gpu' if args.device.startswith('cuda') else 'cpu')

    patch_attack = AdversarialPatch(
        classifier=sur_classifier,
        patch_shape=(3, args.patch_size, args.patch_size),
        learning_rate=5.0,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        scale_min=0.2,
        scale_max=0.5,
        rotation_max=22.5,
        random_location=True,
        verbose=False
    )
    t0 = time.time()
    print('Crafting adversarial patch (surrogate, ART)...')
    patch = patch_attack.generate(x=X_samples, y=Y_samples)
    # ensure shape (3,h,w)
    if patch.ndim == 4:
        patch = patch[0]
    patch_t = torch.tensor(patch, dtype=torch.float32).to(clip_device)
    t1 = time.time()
    print(f'Patch crafted in {(t1-t0)/60:.2f} min.')

    # ---------- evaluate adv accuracy: apply patch to raw test loader in batches (GPU) ----------
    correct_adv = total_adv = 0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=clip_device).view(1,3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=clip_device).view(1,3,1,1)

    # Use raw test loader (not shuffled) to be consistent with clean eval ordering
    test_raw_loader = torch.utils.data.DataLoader(raw_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for raw_imgs, targets in tqdm(test_raw_loader, desc=f'{ds_name} adv'):
            imgs_t = raw_imgs.to(clip_device).float()  # 0-1
            # apply patch (scale fixed for speed; can be randomized)
            patched = apply_patch_batch_tensor(imgs_t, patch_t, scale=0.3, random_place=args.random_placement)
            # normalize for CLIP
            patched_norm = (patched - mean) / std
            feats = clip_model.encode_image(patched_norm)
            feats = feats / feats.norm(dim=1, keepdim=True)
            sims = (100.0 * feats @ text_feats.T)
            preds = sims.argmax(dim=1).cpu().numpy()
            targets_np = targets.numpy() if hasattr(targets, 'numpy') else np.array(targets)
            total_adv += len(preds); correct_adv += (preds == targets_np).sum()

    adv_acc = 100.0 * correct_adv / total_adv
    print(f'Adv acc: {adv_acc:.2f}%')

    return f"{ds_name}\tclean_acc: {clean_acc:.2f}%\tadv_acc: {adv_acc:.2f}%"
    

def main():
    print("main")
    args = parse_args()
    # dataset list
    datasets_list = ['Food101','CIFAR100','OxfordIIITPet','DTD','STL10']
    if args.dataset:
        datasets_list = [args.dataset]

    # load CLIP
    device = args.device
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.to(device)
    # use fp16 on CUDA if available to speed up
    use_fp16 = device.startswith('cuda')
    if use_fp16:
        clip_model = clip_model.half()

    results = []
    for ds in datasets_list:
        res_line = evaluate_dataset(ds, clip_model, device, args)
        results.append(res_line)
        # free memory
        torch.cuda.empty_cache()

    # save results
    with open(args.output, 'w') as f:
        for r in results:
            f.write(r + "\n")
    print('All done. Results saved to', args.output)

if __name__ == '__main__':
    main()
