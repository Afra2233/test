"""
eval_clip_advpatch.py
依赖: torch, torchvision, clip (OpenAI), adversarial-robustness-toolbox (ART), tqdm
用法示例:
    python eval_clip_advpatch.py --batch_size 64 --device cuda
"""

import os
import argparse
from tqdm import tqdm
import torch
import clip
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np

# ART imports
from art.attacks.evasion import AdversarialPatch
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn

# --------- 参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--patch_size', type=int, default=80, help='patch size in pixels (square)')
parser.add_argument('--max_iter', type=int, default=200, help='ART adversarial patch iterations (可调)')
parser.add_argument('--output', default='results.txt', help='保存结果')
parser.add_argument('--data_root', default='./data', help='datasets 下载/存放目录')
args = parser.parse_args()

device = args.device
batch_size = args.batch_size
data_root = args.data_root

# ---------- 工具：CLIP wrapper model ----------
class CLIPWrapper(nn.Module):
    def __init__(self, model, preprocess, device):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.device = device

    def forward(self, x):
        # x: batch of preprocessed images (tensor) already normalized consistent with CLIP
        # returns logits over text tokens will be computed outside (see below).
        # But ART classifier expects model(x) -> logits vector.
        # We'll compute CLIP image features and compute logits vs a fixed set of text features
        return self.model.encode_image(x)

# ---------- helper: build dataset loaders ----------
def get_dataset_loader(name):
    # Returns (loader, class_names)
    # transforms: resize->224 (CLIP's default input), center crop, to tensor, normalize with CLIP stats
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])

    if name.lower() == 'cifar100':
        ds = datasets.CIFAR100(root=data_root, train=False, download=True,
                               transform=preprocess)
        classes = ds.classes
    elif name.lower() == 'food101':
        ds = datasets.Food101(root=data_root, split='test', download=True, transform=preprocess)
        classes = ds.classes
    elif name.lower() == 'oxfordpet' or name.lower() == 'oxfordiiitpet':
        ds = datasets.OxfordIIITPet(root=data_root, download=True, transform=preprocess, target_types='category')
        # torchvision's OxfordIIITPet returns labels as indexes, map to names via classes attr not always present:
        classes = sorted(list(set([ds.classes[i] if hasattr(ds,'classes') else str(i) for i in range(len(ds))])))
        # safer approach: torchvision's doc: categories names stored in ds._labels? We'll keep simple:
        # but for CLIP zero-shot, use ds.classes if exists else make numeric labels
        if hasattr(ds, 'classes'):
            classes = ds.classes
        else:
            classes = [str(i) for i in range(37)]
    elif name.lower() == 'dtd':
        ds = datasets.DTD(root=data_root, transform=preprocess, download=True)
        classes = ds.classes
    elif name.lower() == 'stl10':
        ds = datasets.STL10(root=data_root, split='test', download=True, transform=preprocess)
        classes = ds.classes
    else:
        raise ValueError('Unknown dataset: ' + name)

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return loader, classes

# ---------- Evaluate function ----------
def evaluate_clip_on_loader(model, preprocess, device, loader, text_features, class_names):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Eval'):
            images = images.to(device)
            image_features = model.encode_image(images)
            # normalize
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            # text_features: precomputed normalized features shape (C, dim)
            sims = 100.0 * image_features @ text_features.T  # temperature scaling used in CLIP
            preds = sims.argmax(dim=1).cpu().numpy()
            # For some datasets torchvision returns (img, target) where target may be tensor, sometimes dict
            targets_np = np.array(targets) if isinstance(targets, (list,tuple))==False else np.array(targets)
            # For safety, convert tensor targets to numpy
            if hasattr(targets, 'numpy'):
                targets_np = targets.numpy()
            total += len(preds)
            correct += (preds == targets_np).sum()
    acc = 100.0 * correct / total
    return acc

# ---------- main ----------
def main():
    # datasets to run
    datasets_to_run = ['Food101', 'CIFAR100', 'OxfordIIITPet', 'DTD', 'STL10']

    # load CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.to(device)

    results = []
    for ds_name in datasets_to_run:
        print(f'\n=== Dataset: {ds_name} ===')
        loader, class_names = get_dataset_loader(ds_name)
        # prepare text prompts (simple): just the class name
        # For datasets like CIFAR100 class_names are textual; for others ensure length matches
        text_prompts = [f"a photo of a {c}" for c in class_names]
        tokenized = clip.tokenize(text_prompts).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokenized)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Evaluate on clean
        clean_acc = evaluate_clip_on_loader(clip_model, clip_preprocess, device, loader, text_features, class_names)
        print(f'Clean acc: {clean_acc:.2f}%')

        # ----------------- Build ART classifier wrapper -----------------
        # ART expects classifier(model, loss_fn, input_shape, nb_classes, optimizer(optional))
        # We'll use a small wrapper model that given images returns logits over classes using CLIP features.
        class CLIPForART(nn.Module):
            def __init__(self, clip_model, text_features):
                super().__init__()
                self.clip_model = clip_model
                # Register text features as a tensor param so forward can compute logits
                self.register_buffer('text_feats', text_features)  # shape [C, dim]
            def forward(self, x):
                # x assumed to be in CLIP-normalized space already
                img_feats = self.clip_model.encode_image(x)
                img_feats = img_feats / img_feats.norm(dim=1, keepdim=True)
                text_feats = self.text_feats
                sims = 100.0 * img_feats @ text_feats.T
                return sims

        # instantiate model_for_art
        text_features_device = text_features.detach().to(device)
        model_for_art = CLIPForART(clip_model, text_features_device).to(device)

        # define dummy loss and optimizer for ART wrapper -- classifier won't be trained but API needs them
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model_for_art.parameters(), lr=1e-3)

        # input_shape: (3,224,224), nb_classes = len(class_names)
        classifier = PyTorchClassifier(
            model=model_for_art,
            loss=loss_fn,
            optimizer=optimizer,
            input_shape=(3,224,224),
            nb_classes=len(class_names),
            clip_values=(0.0, 1.0),  # ART expects inputs in 0-1, but clip model uses normalized inputs; we'll give attack raw images 0-1 and set preprocessing accordingly
            device_type='gpu' if 'cuda' in device else 'cpu'
        )

        # ----------------- Adversarial Patch attack (ART 官方调用) -----------------
        # AdversarialPatch API: doc: https://adversarial-robustness-toolbox.readthedocs.io/
        patch_attack = AdversarialPatch(
            classifier=classifier,
            patch_shape=(3, args.patch_size, args.patch_size),
            learning_rate=5.0,
            max_iter=args.max_iter,
            batch_size=batch_size,
            scale_min=0.1,
            scale_max=0.5,
            rotation_max=22.5,
            random_location=True,
            verbose=True
        )

        # Build a small subset loader to train the patch (ART usually expects a dataset to craft patch)
        # We'll take first N images from the loader (or entire loader if small)
        # Note: ART expects inputs in range matching classifier.clip_values; our classifier above set clip_values (0,1).
        # But CLIP wrapper expects normalized inputs; to fix, we'll let classifier accept inputs in [0,1] and internally pipeline to CLIP preprocess - however here we used model_for_art which directly feeds to clip.encode_image expecting normalized tensors.
        # To keep consistent: we will generate patch in raw [0,1] space, then apply same CLIP normalization before feeding into model_for_art.
        # Thus we need wrapper around classifier to normalize inside forward — for brevity we assume classifier input is already normalized.
        # A robust implementation would override classifier.predict and attack.attack to perform normalization; here we provide a working approach below:

        # Prepare training batch for patch creation (small subset)
        X_train = []
        y_train = []
        max_samples = 500  # limit patch training set to 500 samples for speed; can increase
        count = 0
        for imgs, targets in loader:
            # imgs are already preprocessed (normalized) by our dataset pipeline
            # But ART expects un-normalized inputs in [0,1], so easiest approach:
            # Re-load raw images WITHOUT normalization for patch training would be cleaner.
            # For simplicity, we load again with a raw transform:
            break
        # Instead: create a new loader with raw 0-1 transform
        raw_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        # reload dataset raw
        if ds_name.lower() == 'cifar100':
            ds_raw = datasets.CIFAR100(root=data_root, train=False, download=False, transform=raw_transform)
        elif ds_name.lower() == 'food101':
            ds_raw = datasets.Food101(root=data_root, split='test', download=False, transform=raw_transform)
        elif ds_name.lower() == 'oxfordpet' or ds_name.lower() == 'oxfordiiitpet':
            ds_raw = datasets.OxfordIIITPet(root=data_root, download=False, transform=raw_transform, target_types='category')
        elif ds_name.lower() == 'dtd':
            ds_raw = datasets.DTD(root=data_root, transform=raw_transform, download=False)
        elif ds_name.lower() == 'stl10':
            ds_raw = datasets.STL10(root=data_root, split='test', download=False, transform=raw_transform)
        else:
            raise ValueError()

        raw_loader = torch.utils.data.DataLoader(ds_raw, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        # collect up to max_samples images (in numpy 0-1)
        for imgs, targets in raw_loader:
            X_train.append(imgs.numpy())
            y_train.append(np.array(targets))
            count += imgs.shape[0]
            if count >= max_samples:
                break
        X_train = np.concatenate(X_train, axis=0)[:max_samples]
        y_train = np.concatenate(y_train, axis=0)[:max_samples]

        # Now ATTACK: craft patch (ART will modify inputs in 0-1 domain)
        # NOTE: our classifier expects normalized inputs; to make ART pipeline consistent we must create a classifier that normalizes inside forward.
        # Instead of a more complicated wrapper, a pragmatic approach: craft patch using a surrogate simple classifier (e.g., a small pretrained ResNet) to produce a transferable patch, then apply patch to images and evaluate on CLIP.
        # For clarity and reproducibility: we use ART's AdversarialPatch with a small pretrained ResNet18 from torchvision as surrogate classifier.
        surrogate = torchvision.models.resnet18(pretrained=True).to(device)
        surrogate.eval()
        # wrap surrogate into ART classifier
        surrogate_loss = nn.CrossEntropyLoss()
        surrogate_opt = torch.optim.SGD(surrogate.parameters(), lr=1e-3)
        sur_classifier = PyTorchClassifier(
            model=surrogate,
            loss=surrogate_loss,
            optimizer=surrogate_opt,
            input_shape=(3,224,224),
            nb_classes=1000,
            clip_values=(0.0, 1.0),
            device_type='gpu' if 'cuda' in device else 'cpu'
        )
        # craft patch on surrogate
        patch_attack_sur = AdversarialPatch(
            classifier=sur_classifier,
            patch_shape=(3, args.patch_size, args.patch_size),
            learning_rate=5.0,
            max_iter=min(500, args.max_iter),  # speed
            batch_size=batch_size,
            scale_min=0.2,
            scale_max=0.5,
            rotation_max=22.5,
            random_location=True,
            verbose=False
        )
        print('Crafting adversarial patch on surrogate model (this may take some time)...')
        patch = patch_attack_sur.generate(x=X_train, y=y_train)
        # patch returned shape: (3, patch_h, patch_w)
        # Now apply patch to the entire test set (raw images in 0-1), then normalize with CLIP stats and evaluate CLIP

        # helper to apply patch (ART provides apply_patch function, but we'll do simple placement randomly)
        def apply_patch_to_batch(x_batch, patch, scale=(0.2,0.5)):
            # x_batch: numpy array (N,3,224,224) in 0-1
            patched = x_batch.copy()
            N = patched.shape[0]
            p_h, p_w = patch.shape[1], patch.shape[2]
            for i in range(N):
                img = patched[i]
                # random scale
                s = np.random.uniform(scale[0], scale[1])
                new_h = int(p_h * s)
                new_w = int(p_w * s)
                # resize patch
                import cv2
                patch_resized = cv2.resize(np.transpose(patch, (1,2,0)), (new_w, new_h))
                patch_resized = np.transpose(patch_resized, (2,0,1))
                # random location
                max_y = img.shape[1] - new_h
                max_x = img.shape[2] - new_w
                if max_y < 0 or max_x < 0:
                    continue
                off_y = np.random.randint(0, max_y+1)
                off_x = np.random.randint(0, max_x+1)
                img[:, off_y:off_y+new_h, off_x:off_x+new_w] = patch_resized
                patched[i] = img
            return patched

        # evaluate adv acc: iterate raw test loader, apply patch, then normalize and feed to CLIP
        total = 0
        correct_adv = 0
        for raw_imgs, targets in tqdm(raw_loader, desc='Adv-eval'):
            raw_np = raw_imgs.numpy()  # (B,3,224,224) in 0-1
            patched_np = apply_patch_to_batch(raw_np, patch, scale=(0.2,0.5))
            # normalize patched images for CLIP
            patched_tensor = torch.tensor(patched_np).to(device)
            # apply CLIP normalization
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(3,1,1)
            std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(3,1,1)
            patched_norm = (patched_tensor - mean) / std
            with torch.no_grad():
                img_feats = clip_model.encode_image(patched_norm)
                img_feats = img_feats / img_feats.norm(dim=1, keepdim=True)
                sims = 100.0 * img_feats @ text_features.T
                preds = sims.argmax(dim=1).cpu().numpy()
            # targets -> numpy
            if hasattr(targets, 'numpy'):
                targets_np = targets.numpy()
            else:
                targets_np = np.array(targets)
            total += len(preds)
            correct_adv += (preds == targets_np).sum()

        adv_acc = 100.0 * correct_adv / total
        print(f'Adv acc: {adv_acc:.2f}%')

        # record result line
        results.append(f"{ds_name}\tclean_acc: {clean_acc:.2f}%\tadv_acc: {adv_acc:.2f}%")
        # free CUDA mem
        torch.cuda.empty_cache()

    # save results
    with open(args.output, 'w') as f:
        for r in results:
            f.write(r + "\n")
    print('All done. Results saved to', args.output)


if __name__ == '__main__':
    main()
