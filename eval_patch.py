# ================================================
# eval_patch.py — 使用 GPU + PyTorch 评估 patch
# ================================================
import os
import torch
import numpy as np

from torchvision import transforms
from torchvision.datasets import CIFAR10, Flowers102, Food101, DTD, OxfordIIITPet
from torch.utils.data import DataLoader

import clip


# ------------------------------------------------
# apply patch (PyTorch only)
# ------------------------------------------------
def apply_patch_torch(images, patch_np, device):
    patch = torch.from_numpy(patch_np).float().to(device)

    B, C, H, W = images.shape
    _, h, w = patch.shape
    y0 = H - h
    x0 = W - w

    images = images.clone()
    images[:, :, y0:y0 + h, x0:x0 + w] = patch
    return images


# ------------------------------------------------
# build CLIP zero-shot text features
# ------------------------------------------------
def build_text_features(class_names, clip_model, device):
    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in class_names]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        txt = clip_model.encode_text(tokens)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt


# ------------------------------------------------
# Evaluate a dataset
# ------------------------------------------------
def evaluate_dataset(name, dataset, clip_model, device, patch_np, text_features):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    total, clean_correct, patch_correct = 0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # clean
        with torch.no_grad():
            f = clip_model.encode_image(images)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = 100 * f @ text_features.T
        clean_correct += (logits.argmax(1) == labels).sum().item()

        # patched
        patched = apply_patch_torch(images, patch_np, device)
        with torch.no_grad():
            f2 = clip_model.encode_image(patched)
            f2 = f2 / f2.norm(dim=-1, keepdim=True)
            logits2 = 100 * f2 @ text_features.T
        patch_correct += (logits2.argmax(1) == labels).sum().item()

        total += labels.size(0)

    print(f"{name}: clean={clean_correct/total:.4f}, patched={patch_correct/total:.4f}")


# ======================================================
# main()
# ======================================================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # load CLIP
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    # load patch
    patch_np = np.load("artifacts/universal_patch.npy")

    # transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # test datasets
    datasets = {
        "cifar10": CIFAR10("data/cifar10", train=False, download=True, transform=transform),
        "flowers102": Flowers102("data/flowers", split="test", download=True, transform=transform),
        "dtd": DTD("data/dtd", split="test", download=True, transform=transform),
        "pets": OxfordIIITPet("data/pets", split="test", download=True, transform=transform),
        "food101": Food101("data/food", split="test", download=True, transform=transform)
    }

    # evaluate
    for name, ds in datasets.items():
        text_features = build_text_features(ds.classes, clip_model, device)
        evaluate_dataset(name, ds, clip_model, device, patch_np, text_features)


if __name__ == "__main__":
    main()
