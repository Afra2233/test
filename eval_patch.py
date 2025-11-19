# ================================================
# eval_patch.py — 只显示「每个数据集进度 + 剩余时间」
# ================================================
import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CIFAR10, Flowers102, Food101, DTD, OxfordIIITPet
from torch.utils.data import DataLoader
import clip


def apply_patch_torch(images, patch_np, device):
    patch = torch.from_numpy(patch_np).float().to(device)
    B, C, H, W = images.shape
    _, h, w = patch.shape
    y0 = H - h
    x0 = W - w
    images = images.clone()
    images[:, :, y0:y0+h, x0:x0+w] = patch
    return images


def build_text_features(classes, clip_model, device):
    prompts = [f"a photo of a {c.replace('_',' ')}" for c in classes]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        txt = clip_model.encode_text(tokens)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt


def evaluate_dataset(name, dataset, clip_model, device, patch_np, text_features):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    total = 0
    clean_correct = 0
    patch_correct = 0

    print(f"\n===== Evaluating {name} =====")

    # 只需要 tqdm，就能自动显示剩余时间
    for images, labels in tqdm(loader, desc=f"{name}", ncols=120):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            f_clean = clip_model.encode_image(images)
            f_clean = f_clean / f_clean.norm(dim=-1, keepdim=True)
            logits_clean = 100 * f_clean @ text_features.T
        clean_correct += (logits_clean.argmax(1) == labels).sum().item()

        patched = apply_patch_torch(images, patch_np, device)
        with torch.no_grad():
            f_p = clip_model.encode_image(patched)
            f_p = f_p / f_p.norm(dim=-1, keepdim=True)
            logits_p = 100 * f_p @ text_features.T
        patch_correct += (logits_p.argmax(1) == labels).sum().item()

        total += labels.size(0)

    print(f"{name} | clean={clean_correct/total:.4f} | patched={patch_correct/total:.4f}\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("[DEBUG] enter main function", flush=True)

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
 
    print("[DEBUG] load clip", flush=True)
    patch_np = np.load("artifacts/universal_patch.npy")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    print("[DEBUG] dataset load", flush=True)
    # datasets = {
    #     "cifar10": CIFAR10("data/cifar10", train=False, download=True, transform=transform),
    #     "flowers102": Flowers102("data/flowers", split="test", download=True, transform=transform),
    #     "dtd": DTD("data/dtd", split="test", download=True, transform=transform),
    #     "pets": OxfordIIITPet("data/pets", split="test", download=True, transform=transform),
    #     "food101": Food101("data/food", split="test", download=True, transform=transform),
    # }
    print("[DEBUG] start loading datasets...", flush=True)

    datasets = {}

    # ----- CIFAR10 -----
    try:
        print("[DEBUG] loading CIFAR10 ...", flush=True)
        datasets["cifar10"] = CIFAR10("data/cifar10", train=False, download=True, transform=transform)
        print("[DEBUG] CIFAR10 loaded", flush=True)
    except Exception as e:
        print("[ERROR] CIFAR10 failed:", e, flush=True)


    # ----- Flowers102 -----
    try:
        print("[DEBUG] loading Flowers102 ...", flush=True)
        datasets["flowers102"] = Flowers102("data/flowers", split="test", download=True, transform=transform)
        print("[DEBUG] Flowers102 loaded", flush=True)
    except Exception as e:
        print("[ERROR] Flowers102 failed:", e, flush=True)


    # ----- DTD -----
    try:
        print("[DEBUG] loading DTD ...", flush=True)
        datasets["dtd"] = DTD("data/dtd", split="test", download=True, transform=transform)
        print("[DEBUG] DTD loaded", flush=True)
    except Exception as e:
        print("[ERROR] DTD failed:", e, flush=True)


    # ----- Pets -----
    try:
        print("[DEBUG] loading Pets ...", flush=True)
        datasets["pets"] = OxfordIIITPet("data/pets", split="test", download=True, transform=transform)
        print("[DEBUG] Pets loaded", flush=True)
    except Exception as e:
        print("[ERROR] Pets failed:", e, flush=True)


    # ----- Food101 -----
    try:
        print("[DEBUG] loading Food101 ...", flush=True)
        datasets["food101"] = Food101("data/food", split="test", download=False, transform=transform)
        print("[DEBUG] Food101 loaded", flush=True)
    except Exception as e:
        print("[ERROR] Food101 failed:", e, flush=True)

    print("[DEBUG] dataset loading finished", flush=True)

    print("[DEBUG] program started", flush=True)

    for name, ds in datasets.items():
        print("[DEBUG] dataload started", flush=True)
        text_features = build_text_features(ds.classes, clip_model, device)
        print("[DEBUG] evaluate started", flush=True)
        evaluate_dataset(name, ds, clip_model, device, patch_np, text_features)


if __name__ == "__main__":
    print("[DEBUG] main started", flush=True)
    main()
