# ================================================
# eval_patch.py — 显示每个数据集进度 + ETA（完全稳定版）
# ================================================

# ================================================
# eval_patch.py — 显示每个数据集进度 + 剩余时间
# ================================================
import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CIFAR10, Flowers102, Food101, DTD, OxfordIIITPet
from torch.utils.data import DataLoader
import clip

DATA_ROOT = "./data"   # 自动使用当前目录下的 data 文件夹


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
    images[:, :, y0:y0+h, x0:x0+w] = patch
    return images


# ------------------------------------------------
# Safe class list extraction
# ------------------------------------------------
def get_class_list(ds):
    """Different datasets store labels differently."""
    if hasattr(ds, "classes"):
        return ds.classes
    if hasattr(ds, "labels"):
        return ds.labels
    if hasattr(ds, "categories"):
        return ds.categories

    # fallback
    raise RuntimeError("Dataset has no class list attribute.")


# ------------------------------------------------
# build CLIP zero-shot features
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
    total = 0
    clean_correct = 0
    patch_correct = 0

    print(f"\n===== Evaluating {name} =====", flush=True)

    for images, labels in tqdm(loader, desc=f"{name}", ncols=120):
        images, labels = images.to(device), labels.to(device)

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

    print(f"{name} | clean={clean_correct/total:.4f} | patched={patch_correct/total:.4f}\n", flush=True)


# ======================================================
# main()
# ======================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    patch_np = np.load("artifacts/universal_patch.npy")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # load datasets safely
    datasets = {}
    dataset_defs = {
        "cifar10": lambda: CIFAR10(f"{DATA_ROOT}/cifar10", train=False, download=False, transform=transform),
        "flowers102": lambda: Flowers102(f"{DATA_ROOT}/flowers", split="test", download=False, transform=transform),
        "dtd": lambda: DTD(f"{DATA_ROOT}/dtd", split="test", download=False, transform=transform),
        "pets": lambda: OxfordIIITPet(f"{DATA_ROOT}/pets", split="test", download=False, transform=transform),
        "food101": lambda: Food101(f"{DATA_ROOT}/food", split="test", download=False, transform=transform),
    }

    print("[DEBUG] dataset loading started", flush=True)
    for name, fn in dataset_defs.items():
        try:
            datasets[name] = fn()
            print(f"[DEBUG] {name} loaded", flush=True)
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}", flush=True)

    print("[DEBUG] dataset loading finished", flush=True)

    # evaluate all loaded datasets
    for name, ds in datasets.items():
        print("[DEBUG] dataload started", flush=True)
        class_names = get_class_list(ds)
        text_features = build_text_features(class_names, clip_model, device)
        print("[DEBUG] evaluate started", flush=True)
        evaluate_dataset(name, ds, clip_model, device, patch_np, text_features)


if __name__ == "__main__":
    main()
