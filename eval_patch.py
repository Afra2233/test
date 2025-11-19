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


# ================================================
# eval_patch.py — 显示每个数据集进度 + ETA（完全稳定版）
# ================================================

import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CIFAR10, Flowers102, Food101, DTD, OxfordIIITPet
from torch.utils.data import DataLoader
import clip


# ------------------------------------------------
# Apply adversarial patch
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
# Build zero-shot CLIP text embeddings
# ------------------------------------------------
def build_text_features(class_names, clip_model, device):
    prompts = [f"a photo of a {c.replace('_',' ')}" for c in class_names]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        txt = clip_model.encode_text(tokens)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt


# ------------------------------------------------
# Unified class-name extraction
# ------------------------------------------------
def get_classes(ds):
    """Return string class names for any of the supported datasets."""
    # CIFAR10
    if hasattr(ds, "classes"):
        return ds.classes

    # Flowers102: 没有 classes, 所以造 102 个名字
    if isinstance(ds, Flowers102):
        return [f"class_{i:05d}" for i in range(1, 103)]

    # DTD 提供 labels 列表
    if isinstance(ds, DTD):
        return ds.labels

    # Pets dataset
    if isinstance(ds, OxfordIIITPet):
        return ds.classes

    # Food101 — class folders under images/
    if isinstance(ds, Food101):
        base = os.path.join(ds.root, "food-101", "images")
        return sorted(os.listdir(base))

    raise RuntimeError(f"Unsupported dataset type: {type(ds)}")


# ------------------------------------------------
# Evaluate dataset
# ------------------------------------------------
def evaluate_dataset(name, dataset, clip_model, device, patch_np, text_features):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    total = 0
    clean_correct = 0
    patch_correct = 0

    print(f"\n===== Evaluating {name} =====", flush=True)

    for images, labels in tqdm(loader, desc=name, ncols=100):
        images = images.to(device)
        labels = labels.to(device)

        # clean
        with torch.no_grad():
            f_clean = clip_model.encode_image(images)
            f_clean = f_clean / f_clean.norm(dim=-1, keepdim=True)
            logits_clean = 100 * f_clean @ text_features.T
        clean_correct += (logits_clean.argmax(1) == labels).sum().item()

        # patched
        patched = apply_patch_torch(images, patch_np, device)
        with torch.no_grad():
            f_p = clip_model.encode_image(patched)
            f_p = f_p / f_p.norm(dim=-1, keepdim=True)
            logits_p = 100 * f_p @ text_features.T
        patch_correct += (logits_p.argmax(1) == labels).sum().item()

        total += labels.size(0)

    print(f"{name} | clean={clean_correct/total:.4f} | patched={patch_correct/total:.4f}\n",
          flush=True)


# ======================================================
# main()
# ======================================================
def main():

    print("[DEBUG] main started", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    # load CLIP
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    print("[DEBUG] CLIP loaded", flush=True)

    # universal patch
    patch_np = np.load("artifacts/universal_patch.npy")

    # transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    print("[DEBUG] dataset loading ...", flush=True)

    # IMPORTANT: use absolute paths !!
    DATA_ROOT = "/storage/hpc/07/zhang303/conda_envs/stadv/data"

    datasets = {
        "cifar10": CIFAR10(f"{DATA_ROOT}/cifar10", train=False, download=False, transform=transform),
        "flowers102": Flowers102(f"{DATA_ROOT}/flowers", split="test", download=False, transform=transform),
        "dtd": DTD(f"{DATA_ROOT}/dtd", split="test", download=False, transform=transform),
        "pets": OxfordIIITPet(f"{DATA_ROOT}/pets", split="test", download=False, transform=transform),
        "food101": Food101(f"{DATA_ROOT}/food", split="test", download=False, transform=transform),
    }

    print("[DEBUG] dataset loading finished", flush=True)

    # Evaluate each dataset
    print("[DEBUG] evaluation loop started", flush=True)
    for name, ds in datasets.items():
        class_names = get_classes(ds)
        text_features = build_text_features(class_names, clip_model, device)
        evaluate_dataset(name, ds, clip_model, device, patch_np, text_features)


if __name__ == "__main__":
    main()
