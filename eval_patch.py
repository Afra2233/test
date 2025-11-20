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


# ------------------------------------------------
# apply patch using PyTorch
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
# class list resolver for each dataset
# ------------------------------------------------
def get_class_list(name, ds):
    """Return class names for each dataset"""

    # CIFAR10 — OK
    if name == "cifar10":
        return ds.classes

    # Flowers102 — torchvision does NOT provide class names
    if name == "flowers102":
        return [f"a flower class {i}" for i in range(102)]

    # DTD — extract class names from label files
    if name == "dtd":
        label_dir = os.path.join(ds.root, "dtd", "labels")
        files = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])
        class_names = [f[:-4] for f in files]  # remove .txt
        return class_names

    # Pets — torchvision uses internal attribute "_classes"
    if name == "pets":
        return ds.classes

    # Food101 — OK
    if name == "food101":
        return ds.classes

    raise RuntimeError(f"[FATAL] Unknown dataset: {name}")


# ------------------------------------------------
# build CLIP text features
# ------------------------------------------------
def build_text_features(class_names, clip_model, device):
    prompts = [f"a photo of a {c.replace('_',' ')}" for c in class_names]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats


# ------------------------------------------------
# Evaluate dataset
# ------------------------------------------------
def evaluate_dataset(name, ds, clip_model, device, patch_np, text_features):
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    total = 0
    clean_correct = 0
    patch_correct = 0

    print(f"\n===== Evaluating {name} =====")

    # tqdm 自动显示 ETA
    for images, labels in tqdm(loader, desc=f"{name}", ncols=120):
        images = images.to(device)
        labels = labels.to(device)

        # ---------- clean ----------
        with torch.no_grad():
            f = clip_model.encode_image(images)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = 100 * f @ text_features.T
        clean_correct += (logits.argmax(1) == labels).sum().item()

        # ---------- patched ----------
        patched = apply_patch_torch(images, patch_np, device)
        with torch.no_grad():
            f2 = clip_model.encode_image(patched)
            f2 = f2 / f2.norm(dim=-1, keepdim=True)
            logits2 = 100 * f2 @ text_features.T
        patch_correct += (logits2.argmax(1) == labels).sum().item()

        total += labels.size(0)

    print(f"{name}: clean={clean_correct/total:.4f} | patched={patch_correct/total:.4f}\n")


# ======================================================
# main()
# ======================================================
def main():
    print("[DEBUG] main() started", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    # Load CLIP
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    print("[DEBUG] CLIP loaded", flush=True)

    # Load patch
    patch_np = np.load("artifacts/universal_patch.npy")
    print("[DEBUG] Patch loaded", flush=True)

    # Common transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    DATA_ROOT = "data"
    print("[DEBUG] Loading datasets ...", flush=True)

    # All datasets — no download to avoid stuck
    datasets = {
        "cifar10": CIFAR10(f"{DATA_ROOT}/cifar10", train=False, download=True, transform=transform),
        "flowers102": Flowers102(f"{DATA_ROOT}/flowers", split="test", download=True, transform=transform),
        # "dtd": DTD(f"{DATA_ROOT}/dtd", split="test", download=True, transform=transform),
        "pets": OxfordIIITPet(f"{DATA_ROOT}/pets", split="test", download=True, transform=transform),
        "food101": Food101(f"{DATA_ROOT}/food", split="test", download=True, transform=transform),
    }

    print("[DEBUG] All datasets loaded", flush=True)
    print("[DEBUG] Starting evaluation ...", flush=True)

    # Evaluate each dataset
    for name, ds in datasets.items():
        print(f"[DEBUG] Preparing text features for {name}", flush=True)
        class_names = get_class_list(name, ds)
        text_features = build_text_features(class_names, clip_model, device)

        print(f"[DEBUG] Evaluating {name}", flush=True)
        evaluate_dataset(name, ds, clip_model, device, patch_np, text_features)


if __name__ == "__main__":
    main()
