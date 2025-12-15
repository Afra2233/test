import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, Flowers102, Food101, OxfordIIITPet, STL10, FGVCAircraft
from torch.utils.data import DataLoader
import clip
from autoattack import AutoAttack


# =========================================================
# Dataset Class Name Resolution
# =========================================================
def get_class_list(name, ds):
    if name in ["cifar10", "cifar100"]:
        return ds.classes
    if name == "flowers102":
        return [f"a flower class {i}" for i in range(102)]
    if name in ["pets", "food101", "fgvc_aircraft", "stl10"]:
        return ds.classes
    raise RuntimeError(f"[FATAL] Unknown dataset: {name}")


# =========================================================
# Build CLIP Text Features
# =========================================================
def build_text_features(class_names, clip_model, device):
    prompts = [f"a photo of a {c.replace('_',' ')}" for c in class_names]
    tokens = clip.tokenize(prompts).to(device)

    # 建议这里用 no_grad，省显存/更快；不影响 AutoAttack（attack 时需要梯度的是 image 这条路）
    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    return text_feats


# =========================================================
# CLIP Wrapper for AutoAttack (attack in pixel space [0,1])
# =========================================================
class CLIPClassifier(torch.nn.Module):
    def __init__(self, clip_model, text_features, device):
        super().__init__()
        self.clip_model = clip_model
        self.text_features = text_features.to(device)

        # CLIP normalize buffer (expects input in [0,1])
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean.to(device))
        self.register_buffer("std", std.to(device))

    def forward(self, x):
        # x should be in [0,1] for Linf eps to make sense
        x = (x - self.mean) / self.std

        f = self.clip_model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        logits = 100 * f @ self.text_features.T
        return logits


# =========================================================
# Evaluation with Square Attack
# =========================================================
def evaluate_dataset(name, ds, clip_model, device, text_features, eps=8/255, bs_aa=64):

    print(f"\n===== Evaluating {name} with AutoAttack Square =====")

    loader = DataLoader(ds, batch_size=bs_aa, shuffle=False, num_workers=4, pin_memory=True)
    xs, ys = [], []

    for images, labels in tqdm(loader, desc=f"{name}-collect", ncols=120):
        xs.append(images)
        ys.append(labels)

    x_test = torch.cat(xs).to(device, non_blocking=True)
    y_test = torch.cat(ys).to(device, non_blocking=True)

    print(f"[DEBUG] {name}: dataset size = {x_test.size(0)}")

    # Debug mode: only attack first 1000 images
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    print("[DEBUG] Using only first 1000 images to verify attack runs.\n")

    # ---- Critical: force fp32, disable AMP ----
    x_test = x_test.float()
    y_test = y_test.long()

    model_aa = CLIPClassifier(clip_model, text_features, device).to(device)
    model_aa.eval()
    model_aa.float()

    # AutoAttack
    adversary = AutoAttack(
        model_aa,
        norm='Linf',
        eps=eps,
        version='custom',
        verbose=True
    )
    adversary.attacks_to_run = ['square']

    print("[DEBUG] Running attack... (AMP disabled, fp32)")
    if device == "cuda":
        with torch.cuda.amp.autocast(False):
            x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=bs_aa)
    else:
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=bs_aa)

    # ---- Metrics ----
    with torch.no_grad():
        logits_clean = model_aa(x_test)
        logits_adv = model_aa(x_adv)

    preds_clean = logits_clean.argmax(1)
    preds_adv = logits_adv.argmax(1)

    clean_acc = (preds_clean == y_test).float().mean().item()
    adv_acc = (preds_adv == y_test).float().mean().item()

    clean_mask = preds_clean == y_test
    asr = ((preds_adv != y_test) & clean_mask).sum().item() / (clean_mask.sum().item() + 1e-12)

    print(f"\nRESULT: {name}")
    print(f"Clean Accuracy:   {clean_acc:.4f}")
    print(f"Robust Accuracy:  {adv_acc:.4f}")
    print(f"ASR:              {asr:.4f}")
    print("=====================================================\n")

    return clean_acc, adv_acc, asr


# =========================================================
# Main
# =========================================================
def main():

    print("[DEBUG] Starting evaluation script...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] Device: {device}")

    # Load CLIP (force fp32)
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    clip_model = clip_model.float()

    # IMPORTANT: dataset transform should output [0,1] tensor; normalize moved into model wrapper
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    DATA_ROOT = "data"

    datasets = {
        "cifar10": CIFAR10(f"{DATA_ROOT}/cifar10", train=False, download=True, transform=transform),
        "cifar100": CIFAR100(f"{DATA_ROOT}/cifar100", train=False, download=True, transform=transform),
        "food101": Food101(f"{DATA_ROOT}/food", split="test", download=True, transform=transform),
        "pets": OxfordIIITPet(f"{DATA_ROOT}/pets", split="test", download=True, transform=transform),
        "fgvc_aircraft": FGVCAircraft(f"{DATA_ROOT}/fgvc_aircraft", split="test", download=True, transform=transform),
        "stl10": STL10(f"{DATA_ROOT}/stl10", split="test", download=True, transform=transform),
    }

    for name, ds in datasets.items():
        print(f"\nPreparing: {name}")
        class_names = get_class_list(name, ds)
        text_features = build_text_features(class_names, clip_model, device)

        # eps in pixel space now (because x in [0,1])
        evaluate_dataset(name, ds, clip_model, device, text_features, eps=1/255, bs_aa=16)


if __name__ == "__main__":
    print("start")
    main()

