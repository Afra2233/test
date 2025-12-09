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

    if name == "pets":
        return ds.classes

    if name == "food101":
        return ds.classes

    if name == "fgvc_aircraft":
        return ds.classes

    if name == "stl10":
        return ds.classes

    raise RuntimeError(f"[FATAL] Unknown dataset: {name}")


# =========================================================
# Build CLIP Text Features
# =========================================================
def build_text_features(class_names, clip_model, device):
    prompts = [f"a photo of a {c.replace('_',' ')}" for c in class_names]
    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    return text_feats


# =========================================================
# CLIP Wrapper for AutoAttack
# =========================================================
class CLIPClassifier(torch.nn.Module):
    def __init__(self, clip_model, text_features, device):
        super().__init__()
        self.clip_model = clip_model
        self.text_features = text_features.to(device)

    def forward(self, x):
        # ❗ AutoAttack must see a normal forward (NO no_grad)
        f = self.clip_model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        logits = 100 * f @ self.text_features.T
        return logits


# =========================================================
# Evaluation with Square Attack
# =========================================================
def evaluate_dataset(name, ds, clip_model, device, text_features, eps=8/255, bs_aa=64):

    print(f"\n===== Evaluating {name} with AutoAttack Square =====")

    # Load entire dataset into memory
    loader = DataLoader(ds, batch_size=bs_aa, shuffle=False)
    xs, ys = [], []

    for images, labels in tqdm(loader, desc=f"{name}-collect", ncols=120):
        xs.append(images)
        ys.append(labels)

    x_test = torch.cat(xs).to(device)
    y_test = torch.cat(ys).to(device)

    print(f"[DEBUG] {name}: dataset size = {x_test.size(0)}")

    # ⚠ Debug mode: only attack first 1000 images to verify it runs
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    print("[DEBUG] Using only first 1000 images to verify attack runs.\n")

    # Wrap CLIP for AutoAttack
    model_aa = CLIPClassifier(clip_model, text_features, device).to(device)

    adversary = AutoAttack(
        model_aa,
        norm='Linf',
        eps=eps,
        version='custom',
        verbose=True        # 👈 NOW you will see attack progress!
    )
    adversary.attacks_to_run = ['square']

    print("[DEBUG] Running attack...")
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=bs_aa)

    # ---- Metrics ----
    logits_clean = model_aa(x_test)
    logits_adv = model_aa(x_adv)

    preds_clean = logits_clean.argmax(1)
    preds_adv = logits_adv.argmax(1)

    clean_acc = (preds_clean == y_test).float().mean().item()
    adv_acc = (preds_adv == y_test).float().mean().item()

    clean_mask = preds_clean == y_test
    asr = ((preds_adv != y_test) & clean_mask).sum().item() / clean_mask.sum().item()

    print(f"\nRESULT — {name}")
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

    # Load CLIP
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
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

        evaluate_dataset(name, ds, clip_model, device, text_features, eps=8/255)


if __name__ == "__main__":
    main()
