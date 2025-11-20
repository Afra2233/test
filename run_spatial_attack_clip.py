import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import clip

from advertorch.attacks import SpatialTransformAttack

from torchvision.datasets import CIFAR10, CIFAR100, Flowers102, Food101, OxfordIIITPet
from torchvision.datasets import STL10
from torchvision.datasets import FGVCAircraft


##############################################
# CLASS LIST UTILS
##############################################

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
    raise RuntimeError(f"Unknown dataset: {name}")


##############################################
# TEXT FEATURES
##############################################

def build_text_features(class_names, clip_model, device):
    prompts = [f"a photo of a {c.replace('_',' ')}" for c in class_names]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats


##############################################
# EVALUATE ADVERTORCH SPATIAL ATTACK
##############################################

def evaluate_spatial_attack(name, ds, clip_model, device, text_features):
    print(f"\n===== Evaluating {name} (Advertorch Spatial Attack) =====")

    loader = DataLoader(ds, batch_size=16, shuffle=False)

    total = 0
    clean_correct = 0
    adv_correct = 0
    attack_success = 0
    clean_total = 0

    # Build the attacker
    # max_rotation: degrees
    # max_translation: fraction of image (0.125 means 12.5% shift)
    from advertorch.attacks import ElasticTransformAttack

    attacker = ElasticTransformAttack(
        predict=forward_fn,
        loss_fn=torch.nn.CrossEntropyLoss(),
        distortion=0.3,     # 扭曲强度，可调
        sigma=4,            # 高斯核大小，可调
    )

    for images, labels in tqdm(loader, desc=name):
        images = images.to(device)
        labels = labels.to(device)

        #########################################
        # CLEAN PRED
        #########################################
        with torch.no_grad():
            f = clip_model.encode_image(images)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = 100 * f @ text_features.T

        preds_clean = logits.argmax(1)
        clean_correct_batch = (preds_clean == labels)
        clean_correct += clean_correct_batch.sum().item()
        clean_total += clean_correct_batch.sum().item()

        #########################################
        # SPATIAL ADVERSARIAL ATTACK
        #########################################
        # advertorch requires logits; CLIP image encoder outputs features
        # So we compute logits via text_features manually
        # We'll define a wrapper to compute logits inside attacker if needed

        # Build a closure because attacker expects logits directly
        def forward_fn(x):
            with torch.no_grad():
                f = clip_model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                out = 100 * f @ text_features.T
            return out

        attacker.predict = forward_fn

        adv_images = attacker.perturb(images, labels)

        #########################################
        # ADV PRED
        #########################################
        with torch.no_grad():
            f2 = clip_model.encode_image(adv_images)
            f2 = f2 / f2.norm(dim=-1, keepdim=True)
            logits2 = 100 * f2 @ text_features.T

        preds_adv = logits2.argmax(1)

        adv_correct += (preds_adv == labels).sum().item()

        # ASR: clean correct AND adv wrong
        attack_success += ((preds_clean == labels) & (preds_adv != labels)).sum().item()

        total += labels.size(0)

    clean_acc = clean_correct / total
    adv_acc = adv_correct / total
    asr = attack_success / clean_total if clean_total > 0 else 0.

    print(f"{name}: clean={clean_acc:.4f} | attacked={adv_acc:.4f} | ASR={asr:.4f}\n")
    return clean_acc, adv_acc, asr


##############################################
# MAIN
##############################################

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    DATA_ROOT = "data"

    datasets = {
        "cifar10": CIFAR10(f"{DATA_ROOT}/cifar10", train=False, download=True, transform=transform),
        "flowers102": Flowers102(f"{DATA_ROOT}/flowers", split="test", download=True, transform=transform),
        "pets": OxfordIIITPet(f"{DATA_ROOT}/pets", split="test", download=True, transform=transform),
        "food101": Food101(f"{DATA_ROOT}/food", split="test", download=True, transform=transform),
        "cifar100": CIFAR100(f"{DATA_ROOT}/cifar100", train=False, download=True, transform=transform),
        "fgvc_aircraft": FGVCAircraft(f"{DATA_ROOT}/fgvc_aircraft", split="test", download=True, transform=transform),
        "stl10": STL10(f"{DATA_ROOT}/stl10", split="test", download=True, transform=transform),
    }

    for name, ds in datasets.items():
        class_names = get_class_list(name, ds)
        text_features = build_text_features(class_names, clip_model, device)
        evaluate_spatial_attack(name, ds, clip_model, device, text_features)


if __name__ == "__main__":
    main()
