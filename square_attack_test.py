# ================================================
# eval_square_aa.py — AutoAttack Square Attack
# ================================================
import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, Flowers102, Food101, DTD, OxfordIIITPet
from torch.utils.data import DataLoader
import clip

from torchvision.datasets import STL10
from torchvision.datasets import PCAM
from torchvision.datasets import FGVCAircraft

from autoattack import AutoAttack   # <<< 新增


# ------------------------------------------------
# class list resolver for each dataset
# ------------------------------------------------
def get_class_list(name, ds):
    """Return class names for each dataset"""

    # CIFAR10 — OK
    if name in ["cifar10", "cifar100"]:
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

    if name == "fgvc_aircraft":
        return ds.classes  # torchvision 自带 class names

    if name == "stl10":
        return ds.classes  # STL10 有 10 个固定类

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
# 把 CLIP + text_features 包成 AutoAttack 需要的“模型”
# ------------------------------------------------
class CLIPClassifier(torch.nn.Module):
    """
    AutoAttack 要求的 forward:
    - 输入: x in [0,1], shape [B,C,H,W]
    - 输出: logits, shape [B,num_classes]
    """
    def __init__(self, clip_model, text_features):
        super().__init__()
        self.clip_model = clip_model
        # 注册成 buffer，方便一起 .to(device)
        self.register_buffer("text_features", text_features)

    def forward(self, x):
        # Square Attack 是黑盒，不用梯度，这里可以 no_grad
        with torch.no_grad():
            f = self.clip_model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = 100 * f @ self.text_features.T
        return logits


# ------------------------------------------------
# Evaluate dataset with AutoAttack(Square)
# ------------------------------------------------
def evaluate_dataset(name, ds, clip_model, device, text_features,
                     eps=8/255, bs_aa=64):
    """
    使用 AutoAttack 的 Square Attack 对整个数据集做攻击，
    返回 clean_acc / adv_acc / ASR（基于 clean-correct 子集）
    """
    loader = DataLoader(ds, batch_size=bs_aa, shuffle=False)

    print(f"\n===== Evaluating {name} with AutoAttack(Square) =====")

    # 先把整个数据集堆成一个 tensor（可能会比较占内存）
    xs = []
    ys = []
    for images, labels in tqdm(loader, desc=f"{name}-collect", ncols=120):
        xs.append(images)
        ys.append(labels)
    x_test = torch.cat(xs, dim=0).to(device)
    y_test = torch.cat(ys, dim=0).to(device)

    print(f"[DEBUG] {name}: dataset size = {x_test.size(0)}", flush=True)

    # 包装成 AutoAttack 期望的 model
    model_aa = CLIPClassifier(clip_model, text_features.to(device)).to(device)

    # 只跑 Square Attack 的 AutoAttack
    adversary = AutoAttack(
        model_aa,
        norm='Linf',
        eps=eps,
        version='custom'
    )
    adversary.attacks_to_run = ['square']
    # 如果想固定随机性：
    # adversary.seed = 0

    print(f"[DEBUG] Running AutoAttack Square on {name} ...", flush=True)
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=bs_aa)

    # ---------- 统计指标 ----------
    with torch.no_grad():
        logits_clean = model_aa(x_test)
        preds_clean = logits_clean.argmax(dim=1)

        logits_adv = model_aa(x_adv)
        preds_adv = logits_adv.argmax(dim=1)

    total = y_test.size(0)
    clean_acc = (preds_clean == y_test).float().mean().item()
    adv_acc = (preds_adv == y_test).float().mean().item()

    # ASR：只看 clean 预测正确的样本
    mask_clean = preds_clean == y_test
    clean_total = mask_clean.sum().item()
    attack_success = ((preds_adv != y_test) & mask_clean).sum().item()
    asr = attack_success / clean_total if clean_total > 0 else 0.0

    print(
        f"{name}: clean={clean_acc:.4f} | "
        f"adv={adv_acc:.4f} | ASR={asr:.4f} "
        f"(clean_correct={clean_total}/{total})\n"
    )

    return clean_acc, adv_acc, asr


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
        "cifar100": CIFAR100(f"{DATA_ROOT}/cifar100", train=False, download=True, transform=transform),
        # "stanford_cars": StanfordCars(f"{DATA_ROOT}/stanford_cars", split="test", download=True, transform=transform),
        # "pcam": PCAM(f"{DATA_ROOT}/pcam", split="test",
        #          download=True, transform=transform),

        "fgvc_aircraft": FGVCAircraft(f"{DATA_ROOT}/fgvc_aircraft", split="test",
                                      download=True, transform=transform),

        "stl10": STL10(f"{DATA_ROOT}/stl10", split="test",
                       download=True, transform=transform),
    }

    print("[DEBUG] All datasets loaded", flush=True)
    print("[DEBUG] Starting evaluation ...", flush=True)

    # Evaluate each dataset
    for name, ds in datasets.items():
        print(f"[DEBUG] Preparing text features for {name}", flush=True)
        class_names = get_class_list(name, ds)
        text_features = build_text_features(class_names, clip_model, device)

        print(f"[DEBUG] Evaluating {name}", flush=True)
        # eps / bs_aa 可以按需改
        evaluate_dataset(
            name, ds, clip_model, device, text_features,
            eps=8/255,    # L∞ bound
            bs_aa=64      # AutoAttack 内部 batch size
        )


if __name__ == "__main__":
    main()
