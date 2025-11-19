# ================================================
# train_patch.py — 在 GPU 上训练 CLIP adversarial patch（ART）
# ================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

import clip
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AdversarialPatchPyTorch

import torchvision.transforms.functional as TF

# --------------------------------------
# 修复 torchvision resize 接受 np.int64 的问题
# --------------------------------------
_real_resize = TF.resize

def safe_resize(img, size, *args, **kwargs):
    # ART 里会传 numpy.int64，这里统一转成 Python int
    if isinstance(size, (list, tuple)):
        size = tuple(int(s) for s in size)
    else:
        size = int(size)
    return _real_resize(img, size, *args, **kwargs)

TF.resize = safe_resize


# -------------------------
# CLIP Wrapper 给 ART 用
# -------------------------
class ClipForART(torch.nn.Module):
    def __init__(self, clip_model, text_features):
        super().__init__()
        self.clip_model = clip_model          # 在 device 上
        self.text_features = text_features    # [num_classes, dim] 在同一个 device

    def forward(self, images):
        # images: torch.Tensor, on same device as classifier._device
        feats = self.clip_model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = 100 * feats @ self.text_features.T
        return logits


# -------------------------
# Text features (zero-shot)
# -------------------------
def build_text_features(class_names, clip_model, device):
    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in class_names]
    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        txt = clip_model.encode_text(tokens)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt.to(device)


# ======================================================
# main()
# ======================================================
def main():

    # 0. 设备：直接用 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:", device)

    # 1. transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),        # 输出 [0,1]
    ])

    # 2. 加载 CLIP 到 device
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    # 3. CIFAR100 数据
    cifar100 = CIFAR100("data/cifar100", train=True, download=True, transform=transform)
    loader = DataLoader(cifar100, batch_size=8, shuffle=True)

    class_names = cifar100.classes
    text_features = build_text_features(class_names, clip_model, device)  # [100, dim] 在 device 上

    # 4. 包装成 ART classifier
    wrapped = ClipForART(clip_model, text_features).to(device)
    optimizer = optim.Adam(wrapped.parameters(), lr=1e-4)

    classifier = PyTorchClassifier(
        model=wrapped,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=len(class_names),
        clip_values=(0.0, 1.0),           # 必须非 None
        preprocessing_defences=None,
        postprocessing_defences=None,
        device_type="gpu" if device.type == "cuda" else "cpu"
    )

    # 统一让 ART 认为 device 就是我们选的这个
    classifier._device = device

    # ---- 把 ART 里的 preprocessing 也迁到同一个 device（防止它偷用 CPU/GPU）----
    if hasattr(classifier, "preprocessing_operations") and classifier.preprocessing_operations:
        for p in classifier.preprocessing_operations:
            if hasattr(p, "_device"):
                p._device = device

    if hasattr(classifier, "_preprocessing_operations") and classifier._preprocessing_operations:
        for p in classifier._preprocessing_operations:
            if hasattr(p, "_device"):
                p._device = device
    # ----------------------------------------------------------------------

    # 5. 构建 adversarial patch attack
    patch_attack = AdversarialPatchPyTorch(
        estimator=classifier,
        patch_shape=(3, 112, 112),
        rotation_max=0.0,
        scale_min=0.3,
        scale_max=0.6,
        learning_rate=1.0,
        max_iter=200,
        batch_size=8,
        targeted=False,
        verbose=True
    )

    # ART 内部也有自己的 device 记录，强制和 classifier 保持一致
    patch_attack._device = device

    # 6. 收集一小部分 CIFAR100 图像来训练 patch（例如 512 张）
    imgs, lbs = [], []
    max_images = 512

    for x, y in loader:
        imgs.append(x)   # 这里是 CPU tensor，ART 会自己转到 device
        lbs.append(y)
        if sum(t.size(0) for t in imgs) >= max_images:
            break

    x_np = torch.cat(imgs)[:max_images].numpy()
    y_np = torch.cat(lbs)[:max_images].numpy()

    # 7. 开始训练 patch（在 GPU 上优化）
    print("\n[AdversarialPatchPyTorch] Start training universal patch...\n")
    patched_imgs, patch_np = patch_attack.generate(x=x_np, y=y_np)

    # 8. 保存 patch
    os.makedirs("artifacts", exist_ok=True)
    np.save("artifacts/universal_patch.npy", patch_np)
    print("\nSaved universal patch :artifacts/universal_patch.npy\n")


if __name__ == "__main__":
    main()
