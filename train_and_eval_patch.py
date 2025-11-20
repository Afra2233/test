import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import clip
import numpy as np
import random
from tqdm import tqdm
import torchvision.transforms.functional as TF


# -----------------------------------------------------------
# 1. Apply patch with EOT (random location + random scale)
# -----------------------------------------------------------
def apply_patch_eot(images, patch, scale_min=0.3, scale_max=0.7):
    B, C, H, W = images.shape
    patched = images.clone()
    P = patch.size(-1)

    for i in range(B):
        scale = random.uniform(scale_min, scale_max)
        new_size = int(P * scale)

        p = F.interpolate(
            patch.unsqueeze(0),
            size=(new_size, new_size),
            mode="bilinear",
            align_corners=False
        )[0]

        max_y = H - new_size
        max_x = W - new_size
        y0 = random.randint(0, max_y)
        x0 = random.randint(0, max_x)

        patched[i, :, y0:y0+new_size, x0:x0+new_size] = p

    return torch.clamp(patched, 0, 1)


# -----------------------------------------------------------
# 2. Build CLIP text features
# -----------------------------------------------------------
def build_text_features(classes, clip_model, device):
    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in classes]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        txt = clip_model.encode_text(tokens)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt.to(device)


# ===========================================================
# 3. Main: Train universal patch on TinyImageNet
# ===========================================================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training device:", device)

    # --------------------------------------------------------------
    # Load CLIP
    # --------------------------------------------------------------
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    # --------------------------------------------------------------
    # TinyImageNet dataset
    # --------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(
        root="data/tiny-imagenet-200/train",   # <— 你的 TinyImageNet 路径
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    class_names = dataset.classes
    print("Loaded TinyImageNet:", len(dataset), "samples")


    # --------------------------------------------------------------
    # Build CLIP text features
    # --------------------------------------------------------------
    text_features = build_text_features(class_names, clip_model, device)

    # --------------------------------------------------------------
    # Learnable patch
    # --------------------------------------------------------------
    PATCH_SIZE = 112
    patch = nn.Parameter(torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=device))

    optimizer = optim.Adam([patch], lr=5e-2)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 5

    # --------------------------------------------------------------
    # Training loop with progress bars
    # --------------------------------------------------------------
    for epoch in range(EPOCHS):

        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

        batch_bar = tqdm(loader, desc=f"Training", ncols=120)

        running_loss = 0

        for batch_idx, (images, labels) in enumerate(batch_bar):

            images = images.to(device)
            labels = labels.to(device)

            # EOT
            images_patched = apply_patch_eot(images, patch)

            # patched forward
            feats_p = clip_model.encode_image(images_patched)
            feats_p = feats_p / feats_p.norm(dim=-1, keepdim=True)
            logits_p = 100 * feats_p @ text_features.T

            # untargeted CLIP-friendly loss: maximize CE
            loss_ce = criterion(logits_p, labels)
            loss = -loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # keep patch valid
            with torch.no_grad():
                patch.clamp_(0, 1)

            running_loss += loss.item()

            # Update tqdm bar info
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch+1} avg loss: {running_loss/len(loader):.4f}")

    # --------------------------------------------------------------
    # Save patch
    # --------------------------------------------------------------
    os.makedirs("artifacts", exist_ok=True)
    np.save("artifacts/universal_patch_tinyimagenet.npy", patch.detach().cpu().numpy())
    print("\nSaved patch to artifacts/universal_patch_tinyimagenet_2.npy\n")
    

    # 1) Save the raw patch scaled to 224x224 (for visibility)
    patch_img = patch.detach().cpu().clone()  # [3, P, P]
    patch_vis = F.interpolate(
        patch_img.unsqueeze(0),
        size=(224, 224),
        mode="nearest"
    )[0]
    patch_pil = TF.to_pil_image(patch_vis)
    patch_pil.save("artifacts/patch_visual.png")
    print("Saved patch image to artifacts/patch_visual.png")

    # 2) Save one example image with patch applied
    # Take the first sample in the dataset
    sample_img, _ = dataset[0]  # dataset is TinyImageNet train
    sample_img = sample_img.unsqueeze(0).to(device)  # [1,3,224,224]

    # Resize patch to a fixed size for visualization
    p = F.interpolate(patch.unsqueeze(0), size=(112, 112), mode="nearest")

    # Fixed position (easier to visualize): top-left corner
    def apply_patch_fixed(img, patch_tensor, y=10, x=10):
        img = img.clone()
        _, _, ph, pw = patch_tensor.shape
        img[:, :, y:y+ph, x:x+pw] = patch_tensor
        return torch.clamp(img, 0, 1)

    patched_img = apply_patch_fixed(sample_img, p)

    patched_pil = TF.to_pil_image(patched_img[0].cpu())
    patched_pil.save("artifacts/patched_example.png")
    print("Saved sample with patch to artifacts/patched_example.png\n")

if __name__ == "__main__":
    main()
























# # ================================================
# # train_patch.py — 在 GPU 上训练 CLIP adversarial patch（ART）
# # ================================================
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# from torchvision import transforms
# from torchvision.datasets import CIFAR100
# from torch.utils.data import DataLoader

# import clip
# from art.estimators.classification import PyTorchClassifier
# from art.attacks.evasion import AdversarialPatchPyTorch

# import torchvision.transforms.functional as TF

# # --------------------------------------
# # 修复 torchvision resize 接受 np.int64 的问题
# # --------------------------------------
# _real_resize = TF.resize

# def safe_resize(img, size, *args, **kwargs):
#     # ART 里会传 numpy.int64，这里统一转成 Python int
#     if isinstance(size, (list, tuple)):
#         size = tuple(int(s) for s in size)
#     else:
#         size = int(size)
#     return _real_resize(img, size, *args, **kwargs)

# TF.resize = safe_resize


# # -------------------------
# # CLIP Wrapper 给 ART 用
# # -------------------------
# class ClipForART(torch.nn.Module):
#     def __init__(self, clip_model, text_features):
#         super().__init__()
#         self.clip_model = clip_model          # 在 device 上
#         self.text_features = text_features    # [num_classes, dim] 在同一个 device

#     def forward(self, images):
#         # images: torch.Tensor, on same device as classifier._device
#         feats = self.clip_model.encode_image(images)
#         feats = feats / feats.norm(dim=-1, keepdim=True)
#         logits = 100 * feats @ self.text_features.T
#         return logits


# # -------------------------
# # Text features (zero-shot)
# # -------------------------
# def build_text_features(class_names, clip_model, device):
#     prompts = [f"a photo of a {c.replace('_', ' ')}" for c in class_names]
#     tokens = clip.tokenize(prompts).to(device)

#     with torch.no_grad():
#         txt = clip_model.encode_text(tokens)
#         txt = txt / txt.norm(dim=-1, keepdim=True)
#     return txt.to(device)


# # ======================================================
# # main()
# # ======================================================
# def main():

#     # 0. 设备：直接用 GPU（如果可用）
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Training device:", device)

#     # 1. transforms
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),        # 输出 [0,1]
#     ])

#     # 2. 加载 CLIP 到 device
#     clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
#     clip_model.eval()

#     # 3. CIFAR100 数据
#     cifar100 = CIFAR100("data/cifar100", train=True, download=True, transform=transform)
#     loader = DataLoader(cifar100, batch_size=8, shuffle=True)

#     class_names = cifar100.classes
#     text_features = build_text_features(class_names, clip_model, device)  # [100, dim] 在 device 上

#     # 4. 包装成 ART classifier
#     wrapped = ClipForART(clip_model, text_features).to(device)
#     optimizer = optim.Adam(wrapped.parameters(), lr=1e-4)

#     classifier = PyTorchClassifier(
#         model=wrapped,
#         loss=nn.CrossEntropyLoss(),
#         optimizer=optimizer,
#         input_shape=(3, 224, 224),
#         nb_classes=len(class_names),
#         clip_values=(0.0, 1.0),           # 必须非 None
#         preprocessing_defences=None,
#         postprocessing_defences=None,
#         device_type="gpu" if device.type == "cuda" else "cpu"
#     )

#     # 统一让 ART 认为 device 就是我们选的这个
#     classifier._device = device

#     # ---- 把 ART 里的 preprocessing 也迁到同一个 device（防止它偷用 CPU/GPU）----
#     if hasattr(classifier, "preprocessing_operations") and classifier.preprocessing_operations:
#         for p in classifier.preprocessing_operations:
#             if hasattr(p, "_device"):
#                 p._device = device

#     if hasattr(classifier, "_preprocessing_operations") and classifier._preprocessing_operations:
#         for p in classifier._preprocessing_operations:
#             if hasattr(p, "_device"):
#                 p._device = device
#     # ----------------------------------------------------------------------

#     # 5. 构建 adversarial patch attack
#     patch_attack = AdversarialPatchPyTorch(
#         estimator=classifier,
#         patch_shape=(3, 112, 112),
#         rotation_max=0.0,
#         scale_min=0.3,
#         scale_max=0.6,
#         learning_rate=1.0,
#         max_iter=200,
#         batch_size=8,
#         targeted=False,
#         verbose=True
#     )

#     # ART 内部也有自己的 device 记录，强制和 classifier 保持一致
#     patch_attack._device = device

#     # 6. 收集一小部分 CIFAR100 图像来训练 patch（例如 512 张）
#     imgs, lbs = [], []
#     max_images = 512

#     for x, y in loader:
#         imgs.append(x)   # 这里是 CPU tensor，ART 会自己转到 device
#         lbs.append(y)
#         if sum(t.size(0) for t in imgs) >= max_images:
#             break

#     x_np = torch.cat(imgs)[:max_images].numpy()
#     y_np = torch.cat(lbs)[:max_images].numpy()

#     # 7. 开始训练 patch（在 GPU 上优化）
#     print("\n[AdversarialPatchPyTorch] Start training universal patch...\n")
#     patched_imgs, patch_np = patch_attack.generate(x=x_np, y=y_np)

#     # 8. 保存 patch
#     os.makedirs("artifacts", exist_ok=True)
#     np.save("artifacts/universal_patch.npy", patch_np)
#     print("\nSaved universal patch :artifacts/universal_patch.npy\n")


# if __name__ == "__main__":
#     main()
