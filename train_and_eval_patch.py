import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10, Flowers102, Food101, DTD, OxfordIIITPet
from torch.utils.data import DataLoader

import clip
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AdversarialPatchPyTorch
import art

# ===============================
# CLIP Wrapper 给 ART 用
# ===============================
class ClipForART(torch.nn.Module):
    def __init__(self, clip_model, text_features):
        super().__init__()
        self.clip_model = clip_model
        self.text_features = text_features

    def forward(self, images):
        feats = self.clip_model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = 100 * feats @ self.text_features.T
        return logits


# ===============================
# 为数据集构建文本特征（zero-shot）
# ===============================
def build_text_features(class_names, clip_model, device):
    prompts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats


# ===============================
# 给图片贴 universal patch
# ===============================
def apply_universal_patch(images_tensor, patch_np, patch_attack):
    images_np = images_tensor.numpy()
    patched_np = patch_attack.apply_patch(images_np, patch=patch_np)
    return torch.from_numpy(patched_np)


def apply_patch_torch(images, patch_np, device):
    patch = torch.from_numpy(patch_np).to(device)
    B, C, H, W = images.shape
    _, h, w = patch.shape
    y0 = H - h
    x0 = W - w
    images = images.clone()
    images[:, :, y0:y0+h, x0:x0+w] = patch
    return images


# ===============================
# 在一个数据集上评估 zero-shot
# ===============================
def evaluate_dataset(name, dataset, clip_model, device, patch_np, patch_attack):
    print(f"\nEvaluating on: {name}")

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 类别名称
    if hasattr(dataset, "classes"):
        class_names = dataset.classes
    else:
        raise ValueError(f"Dataset {name} has no .classes attribute")

    # 为该数据集构建 text features
    text_features = build_text_features(class_names, clip_model, device)

    total = 0
    clean_correct = 0
    patched_correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # ---------- 干净 ----------
        with torch.no_grad():
            feats = clip_model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = 100 * feats @ text_features.T.to(device)

        preds_clean = logits.argmax(dim=1)
        clean_correct += (preds_clean == labels).sum().item()

        # ---------- 加补丁 ----------
        # patched_images = apply_universal_patch(images.cpu(), patch_np, patch_attack).to(device)
        patched_images = apply_patch_torch(images, patch_np, device)

        with torch.no_grad():
            feats_p = clip_model.encode_image(patched_images)
            feats_p = feats_p / feats_p.norm(dim=-1, keepdim=True)
            logits_p = 100 * feats_p @ text_features.T.to(device)

        preds_patched = logits_p.argmax(dim=1)
        patched_correct += (preds_patched == labels).sum().item()

        total += labels.size(0)

    print(f"{name}: clean_acc={clean_correct/total:.4f}, patched_acc={patched_correct/total:.4f}")


# ===============================
# main()
# ===============================
def main():
    # ---------------------------
    # 0. 设置 transform
    # ---------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 输出在 [0,1]
    ])

    # ============================================================
    # PHASE 1：在 CPU 上训练 universal patch（ART 只能用 CPU）
    # ============================================================
    print("=== Phase 1: Train universal patch on CPU ===")
    cpu_device = "cpu"

    # 1) 加载 CLIP（CPU）
    clip_model_cpu, _ = clip.load("ViT-B/32", device=cpu_device, jit=False)
    for p in clip_model_cpu.parameters():
        p.requires_grad = False
    clip_model_cpu.eval()

    # 2) CIFAR100 用于训练补丁
    cifar100_train = CIFAR100("data/cifar100", train=True, download=True, transform=transform)
    train_loader = DataLoader(cifar100_train, batch_size=8, shuffle=True)

    # 3) 文本特征（zero-shot）
    cifar100_class_names = cifar100_train.classes
    cifar100_text_features = build_text_features(cifar100_class_names, clip_model_cpu, cpu_device)

    # 4) 包装成 ART 模型
    wrapped_cpu = ClipForART(clip_model_cpu, cifar100_text_features).to(cpu_device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(wrapped_cpu.parameters(), lr=1e-4)  # 不会被真的训练

    # 5) ART classifier（CPU ONLY）
    classifier = PyTorchClassifier(
        model=wrapped_cpu,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=len(cifar100_class_names),
        clip_values=(0.0, 1.0),   # 非常重要
        preprocessing_defences=None,
        postprocessing_defences=None
    )

    # 6) 配置补丁攻击
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

    # 7) 准备训练数据（选 512 张）
    print("Generating universal adversarial patch on CPU...")
    all_images, all_labels = [], []
    max_images = 512

    for images, labels in train_loader:
        all_images.append(images)
        all_labels.append(labels)
        if sum(x.size(0) for x in all_images) >= max_images:
            break

    images_tensor = torch.cat(all_images)[:max_images]
    labels_tensor = torch.cat(all_labels)[:max_images]

    # ART 需要 numpy
    images_np = images_tensor.numpy()
    labels_np = labels_tensor.numpy()

    # 8) 训练补丁（CPU）
    patched_images_np, patch_np = patch_attack.generate(
        x=images_np,
        y=labels_np
    )

    # 保存补丁
    os.makedirs("artifacts", exist_ok=True)
    np.save("artifacts/universal_patch.npy", patch_np)
    print("Saved patch to artifacts/universal_patch.npy")

    # ============================================================
    # PHASE 2：用 GPU + PyTorch 对补丁效果进行评估（不再使用 ART）
    # ============================================================
    print("\n=== Phase 2: Evaluate on GPU ===")

    gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Evaluation device:", gpu_device)

    # 重新加载 CLIP 到 GPU
    clip_model_gpu, _ = clip.load("ViT-B/32", device=gpu_device, jit=False)
    for p in clip_model_gpu.parameters():
        p.requires_grad = False
    clip_model_gpu.eval()

    # -------------------------
    # 评估用数据集
    # -------------------------
    test_datasets = {
        "cifar10": CIFAR10("data/cifar10", train=False, download=True, transform=transform),
        "dtd": DTD("data/dtd", split="test", download=True, transform=transform),
        "flowers102": Flowers102("data/flowers", split="test", download=True, transform=transform),
        "pets": OxfordIIITPet("data/pets", split="test", download=True, transform=transform),
        "food101": Food101("data/food", split="test", download=True, transform=transform),
    }

    # -------------------------
    # 评估
    # -------------------------
    for name, dataset in test_datasets.items():
        evaluate_dataset(name, dataset, clip_model_gpu, gpu_device, patch_np, patch_attack)

if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# """
# train_and_eval_patch.py

# 流程:
# 1) 在 CIFAR-100 上训练/加载 ResNet50 (victim)
# 2) 使用 ART AdversarialPatch 在 CIFAR-100 上生成 universal patch
# 3) 把 patch 应用到多个数据集的验证/测试集 (CIFAR10, CIFAR100, DTD, Food101, Flowers102)
# 4) 使用 CLIP (zero-shot) 评估 clean/adv 图像分类准确率并打印结果

# 用法示例:
# python train_and_eval_patch.py --workdir ./work --device cuda:0 --epochs 30 --batch-size 128
# """

# import os
# import argparse
# import time
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms, datasets, models

# # ART 和 CLIP
# try:
#     from art.attacks.evasion import AdversarialPatch
#     from art.estimators.classification import PyTorchClassifier
# except Exception as e:
#     raise ImportError("需要安装 'adversarial-robustness-toolbox' (art). pip install adversarial-robustness-toolbox") from e

# try:
#     import clip
# except Exception as e:
#     raise ImportError("需要安装 openai/clip: pip install git+https://github.com/openai/CLIP.git") from e

# # ----- util: datasets loader helpers -----
# def get_dataset_and_transforms(name, root, split='test'):
#     """
#     返回 (dataset, classes)：
#     - name: 'cifar10','cifar100','dtd','food101','flowers102'
#     - root: 数据根目录
#     - split: 'train' or 'test'/'val' (按每个 dataset API 的约定)
#     """
#     name = name.lower()
#     if name == 'cifar10':
#         transform = transforms.Compose([transforms.ToTensor()])
#         ds = datasets.CIFAR10(root=root, train=(split=='train'), download=True, transform=transform)
#         classes = ds.classes
#         return ds, classes
#     if name == 'cifar100':
#         transform = transforms.Compose([transforms.ToTensor()])
#         ds = datasets.CIFAR100(root=root, train=(split=='train'), download=True, transform=transform)
#         classes = ds.classes
#         return ds, classes
#     if name == 'dtd':
#         # Describable Textures Dataset, torchvision.datasets.DTD exists
#         transform = transforms.Compose([transforms.ToTensor()])
#         ds = datasets.DTD(root=root, split=split if split in ('train','test','val') else 'test', download=True, transform=transform)
#         classes = ds.classes
#         return ds, classes
#     if name == 'food101':
#         transform = transforms.Compose([transforms.ToTensor()])
#         # torchvision uses split 'train' or 'test'
#         ds = datasets.Food101(root=root, split=split if split in ('train','test') else 'test', download=True, transform=transform)
#         classes = ds.classes
#         return ds, classes
#     if name in ('flowers102','oxfordflowers102','flower102'):
#         transform = transforms.Compose([transforms.ToTensor()])
#         ds = datasets.Flowers102(root=root, split=split if split in ('train','test','val') else 'test', download=True, transform=transform)
#         classes = ds.classes
#         return ds, classes
#     raise ValueError(f'Unknown dataset: {name}')


# # ----- victim model training / load -----
# def build_resnet50(num_classes=100, pretrained=True):
#     model = models.resnet50(pretrained=pretrained)
#     # replace final fc
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model

# def train_resnet50_cifar100(workdir, device, epochs=30, batch_size=128, lr=0.01, pretrained=True, resume=False):
#     model_path = Path(workdir) / "resnet50_cifar100.pth"
#     model = build_resnet50(num_classes=100, pretrained=pretrained).to(device)
#     if resume and model_path.exists():
#         print("Loading saved ResNet50 from", model_path)
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         return model
#     # dataset & loader
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])
#     transform_val = transforms.ToTensor()
#     trainset = datasets.CIFAR100(root=workdir, train=True, download=True, transform=transform_train)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     testset = datasets.CIFAR100(root=workdir, train=False, download=True, transform=transform_val)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)

#     best_acc = 0.0
#     for epoch in range(epochs):
#         model.train()
#         running = 0.0
#         t0 = time.time()
#         for images, labels in trainloader:
#             images = images.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running += loss.item() * images.size(0)
#         scheduler.step()
#         # eval
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for images, labels in testloader:
#                 images = images.to(device)
#                 labels = labels.to(device)
#                 outputs = model(images)
#                 _, preds = outputs.max(1)
#                 correct += (preds == labels).sum().item()
#                 total += labels.size(0)
#         acc = correct / total
#         print(f"Epoch {epoch+1}/{epochs} loss_avg={running/len(trainset):.4f} test_acc={acc:.4f} time={time.time()-t0:.1f}s")
#         if acc > best_acc:
#             best_acc = acc
#             torch.save(model.state_dict(), model_path)
#             print("Saved best model to", model_path)
#     # load best
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     print("Training finished. Best test acc:", best_acc)
#     return model

# # ----- ART wrapper -----
# def make_art_classifier(model, device, input_shape=(3,32,32), nb_classes=100):
#     """
#     Wrap PyTorch model into ART PyTorchClassifier
#     注意：ART 需要一个 loss 函数 & optimizer & input_shape & clip_values
#     """
#     loss = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.001)  # optimizer only for ART internals (not used heavily here)
#     # clip_values: images in [0,1]
#     from art.estimators.classification import PyTorchClassifier
#     classifier = PyTorchClassifier(
#         model=model,
#         loss=loss,
#         optimizer=optimizer,
#         input_shape=input_shape,
#         nb_classes=nb_classes,
#         clip_values=(0.0, 1.0),
#         device_type='gpu' if device and 'cuda' in str(device) else 'cpu'
#     )
#     return classifier

# # ----- generate adversarial patch on CIFAR-100 -----
# def generate_universal_patch(art_clf, trainset, workdir, device, max_iter=300, patch_size=32):
#     """
#     生成 universal adversarial patch
#     自动适配不同输入数据集的通道和尺寸
#     """

#     # 自动检测输入形状 (C, H, W)
#     input_shape = art_clf.input_shape
#     if len(input_shape) != 3:
#         raise ValueError(f"Unexpected input shape {input_shape}, expected (C,H,W)")
#     c, h, w = input_shape

#     # 设定 patch 尺寸比例，自动计算 patch 的高宽
#     patch_side = min(patch_size, h, w)   # 强制正方形
#     patch_shape = (c,patch_side, patch_side) # ART expects (H, W, C)

#     print(f"[INFO] Detected dataset input shape: (C,H,W)={input_shape}")
#     print(f"[INFO] Using patch shape (H,W,C)={patch_shape}")

    
#     # 初始化 ART Patch 攻击
#     attack = AdversarialPatch(
#         classifier=art_clf,
#         max_iter=max_iter,
#         patch_shape=patch_shape,
#         learning_rate=5.0,
#         verbose=True,
#     )

#     # 取少量样本生成 universal patch
#     print("[INFO] Generating universal adversarial patch...")
#     sample_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
#     batch = next(iter(sample_loader))
#     images, labels = batch[0], batch[1]
#     images_np = images.numpy()
#     labels_np = labels.numpy()

#     # 调用 ART 攻击生成 patch
#     patch = attack.generate(x=images_np, y=labels_np)
#     print("[INFO] Patch generation complete.")

#     # 保存 patch 文件
#     patch_path = os.path.join(workdir, "universal_patch.npy")
#     np.save(patch_path, patch)
#     print(f"[INFO] Patch saved to: {patch_path}")

#     return patch, attack

# # ----- apply patch to images (simple pasting) -----
# def paste_patch_on_batch(images_np, patch_np, location="random", scale=1.0):
#     """
#     images_np: (N,H,W,C) float32 [0,1]
#     patch_np: (ph,pw,C) float32 [0,1]
#     returns patched images (same shape)
#     location: 'random' or 'center'
#     scale: global scale multiplier for patch size (not implemented complexly here)
#     """
#     imgs = images_np.copy()
#     N, H, W, C = imgs.shape
#     ph, pw, pc = patch_np.shape
#     # if patch larger than image, resize (simple center crop/resize via numpy; better to use cv2 if available)
#     from PIL import Image
#     for i in range(N):
#         # optionally resize patch relative to image
#         # here we keep patch size but could scale
#         p = patch_np
#         # choose location
#         if location == "random":
#             max_y = max(0, H - p.shape[0])
#             max_x = max(0, W - p.shape[1])
#             y0 = np.random.randint(0, max_y+1) if max_y>0 else 0
#             x0 = np.random.randint(0, max_x+1) if max_x>0 else 0
#         else:
#             # center
#             y0 = max(0, (H - p.shape[0]) // 2)
#             x0 = max(0, (W - p.shape[1]) // 2)
#         # paste (simple overwrite)
#         imgs[i, y0:y0+p.shape[0], x0:x0+p.shape[1], :] = p
#     return imgs

# # ----- CLIP evaluation -----
# def make_clip_model(device):
#     model, preprocess = clip.load("ViT-B/32", device=device)  # 使用 ViT-B/32 zero-shot
#     model.eval()
#     return model, preprocess

# def compute_clip_accuracy_on_dataset(clip_model, preprocess, dataset, classes, patch_np=None, patch_apply_fn=None, device='cpu', batch_size=64, use_random_loc=True):
#     """
#     Evaluate CLIP (zero-shot) accuracy on a dataset.
#     - dataset: torchvision Dataset yielding PIL images or tensors. If tensors, assumed in [0,1].
#     - classes: list of class names (strings) in same order as dataset.targets (for CIFAR etc)
#     - patch_np: if not None, apply patch to images before feeding to CLIP
#     - patch_apply_fn: function(images_np, patch_np) -> patched_images_np
#     返回: accuracy (float)
#     """
#     # build text prompts to embed
#     # simple prompts: just class names; optionally "a photo of a {label}"
#     text_prompts = [f"a photo of a {c}" for c in classes]
#     with torch.no_grad():
#         text_tokens = clip.tokenize(text_prompts).to(device)
#         text_feats = clip_model.encode_text(text_tokens)  # (num_classes, dim)
#         text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#     correct = 0
#     total = 0
#     for batch in loader:
#         if isinstance(batch, (list, tuple)):
#             imgs, labels = batch[0], batch[1]
#         else:
#             # some datasets return dict
#             imgs = batch['image']
#             labels = batch['label']
#         # imgs might be PIL.Image or Tensor
#         # convert to numpy HWC [0,1]
#         imgs_np_list = []
#         for im in imgs:
#             if isinstance(im, torch.Tensor):
#                 im_np = im.numpy().transpose(1,2,0)  # CHW -> HWC
#             else:
#                 # PIL Image
#                 im_np = np.array(im).astype(np.float32) / 255.0
#                 if im_np.ndim == 2:
#                     im_np = np.stack([im_np]*3, axis=-1)
#                 elif im_np.shape[2] == 4:
#                     im_np = im_np[:,:,:3]
#             # ensure size for CLIP preprocess will be handled by preprocess later
#             imgs_np_list.append(im_np)
#         # convert to PIL images to pass through CLIP preprocess
#         input_pils = []
#         for im_np in imgs_np_list:
#             from PIL import Image
#             im_uint8 = (np.clip(im_np,0,1)*255).astype(np.uint8)
#             input_pils.append(Image.fromarray(im_uint8))
#         # apply patch if requested: convert back to numpy HWC then apply patch_apply_fn
#         if patch_np is not None:
#             # CLIP preprocess may resize; for accurate patch we should resize patch to match preprocessed image size (e.g., 224x224)
#             # Simpler approach: apply patch after CLIP preprocess on the preprocessed images' numpy arrays.
#             # So we will preprocess first to tensors, convert to numpy HWC (224x224), apply patch, then feed to CLIP model.
#             # Preprocess batch:
#             preproc_tensors = torch.stack([preprocess(p).to(device) for p in input_pils])  # (B,3,224,224)
#             imgs_pre_np = preproc_tensors.cpu().numpy().transpose(0,2,3,1)  # B,H,W,C
#             # ensure patch shape matches H,W scale if needed
#             patched_np = patch_apply_fn(imgs_pre_np, patch_np) if patch_apply_fn is not None else paste_patch_on_batch(imgs_pre_np, patch_np)
#             # convert patched_np back to tensors for CLIP
#             patched_t = torch.from_numpy(patched_np.transpose(0,3,1,2)).to(device).float()
#             # normalize as CLIP expects (preprocess includes normalization) - but since we used preprocess above then converted to numpy, skip extra normalize
#             image_feats = clip_model.encode_image(patched_t)
#         else:
#             # no patch: simple preprocess and encode
#             preproc_tensors = torch.stack([preprocess(p).to(device) for p in input_pils])  # (B,3,224,224)
#             with torch.no_grad():
#                 image_feats = clip_model.encode_image(preproc_tensors)
#         image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
#         # similarity
#         logits = (100.0 * image_feats @ text_feats.T).cpu().numpy()  # (B, num_classes)
#         preds = np.argmax(logits, axis=1)
#         labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
#         # For datasets where label indices align with classes order, this works. For some datasets (Flowers102) mapping may differ; user should verify.
#         correct += (preds == labels_np).sum()
#         total += len(labels_np)
#     acc = correct / total
#     return acc

# # ----- main pipeline -----
# def main(args):
#     device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
#     workdir = args.workdir
#     os.makedirs(workdir, exist_ok=True)

#     # 1) train or load ResNet50 on CIFAR-100
#     if args.load_resnet and Path(args.load_resnet).exists():
#         print("Loading ResNet from", args.load_resnet)
#         model = build_resnet50(num_classes=100, pretrained=False).to(device)
#         model.load_state_dict(torch.load(args.load_resnet, map_location=device))
#     else:
#         model = train_resnet50_cifar100(workdir, device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, pretrained=args.pretrained, resume=args.resume_train)

#     # Wrap with ART classifier
#     art_clf = make_art_classifier(model, device, input_shape=(3,32,32), nb_classes=100)

#     # 2) generate or load patch
#     patch_path = Path(workdir) / args.patch_name
#     if args.load_patch and patch_path.exists():
#         print("Loading patch from", patch_path)
#         patch = np.load(patch_path)
#         # ensure shape HWC and values in [0,1]
#         patch = np.clip(patch, 0.0, 1.0)
#         # For ART attacks we also want an attack object to use apply_patch helper if available.
#         h_p, w_p, c_p = patch.shape
#         attack = AdversarialPatch(classifier=art_clf, patch_shape=(c_p, h_p, w_p))
#         attack.patch = patch
#     else:
#         # prepare train dataset for patch generation
#         trainset = datasets.CIFAR100(root=workdir, train=True, download=True, transform=transforms.ToTensor())
#         patch, attack = generate_universal_patch(art_clf, trainset, workdir, device,
#                                                 max_iter=args.max_iter, patch_size=max(args.patch_h, args.patch_w),
#                                                 )

#     # 3) prepare CLIP
#     clip_model, clip_preprocess = make_clip_model(device)

#     # 4) datasets to evaluate
#     datasets_list = ['cifar10','cifar100','dtd','food101','flowers102']
#     results = {}
#     for ds_name in datasets_list:
#         print(f"\nEvaluating dataset: {ds_name}")
#         ds, classes = get_dataset_and_transforms(ds_name, root=args.workdir, split='test')
#         # compute clean accuracy
#         acc_clean = compute_clip_accuracy_on_dataset(clip_model, clip_preprocess, ds, classes, patch_np=None, patch_apply_fn=None, device=device, batch_size=args.eval_batch)
#         print(f"{ds_name} CLIP clean acc: {acc_clean:.4f}")
#         # compute adv accuracy: we need patch applied to preprocessed CLIP images (224)
#         # So construct patch rescaled to CLIP image size (224x224) if needed inside compute function it handles
#         # We'll pass patch_np (we assume patch is small), and patch_apply_fn which pastes patch onto preprocessed 224x224 images
#         def patch_apply_fn(images_np, patch_np_local):
#             # images_np: (B,H,W,C) typically H=W=224
#             # resize patch to about 0.2*W if patch smaller: let's resize via PIL
#             from PIL import Image
#             B,H,W,C = images_np.shape
#             # target patch size: scale ratio based on original patch vs 32x32: proportion = args.patch_w / 32
#             # We'll set patch target width = int(0.2 * W) as default
#             target_w = max(4, int(0.2 * W))
#             target_h = int(target_w * (patch_np_local.shape[0]/patch_np_local.shape[1]))
#             p_img = Image.fromarray((patch_np_local*255).astype(np.uint8))
#             p_resized = np.array(p_img.resize((target_w, target_h))).astype(np.float32)/255.0
#             return paste_patch_on_batch(images_np, p_resized, location='random')
#         acc_adv = compute_clip_accuracy_on_dataset(clip_model, clip_preprocess, ds, classes, patch_np=patch, patch_apply_fn=patch_apply_fn, device=device, batch_size=args.eval_batch)
#         print(f"{ds_name} CLIP adv acc (with patch): {acc_adv:.4f}")
#         results[ds_name] = {'clean': float(acc_clean), 'adv': float(acc_adv)}

#     # print summary
#     print("\n==== Summary ====")
#     for ds, v in results.items():
#         print(f"{ds}: clean_acc={v['clean']:.4f}, adv_acc={v['adv']:.4f}")
#     # save results
#     import json
#     with open(Path(workdir)/"clip_patch_eval_results.json", "w") as f:
#         json.dump(results, f, indent=2)
#     print("Saved results to", Path(workdir)/"clip_patch_eval_results.json")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--workdir', type=str, default='./work', help='工作目录，保存模型/patch等')
#     parser.add_argument('--device', type=str, default='cuda:0', help='设备')
#     parser.add_argument('--epochs', type=int, default=30)
#     parser.add_argument('--batch-size', type=int, default=128)
#     parser.add_argument('--eval-batch', type=int, default=64)
#     parser.add_argument('--lr', type=float, default=0.01)
#     parser.add_argument('--pretrained', action='store_true', help='是否将 resnet50 从 ImageNet 预训练权重微调')
#     parser.add_argument('--resume-train', action='store_true', help='若已存在模型，直接加载')
#     parser.add_argument('--max_iter', type=int, default=300, help='AdversarialPatch 内部最大迭代次数 (可调)')
#     parser.add_argument('--patch-h', type=int, default=8, help='patch 高 (像素，针对 32x32 图像)')
#     parser.add_argument('--patch-w', type=int, default=8, help='patch 宽 (像素)')
#     parser.add_argument('--patch-name', type=str, default='universal_patch.npy', help='保存 patch 文件名')
#     parser.add_argument('--load-patch', action='store_true', help='若工作目录有 patch 则加载')
#     parser.add_argument('--load-resnet', type=str, default='', help='若提供路径则加载 resnet 权重')
#     args = parser.parse_args()
#     main(args)
