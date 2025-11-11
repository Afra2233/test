#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00


module load anaconda3/2022.05

unzip tiny-imagenet-200.zip

cd tiny-imagenet-200

# 修复 val 文件结构
python3 << 'EOF'
import os, shutil

ann = open("val/val_annotations.txt").read().strip().split("\n")
mapping = [x.split("\t") for x in ann]

os.makedirs("val_fixed", exist_ok=True)
for img, cls, *_ in mapping:
    src = os.path.join("val/images", img)
    dst_dir = os.path.join("val_fixed", cls)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.move(src, os.path.join(dst_dir, img))

shutil.rmtree("val")
os.rename("val_fixed", "val")
EOF

echo "tinyImageNet 准备完成"