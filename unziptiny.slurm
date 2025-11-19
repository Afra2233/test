#!/bin/bash
#SBATCH -p compute            # CPU partition，比 GPU 快排队
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8     # 多线程解压更快
#SBATCH --mem=16G
#SBATCH --time=24:00:00       # 足够解压
#SBATCH --job-name=extract_food101


set -e

echo "[INFO] Preparing Food101 dataset..."

DATA_DIR="data/food"
TAR_PATH="$DATA_DIR/food-101.tar.gz"
ZIP_PATH="$DATA_DIR/food-101.zip"
TARGET_DIR="$DATA_DIR/food-101"

mkdir -p "$DATA_DIR"

# ===============================
# 1. Download food-101.tar.gz
# ===============================
if [ ! -f "$TAR_PATH" ]; then
    echo "[INFO] Downloading food-101.tar.gz..."
    wget -O "$TAR_PATH" "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
else
    echo "[INFO] TAR already exists: $TAR_PATH"
fi

# ===============================
# 2. Extract .tar.gz if folder missing
# ===============================
if [ ! -d "$TARGET_DIR" ]; then
    echo "[INFO] Extracting food-101.tar.gz ..."
    tar -xzf "$TAR_PATH" -C "$DATA_DIR"
else
    echo "[INFO] Target directory already exists: $TARGET_DIR"
fi

# ===============================
# 3. Optional: ZIP support (your request)
# ===============================

ZIP_URL="https://example.com/food-101.zip"
# 如果你没有 zip 源，下面这段会自动跳过
echo "[INFO] Checking for ZIP..."
if [ ! -f "$ZIP_PATH" ]; then
    echo "[INFO] Trying to download ZIP..."
    if wget -O "$ZIP_PATH" "$ZIP_URL"; then
        echo "[INFO] ZIP downloaded. Extracting..."
        unzip -o "$ZIP_PATH" -d "$DATA_DIR"
    else
        echo "[WARN] ZIP download failed or URL invalid. Skipping ZIP."
    fi
else
    echo "[INFO] ZIP already exists: $ZIP_PATH"
fi

echo "[INFO] Food101
