import os
import requests
from tqdm import tqdm
import zipfile

# ---------------------------------------------------------
# Download TinyImageNet with progress bar
# ---------------------------------------------------------
def download_url(url, save_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(save_path)}",
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


# ---------------------------------------------------------
# Unzip with progress bar
# ---------------------------------------------------------
def unzip_with_progress(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as z:
        files = z.namelist()

        for file in tqdm(files, desc="Extracting", unit="files"):
            z.extract(file, extract_to)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    os.makedirs("data", exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = "data/tiny-imagenet-200.zip"
    extract_dir = "data"

    # Step 1: download
    if not os.path.exists(zip_path):
        download_url(url, zip_path)
    else:
        print("Zip file already exists, skipping download.")

    # Step 2: extract
    if not os.path.exists("data/tiny-imagenet-200"):
        unzip_with_progress(zip_path, extract_dir)
    else:
        print("Already extracted.")

    print("\nTinyImageNet is ready at: data/tiny-imagenet-200/\n")


if __name__ == "__main__":
    main()
