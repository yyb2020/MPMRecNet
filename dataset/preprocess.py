# dataset/preprocess.py

"""
Convert full RGB PNG microscopy images into .pt patch files for training.
- Cuts each image into 512x512 patches (configurable)
- Resizes each patch to 224x224 for MaxViT
- Extracts Red channel (SHG) as modality "b"
- Extracts Green channel (TPEF) as modality "a"
- Stores in format: {patch_id: {"a": tensor, "b": tensor}}

"""

import os
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ==== Config ====
INPUT_DIR = "/path/to/png_images"  # folder of  images
OUTPUT_DIR = "/path/to/output_pt"
PATCH_SIZE = 512
RESIZE_TO = 224
STRIDE = 512  # you can make it smaller if you want overlap

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image transforms
base_transform = transforms.Compose([
    transforms.Resize((RESIZE_TO, RESIZE_TO), interpolation=Image.BICUBIC),
    transforms.ToTensor()
])

normalize_r = transforms.Normalize(mean=[0.485], std=[0.229])
normalize_g = transforms.Normalize(mean=[0.485], std=[0.229])

def extract_patches(img: Image.Image, patch_size: int, stride: int):
    w, h = img.size
    pad_w = (math.ceil(w / patch_size) * patch_size) - w
    pad_h = (math.ceil(h / patch_size) * patch_size) - h

    padded_img = Image.new("RGB", (w + pad_w, h + pad_h), color=(0, 0, 0))
    padded_img.paste(img, (0, 0))

    patches = []
    positions = []

    for r in range(0, padded_img.height, stride):
        for c in range(0, padded_img.width, stride):
            patch = padded_img.crop((c, r, c + patch_size, r + patch_size))
            patches.append(patch)
            positions.append(f"r{r // stride}_c{c // stride}")

    return patches, positions

def convert_png_to_pt(fname):
    name = os.path.splitext(fname)[0]
    img_path = os.path.join(INPUT_DIR, fname)
    save_path = os.path.join(OUTPUT_DIR, f"{name}.pt")

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error reading {fname}: {e}")
        return

    patch_dict = {}

    patches, keys = extract_patches(img, PATCH_SIZE, STRIDE)

    for patch, key in zip(patches, keys):
        tensor = base_transform(patch)  # shape [3, H, W]

        r = normalize_r(tensor[0:1])  # SHG
        g = normalize_g(tensor[1:2])  # TPEF

        patch_dict[key] = {"a": g, "b": r}

    torch.save(patch_dict, save_path)
    print(f"✅ Saved: {save_path}")

# ==== Run Batch Conversion ====
if __name__ == "__main__":
    file_list = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".png")]

    for fname in tqdm(file_list):
        convert_png_to_pt(fname)

