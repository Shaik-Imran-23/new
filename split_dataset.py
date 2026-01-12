import os
import shutil
import random

SRC_DIR = "."
OUT_DIR = "number_dataset"
TRAIN_RATIO = 0.8

os.makedirs(f"{OUT_DIR}/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/val", exist_ok=True)

for label in os.listdir(SRC_DIR):
    if not label.isdigit():
        continue

    imgs = os.listdir(label)
    imgs = [i for i in imgs if i.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(imgs) == 0:
        continue

    random.shuffle(imgs)
    split_idx = int(len(imgs) * TRAIN_RATIO)

    train_imgs = imgs[:split_idx]
    val_imgs   = imgs[split_idx:]

    os.makedirs(f"{OUT_DIR}/train/{label}", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/val/{label}", exist_ok=True)

    for img in train_imgs:
        shutil.copy(
            os.path.join(label, img),
            os.path.join(OUT_DIR, "train", label, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(label, img),
            os.path.join(OUT_DIR, "val", label, img)
        )

print("✅ Dataset split completed successfully")

