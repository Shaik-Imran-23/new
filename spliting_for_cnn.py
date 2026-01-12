import os
import shutil
import random

SRC = "cnn_labeled_data_set"
DST = "cnn_1_90_dataset"

TRAIN_RATIO = 0.8
MIN_IMAGES = 5
random.seed(42)

for split in ["train", "val"]:
    os.makedirs(os.path.join(DST, split), exist_ok=True)

for label in range(1, 91):
    label = str(label)
    src_dir = os.path.join(SRC, label)

    if not os.path.isdir(src_dir):
        continue

    images = os.listdir(src_dir)
    if len(images) < MIN_IMAGES:
        print(f"⚠️ Skipping {label} (only {len(images)} images)")
        continue

    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)

    for split, imgs in {
        "train": images[:split_idx],
        "val": images[split_idx:]
    }.items():

        dst_dir = os.path.join(DST, split, label)
        os.makedirs(dst_dir, exist_ok=True)

        for img in imgs:
            shutil.copy(
                os.path.join(src_dir, img),
                os.path.join(dst_dir, img)
            )

print("✅ cnn_1_90_dataset created successfully")

