import os

BASE_DIR = "number_dataset"

removed = []

for split in ["train", "val"]:
    split_dir = os.path.join(BASE_DIR, split)

    for label in os.listdir(split_dir):
        label_path = os.path.join(split_dir, label)

        if not os.path.isdir(label_path):
            continue

        imgs = [
            f for f in os.listdir(label_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if len(imgs) == 0:
            os.rmdir(label_path)
            removed.append(f"{split}/{label}")

print("Removed empty folders:")
for r in removed:
    print(" -", r)

