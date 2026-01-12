import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = "number_dataset"
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
IMG_SIZE = 64
NUM_CLASSES = 150
MODEL_PATH = "balloon_number_cnn.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# TRANSFORMS
# -----------------------------
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -----------------------------
# DATASET & LOADER
# -----------------------------
train_ds = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "train"),
    transform=train_transform
)
val_ds = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "val"),
    transform=val_transform
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Training samples: {len(train_ds)}")
print(f"✅ Validation samples: {len(val_ds)}")
print(f"✅ Classes detected: {len(train_ds.classes)}")

# -----------------------------
# MODEL
# -----------------------------
class BalloonNumberCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = BalloonNumberCNN(NUM_CLASSES).to(DEVICE)

# -----------------------------
# LOSS & OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# TRAINING LOOP
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = 100 * train_correct / train_total

    # -----------------------------
    # VALIDATION
    # -----------------------------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}%"
    )

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved as {MODEL_PATH}")

