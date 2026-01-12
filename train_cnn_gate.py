import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score
import numpy as np

# ---------------- CONFIG ----------------
DATA_DIR = "cnn_gate_dataset"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- TRANSFORMS ----------------
train_tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ---------------- DATA ----------------
train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tf)
val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", train_ds.classes)  # ['not_number', 'number']

# ---------------- CLASS WEIGHT (CONSERVATIVE GATE) ----------------
# class 1 = NUMBER, class 0 = NOT_NUMBER
num_number = sum(train_ds.targets)
num_not_number = len(train_ds.targets) - num_number

# pos_weight < 1 → penalize false positives (junk → number)
pos_weight = torch.tensor([num_not_number / num_number]).to(DEVICE)
print("pos_weight (conservative):", pos_weight.item())

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ---------------- MODEL ----------------
class CNNGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = CNNGate().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAIN LOOP ----------------
for epoch in range(EPOCHS):
    model.train()
    train_preds, train_gt = [], []

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.float().to(DEVICE)

        logits = model(x).view(-1)   # ✅ SAFE SHAPE
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        train_preds.extend(preds.cpu().numpy())
        train_gt.extend(y.cpu().numpy())

    train_acc = np.mean(np.array(train_preds) == np.array(train_gt)) * 100

    # ---------------- VALIDATION ----------------
    model.eval()
    val_preds, val_gt = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.float().to(DEVICE)

            logits = model(x).view(-1)  # ✅ SAFE SHAPE
            probs = torch.sigmoid(logits)

            # 🔒 CONSERVATIVE THRESHOLD
            preds = (probs > 0.7).int()

            val_preds.extend(preds.cpu().numpy())
            val_gt.extend(y.cpu().numpy())

    val_acc = np.mean(np.array(val_preds) == np.array(val_gt)) * 100
    precision = precision_score(val_gt, val_preds, zero_division=0)
    recall = recall_score(val_gt, val_preds, zero_division=0)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}% | "
        f"Precision: {precision:.2f} | "
        f"Recall: {recall:.2f}"
    )

# ---------------- SAVE ----------------
torch.save(model.state_dict(), "cnn_gate.pth")
print("✅ Saved cnn_gate.pth")

