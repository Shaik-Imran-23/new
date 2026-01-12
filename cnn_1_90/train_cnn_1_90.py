import torch
from torch.utils.data import DataLoader
from model import DigitCNN
from dataset import NumberDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = NumberDataset("../cnn_1_90_dataset/train")
val_ds   = NumberDataset("../cnn_1_90_dataset/val")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)

model = DigitCNN(num_classes=len(train_ds.classes)).to(DEVICE)

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.3
)

for epoch in range(40):
    model.train()
    correct = total = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.2f}%")

    scheduler.step(loss)

torch.save({
    "model": model.state_dict(),
    "classes": train_ds.classes
}, "cnn_1_90.pth")

print("✅ CNN 1–90 model saved as cnn_1_90.pth")

