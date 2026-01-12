import sys
import torch
import torchvision.transforms as T
from PIL import Image
from model import DigitCNN   # ← CHANGE THIS if needed

MODEL_PATH = "cnn_1_90_v1.pth"
IMG_SIZE = 64
device = "cpu"

model = DigitCNN(num_classes=90).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

img_path = sys.argv[1]
img = Image.open(img_path).convert("L")  # GRAYSCALE
x = transform(img).unsqueeze(0)

with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(dim=1).item() + 1  # labels 1–90

print(f"Predicted Number: {pred}")

