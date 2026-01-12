import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class NumberDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir), key=lambda x: int(x))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            for img in os.listdir(os.path.join(root_dir, cls)):
                self.samples.append(
                    (os.path.join(root_dir, cls, img), self.class_to_idx[cls])
                )

        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

