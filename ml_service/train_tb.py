#!/usr/bin/env python3
"""
Entraînement classifieur TB (Montgomery + Shenzhen).

Structure attendue :
  data/tb/
    train/
      TB/       *.png
      NORMAL/   *.png
    val/
      TB/
      NORMAL/

Usage:
  python train_tb.py --data_dir data/tb --epochs 20
  → models/custom/tb_classifier.pt
"""
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from pipelines.tb_model import TBDenseNetClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_SIZE = 224


class TBFolderDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.samples = []
        root = Path(root)
        for label, sub in enumerate(("NORMAL", "TB")):
            folder = root / sub
            if not folder.is_dir():
                continue
            for p in folder.glob("*"):
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    self.samples.append((str(p), float(label)))
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        logger.info("Dataset %s: %d images", root, len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return self.transform(img), torch.tensor([label], dtype=torch.float32)


def train(data_dir: str, epochs: int, batch_size: int, save_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TBFolderDataset(f"{data_dir}/train")
    val_ds = TBFolderDataset(f"{data_dir}/val")
    if len(train_ds) == 0:
        raise SystemExit("No training images. See train_tb.py docstring for folder layout.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TBDenseNetClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val = 0.0
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = (model(x) >= 0.5).float()
                correct += (pred == y).sum().item()
                total += y.numel()

        acc = correct / max(total, 1)
        logger.info("Epoch %d/%d loss=%.4f val_acc=%.4f", epoch, epochs, train_loss / len(train_loader), acc)
        if acc >= best_val:
            best_val = acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_classes": 1,
                    "meta": {"datasets": ["Montgomery", "Shenzhen"], "task": "tb_binary"},
                },
                save_path,
            )
            logger.info("Saved %s", save_path)

    print(f"Done. Best val acc: {best_val:.4f} → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/tb")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_path", default="models/custom/tb_classifier.pt")
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.save_path)
