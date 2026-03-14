"""
RadioX.AI - Model Training Script
Train DenseNet-121 on NIH ChestX-ray14 or CheXpert dataset.

Usage:
    python train.py --dataset nih --data_dir /path/to/data --epochs 50

Dataset options:
    - nih: NIH ChestX-ray14 (https://nihcc.app.box.com/v/ChestXray-NIHCC)
    - chexpert: CheXpert (https://stanfordmlgroup.github.io/competitions/chexpert/)
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
import logging
import json
from tqdm import tqdm
from PIL import Image
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipelines.inference import ChestXRayModel, PATHOLOGY_LABELS, IMAGE_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChestXRayDataset(Dataset):
    """
    Dataset class compatible with NIH ChestX-ray14 and CheXpert.
    
    Expected CSV format:
    - NIH: Image Index, Finding Labels, ...
    - CheXpert: Path, Atelectasis, Cardiomegaly, ...
    """

    NIH_TO_MODEL = {
        "No Finding": "Normal",
        "Pneumonia": "Pneumonie",
        "Pleural Effusion": "Épanchement pleural",
        "Cardiomegaly": "Cardiomégalie",
        "Mass": "Cancer du poumon",
        "Nodule": "Cancer du poumon",
        "Infiltration": "Tuberculose",
    }

    def __init__(self, csv_path: str, image_dir: str, dataset_type: str = "nih"):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.dataset_type = dataset_type

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.labels = self._parse_labels()
        logger.info(f"Dataset loaded: {len(self.df)} samples")

    def _parse_labels(self) -> np.ndarray:
        """Parse labels from CSV into multi-hot numpy array."""
        labels = np.zeros((len(self.df), len(PATHOLOGY_LABELS)), dtype=np.float32)

        if self.dataset_type == "nih":
            for i, row in self.df.iterrows():
                findings = str(row.get("Finding Labels", "No Finding")).split("|")
                for finding in findings:
                    finding = finding.strip()
                    model_label = self.NIH_TO_MODEL.get(finding)
                    if model_label and model_label in PATHOLOGY_LABELS:
                        idx = PATHOLOGY_LABELS.index(model_label)
                        labels[i, idx] = 1.0

        return labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.dataset_type == "nih":
            img_name = self.df.iloc[idx]["Image Index"]
        else:
            img_name = self.df.iloc[idx]["Path"]

        img_path = self.image_dir / img_name
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = torch.FloatTensor(self.labels[idx])
        return img, label


class Trainer:
    """Training loop with AUC evaluation."""

    def __init__(self, model, train_loader, val_loader, device, lr=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Binary Cross-Entropy Loss for multi-label classification
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, factor=0.5
        )

        self.best_auc = 0.0
        self.history = []

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_imgs, batch_labels in tqdm(self.train_loader, desc="Training"):
            batch_imgs = batch_imgs.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_imgs)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0

        for batch_imgs, batch_labels in tqdm(self.val_loader, desc="Validation"):
            batch_imgs = batch_imgs.to(self.device)
            batch_labels = batch_labels.to(self.device)

            outputs = self.model(batch_imgs)
            loss = self.criterion(outputs, batch_labels)
            total_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Per-class AUC
        aucs = {}
        for i, label in enumerate(PATHOLOGY_LABELS):
            if all_labels[:, i].sum() > 0:
                aucs[label] = roc_auc_score(all_labels[:, i], all_preds[:, i])

        mean_auc = np.mean(list(aucs.values())) if aucs else 0.0
        return {
            "loss": total_loss / len(self.val_loader),
            "mean_auc": mean_auc,
            "per_class_auc": aucs,
        }

    def train(self, epochs: int, save_path: str = "models/chestxray_densenet121.pth"):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{epochs}")

            train_loss = self.train_epoch()
            val_metrics = self.evaluate()

            self.scheduler.step(val_metrics["loss"])

            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Mean AUC: {val_metrics['mean_auc']:.4f}")

            for pathology, auc in val_metrics["per_class_auc"].items():
                logger.info(f"  {pathology}: AUC={auc:.4f}")

            # Save best model
            if val_metrics["mean_auc"] > self.best_auc:
                self.best_auc = val_metrics["mean_auc"]
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"✓ New best model saved (AUC={self.best_auc:.4f})")

            self.history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **val_metrics,
            })

        # Save training history
        with open("models/training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"\nTraining complete. Best AUC: {self.best_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="RadioX.AI Model Training")
    parser.add_argument("--dataset", choices=["nih", "chexpert"], default="nih")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset images")
    parser.add_argument("--csv_train", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--csv_val", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_path", type=str, default="models/chestxray_densenet121.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # Datasets
    train_dataset = ChestXRayDataset(args.csv_train, args.data_dir, args.dataset)
    val_dataset = ChestXRayDataset(args.csv_val, args.data_dir, args.dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = ChestXRayModel(num_classes=len(PATHOLOGY_LABELS))

    # Train
    trainer = Trainer(model, train_loader, val_loader, device, lr=args.lr)
    trainer.train(epochs=args.epochs, save_path=args.save_path)


if __name__ == "__main__":
    main()
