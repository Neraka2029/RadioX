"""
Classifieur binaire Tuberculose (TB vs Normal) — Montgomery / Shenzhen.

Checkpoint attendu : ml_service/models/custom/tb_classifier.pt
Entrée : tenseur (1, 1, 224, 224) normalisé xrv (identique au pipeline NIH).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

CUSTOM_DIR = Path(__file__).resolve().parent.parent / "models" / "custom"
CHECKPOINT_FILES = (
    "tb_classifier.pt",
    "tb_classifier.pth",
    "tuberculosis_classifier.pt",
    "tuberculosis.pt",
    "tuberculosis.pth",
)
TB_THRESHOLD = 0.50


class TBDenseNetClassifier(nn.Module):
    """DenseNet-121 binaire TB (Montgomery / Shenzhen)."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        from torchvision.models import densenet121, DenseNet121_Weights

        base = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        base.classifier = nn.Sequential(
            nn.Linear(base.classifier.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.backbone = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.backbone(x))


class TBModelService:
    """Singleton — chargement unique au démarrage."""

    def __init__(self, device: torch.device):
        self.device = device
        self._model: Optional[nn.Module] = None
        self._checkpoint_path: Optional[Path] = None
        self._load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False

        path = self._resolve_checkpoint()
        if path is None:
            logger.info(
                "TB classifier not found in %s — heuristic fallback will be used",
                CUSTOM_DIR,
            )
            return False

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = self._build_from_checkpoint(checkpoint)
            model.to(self.device)
            model.eval()
            self._model = model
            self._checkpoint_path = path
            logger.info("TB classifier loaded from %s (device=%s)", path.name, self.device)
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.error("TB classifier load failed (%s): %s", path, e)
            return False

    @staticmethod
    def _resolve_checkpoint() -> Optional[Path]:
        CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
        for name in CHECKPOINT_FILES:
            p = CUSTOM_DIR / name
            if p.is_file():
                return p
        return None

    def _build_from_checkpoint(self, checkpoint) -> nn.Module:
        if isinstance(checkpoint, nn.Module):
            return checkpoint
        model = TBDenseNetClassifier(num_classes=1)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint and isinstance(checkpoint["model"], nn.Module):
                return checkpoint["model"]
            state = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
            if state is not None:
                model.load_state_dict(state, strict=False)
                return model
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
            return model
        raise ValueError("Unsupported TB checkpoint format")

    @staticmethod
    def _prepare_input(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        x = tensor.to(device)
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    @torch.no_grad()
    def predict_probability(self, tensor: torch.Tensor) -> Optional[float]:
        """Probabilité TB [0, 1] ou None si modèle indisponible / erreur."""
        if not self.load():
            return None
        try:
            x = self._prepare_input(tensor, self.device)
            out = self._model(x)
            if out.dim() > 1:
                out = out.squeeze()
            prob = float(out.item())
            from pipelines.heuristic_predictors import normalize_probability

            prob = normalize_probability(prob)
            logger.info("TB classifier inference: prob=%.4f", prob)
            return prob
        except Exception as e:
            logger.error("TB classifier inference failed: %s", e, exc_info=True)
            return None

    def apply_threshold(self, prob: float, threshold: float = TB_THRESHOLD) -> float:
        return prob if prob >= threshold else 0.0
