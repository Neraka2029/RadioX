"""
Support clinique secondaire via TorchXRayVision densenet121-res224-all.

Le modèle NIH reste la source officielle des prédictions et du Grad-CAM.
Ce module enrichit uniquement les scores TB, fracture et les probabilités dérivées.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from pipelines.heuristic_predictors import normalize_probability

logger = logging.getLogger(__name__)

TB_XRV_WEIGHTS = {
    "Infiltration": 0.20,
    "Consolidation": 0.20,
    "Nodule": 0.15,
    "Fibrosis": 0.15,
    "Pleural_Thickening": 0.15,
    "Effusion": 0.15,
}

FRACTURE_XRV_WEIGHTS = {
    "Fracture": 0.35,
    "Pneumothorax": 0.30,
    "Pleural_Thickening": 0.15,
    "Lung Opacity": 0.10,
    "Effusion": 0.10,
}

DISPLAY_MIN_SCORE = 0.05


class XRVAllSupport:
    """Singleton lazy — densenet121-res224-all pour enrichissement clinique."""

    _instance: Optional["XRVAllSupport"] = None
    _WEIGHTS = "densenet121-res224-all"

    def __new__(cls, device: Optional[torch.device] = None) -> "XRVAllSupport":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, device: Optional[torch.device] = None):
        if getattr(self, "_initialized", False):
            return
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[nn.Module] = None
        self.labels: Optional[List[str]] = None
        self._load_error: Optional[str] = None
        self._initialized = True

    def load_once(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import torchxrayvision as xrv

            model = xrv.models.DenseNet(weights=self._WEIGHTS)
            self.labels = list(model.pathologies)
            self._model = model.to(self.device).eval()
            logger.info(
                "XRV-ALL support loaded (%d labels, device=%s)",
                len(self.labels),
                self.device,
            )
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.error("XRV-ALL support load failed: %s", e)
            return False

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @torch.no_grad()
    def predict_scores(self, tensor: torch.Tensor) -> Optional[Dict[str, float]]:
        """
        Inférence ALL sur tenseur (1,1,224,224) normalisé xrv.

        Returns:
            Dict avec raw_scores, tuberculosis_probability, fracture_risk_score
            ou None si le modèle est indisponible.
        """
        if not self.load_once():
            return None
        try:
            x = tensor.to(self.device)
            if x.dim() != 4 or x.shape[2:] != (224, 224):
                raise ValueError(f"Expected (*, *, 224, 224), got {tuple(x.shape)}")

            output = self._model(x)
            probs = np.clip(output.squeeze().detach().cpu().numpy(), 0.0, 1.0)
            raw_scores = {
                self.labels[i]: round(float(probs[i]), 4)
                for i in range(len(self.labels))
            }

            tb_prob = self._compute_tuberculosis(raw_scores)
            fracture_prob = self._compute_fracture(raw_scores)

            logger.info(
                "XRV-ALL scores: TB=%.4f fracture_risk=%.4f",
                tb_prob,
                fracture_prob,
            )
            return {
                "raw_scores": raw_scores,
                "tuberculosis_probability": tb_prob,
                "fracture_risk_score": fracture_prob,
            }
        except Exception as e:
            logger.error("XRV-ALL predict_scores failed: %s", e, exc_info=True)
            return None

    def _compute_tuberculosis(self, raw_scores: Dict[str, float]) -> float:
        parts = {k: raw_scores.get(k, 0.0) for k in TB_XRV_WEIGHTS}
        raw = sum(parts[k] * w for k, w in TB_XRV_WEIGHTS.items())
        return normalize_probability(raw)

    def _compute_fracture(self, raw_scores: Dict[str, float]) -> float:
        raw = sum(
            raw_scores.get(k, 0.0) * w for k, w in FRACTURE_XRV_WEIGHTS.items()
        )
        return normalize_probability(raw)

    @staticmethod
    def display_score(raw: float, min_score: float = DISPLAY_MIN_SCORE) -> float:
        """Score affiché dans predictions[] — 0 si sous le seuil UI."""
        if raw < min_score:
            return 0.0
        return raw
