"""
Scores dérivés Tuberculose / Fracture via TorchXRayVision DenseNet-ALL (singleton).

Ce ne sont pas des labels diagnostiques natifs RadioX — combinaisons contrôlées
à partir des sorties du modèle densenet121-res224-all.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

TB_XRV_WEIGHTS = {
    "Infiltration": 0.30,
    "Consolidation": 0.30,
    "Effusion": 0.25,
    "Nodule": 0.15,
}

# Fracture costale : proxy (Fracture XRV = signal partiel, pas diagnostic osseux seul)
FRACTURE_XRV_WEIGHTS = {
    "Fracture": 0.35,
    "Pneumothorax": 0.30,
    "Pleural_Thickening": 0.15,
    "Lung Opacity": 0.10,
    "Effusion": 0.10,
}

FRACTURE_WEAK_SIGNAL_LABELS = (
    "Pleural_Thickening",
    "Lung Opacity",
    "Atelectasis",
)


class XRVAllPredictor:
    """Singleton lazy — modèle densenet121-res224-all, chargé une seule fois."""

    _WEIGHTS = "densenet121-res224-all"

    def __init__(self, device: torch.device):
        self.device = device
        self._model: Optional[nn.Module] = None
        self.labels: Optional[List[str]] = None
        self._available = False
        self._load_error: Optional[str] = None

    def load(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import torchxrayvision as xrv

            model = xrv.models.DenseNet(weights=self._WEIGHTS)
            self.labels = list(model.pathologies)
            self._model = model.to(self.device).eval()
            self._available = True
            logger.info(
                "TorchXRayVision ALL loaded (singleton, %d labels, device=%s)",
                len(self.labels),
                self.device,
            )
            return True
        except Exception as e:
            self._load_error = str(e)
            self._available = False
            logger.error("TorchXRayVision ALL load failed: %s", e)
            return False

    @property
    def is_available(self) -> bool:
        return self._available and self._model is not None

    @torch.no_grad()
    def predict_raw_scores(self, tensor: torch.Tensor) -> Optional[Dict[str, float]]:
        """Inférence ALL sur tenseur (1,1,224,224) déjà normalisé xrv."""
        if not self.load():
            return None
        try:
            x = tensor.to(self.device)
            if x.dim() != 4 or x.shape[2:] != (224, 224):
                raise ValueError(f"Expected (*, *, 224, 224), got {tuple(x.shape)}")
            output = self._model(x)
            probs = np.clip(output.squeeze().detach().cpu().numpy(), 0.0, 1.0)
            scores = {
                self.labels[i]: round(float(probs[i]), 4)
                for i in range(len(self.labels))
            }
            logger.info("XRV-ALL inference OK (%d labels)", len(scores))
            return scores
        except Exception as e:
            logger.error("XRV-ALL inference failed: %s", e, exc_info=True)
            return None


def compute_tuberculosis_from_xrv(
    raw_scores: Dict[str, float],
    threshold: float = 0.60,
) -> float:
    """Score TB dérivé (infiltration + consolidation + effusion + nodule)."""
    parts = {k: raw_scores.get(k, 0.0) for k in TB_XRV_WEIGHTS}
    score = round(sum(parts[k] * w for k, w in TB_XRV_WEIGHTS.items()), 4)
    logger.info(
        "TB derived (XRV-ALL): infiltration=%.4f consolidation=%.4f effusion=%.4f nodule=%.4f -> %.4f",
        parts["Infiltration"],
        parts["Consolidation"],
        parts["Effusion"],
        parts["Nodule"],
        score,
    )
    return score if score >= threshold else 0.0


def compute_fracture_from_xrv(
    raw_scores: Dict[str, float],
    threshold: float = 0.55,
) -> float:
    """Proxy fracture : anomalies thoraciques + composante signal faible contrôlée."""
    weighted = sum(
        raw_scores.get(k, 0.0) * w for k, w in FRACTURE_XRV_WEIGHTS.items()
    )
    weak_vals = [raw_scores.get(k, 0.0) for k in FRACTURE_WEAK_SIGNAL_LABELS]
    weak_signal = (sum(weak_vals) / max(len(weak_vals), 1)) * 0.12
    score = round(min(1.0, weighted + weak_signal), 4)
    logger.info(
        "Fracture proxy (XRV-ALL): fracture=%.4f pneumothorax=%.4f weak=%.4f -> %.4f",
        raw_scores.get("Fracture", 0.0),
        raw_scores.get("Pneumothorax", 0.0),
        weak_signal,
        score,
    )
    return score if score >= threshold else 0.0


def build_derived_alerts(
    tuberculose_score: float,
    fracture_score: float,
    source: str,
) -> List[str]:
    alerts: List[str] = []
    if source == "heuristic":
        alerts.append(
            "Scores TB/Fracture calculés via heuristique (TorchXRayVision ALL indisponible)."
        )
    if tuberculose_score >= 0.50:
        alerts.append(
            f"Score tuberculose dérivé élevé ({tuberculose_score:.0%}) — corrélation radiologique, non confirmatoire."
        )
    if fracture_score >= 0.45:
        alerts.append(
            f"Score fracture costale (proxy) élevé ({fracture_score:.0%}) — validation clinique recommandée."
        )
    return alerts
