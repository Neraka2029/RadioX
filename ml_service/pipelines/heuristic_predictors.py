"""
Prédicteurs heuristiques pour pathologies sans modèle spécialisé chargé.
Utilisés en fallback lorsque aucun checkpoint .pt/.pth n'est disponible.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def normalize_probability(score: float) -> float:
    """Garantit un score dans [0, 1] pour tous les outputs API."""
    return round(float(np.clip(float(score), 0.0, 1.0)), 4)


def _prob_for(results: List[Dict], pathology: str) -> float:
    for r in results:
        if r.get("pathology") == pathology:
            return float(r.get("probability", 0.0))
    return 0.0


def compute_tuberculosis_risk_raw(nih_results: List[Dict]) -> float:
    """Indicateur de risque TB dérivé des scores NIH (sans seuil)."""
    consolidation = _prob_for(nih_results, "Consolidation")
    infiltration = _prob_for(nih_results, "Infiltration")
    effusion = _prob_for(nih_results, "Epanchement pleural")
    nodule = _prob_for(nih_results, "Nodule")
    raw = (
        infiltration * 0.30
        + consolidation * 0.30
        + effusion * 0.25
        + nodule * 0.15
    )
    return normalize_probability(raw)


def compute_tuberculosis_heuristic(
    nih_results: List[Dict],
    threshold: float = 0.60,
) -> float:
    """Score affiché TB (fallback NIH) — 0 si sous seuil."""
    tb_prob = compute_tuberculosis_risk_raw(nih_results)
    if tb_prob < threshold:
        return 0.0
    logger.debug("TB risk indicator (NIH): %.4f (threshold=%.2f)", tb_prob, threshold)
    return tb_prob


def compute_fracture_risk_raw(nih_results: List[Dict]) -> float:
    """
    Indicateur de risque fracture (fallback NIH uniquement).
    Basé sur : pneumothorax, épanchement, opacité pulmonaire, atélectasie.
    Pas de bounding boxes — aide au triage uniquement.
    """
    pneumothorax = _prob_for(nih_results, "Pneumothorax")
    effusion = _prob_for(nih_results, "Epanchement pleural")
    lung_opacity = max(
        _prob_for(nih_results, "Infiltration"),
        _prob_for(nih_results, "Consolidation"),
    )
    atelectasis = _prob_for(nih_results, "Atelectasie")
    raw = (
        pneumothorax * 0.35
        + effusion * 0.30
        + lung_opacity * 0.20
        + atelectasis * 0.15
    )
    return normalize_probability(raw)


def compute_fracture_heuristic(
    nih_results: List[Dict],
    threshold: float = 0.55,
) -> float:
    """Score affiché fracture risk (fallback) — 0 si sous seuil."""
    risk = compute_fracture_risk_raw(nih_results)
    if risk < threshold:
        return 0.0
    logger.debug("Fracture risk indicator (NIH): %.4f (threshold=%.2f)", risk, threshold)
    return risk


def make_additional_result(
    radiox_label: str,
    probability: float,
    severity_fn: Callable[[float], str],
    descriptions: Dict[str, str],
    colors: Dict[str, str],
) -> Optional[Dict]:
    """Construit une entrée résultat API ou None si probabilité nulle."""
    if probability <= 0:
        return None
    return {
        "pathology": radiox_label,
        "probability": probability,
        "severity": severity_fn(probability),
        "description": descriptions.get(radiox_label, ""),
        "color": colors.get(radiox_label, "#ffffff"),
    }
