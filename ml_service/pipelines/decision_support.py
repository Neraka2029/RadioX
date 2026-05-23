"""
Couche d'aide à la décision — TB / fracture (non diagnostique).

Principe : NO MODEL FAILURE SHOULD BREAK THE SYSTEM
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

TB_SUSPECT_THRESHOLD = 0.50
FRACTURE_RISK_THRESHOLD = 0.45

DISCLAIMER = (
    "Aide à la décision uniquement — ne remplace pas l'interprétation d'un radiologue."
)


def build_decision_support(
    *,
    tuberculosis_probability: float,
    tuberculosis_display_score: float,
    tuberculosis_mode: str,
    fracture_risk_score: float,
    fracture_display_score: float,
    fracture_mode: str,
    fracture_detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Construit decision_support + alertes structurées."""
    tb_suspected = tuberculosis_probability >= TB_SUSPECT_THRESHOLD
    fracture_suspected = (
        len(fracture_detections) > 0
        or (fracture_mode != "none" and fracture_risk_score >= FRACTURE_RISK_THRESHOLD)
    )

    actions: List[str] = []
    alerts: List[str] = [DISCLAIMER]

    if tuberculosis_mode == "xrv-all":
        alerts.append(
            "Tuberculose : score dérivé TorchXRayVision ALL (aide à la décision, non confirmatoire)."
        )
    elif tuberculosis_mode == "heuristic":
        alerts.append(
            "Tuberculose : indicateur de risque dérivé des scores NIH (XRV-ALL indisponible)."
        )

    if fracture_mode == "xrv-all":
        alerts.append(
            "Fracture : proxy de risque traumatique thoracique (XRV-ALL) — aucune localisation osseuse."
        )
    elif fracture_mode == "heuristic":
        alerts.append(
            "Fracture : indicateur de risque basé sur anomalies NIH — aucune localisation osseuse."
        )

    if tb_suspected:
        actions.append(
            "Suspicion TB — envisager dépistage (BK crachats, IGRA) et corrélation clinique."
        )
    if fracture_suspected:
        if fracture_detections:
            actions.append(
                "Zones suspectes sur radiographie — corrélation clinique et imagerie complémentaire si indiqué."
            )
        else:
            actions.append(
                "Risque traumatique thoracique possible — corrélation clinique recommandée (pas de fracture localisée par IA)."
            )

    recommended = (
        " | ".join(actions)
        if actions
        else "Aucun signal TB/fracture au-delà des seuils de triage — interprétation radiologique standard."
    )

    support = {
        "tb_suspected": tb_suspected,
        "fracture_suspected": fracture_suspected,
        "recommended_action": recommended,
        "tb_threshold": TB_SUSPECT_THRESHOLD,
        "fracture_threshold": FRACTURE_RISK_THRESHOLD,
        "tuberculosis_mode": tuberculosis_mode,
        "fracture_mode": fracture_mode,
        "tuberculosis_display_score": tuberculosis_display_score,
        "fracture_display_score": fracture_display_score,
    }
    logger.info(
        "Decision support: tb_suspected=%s (mode=%s) fracture_suspected=%s (mode=%s)",
        tb_suspected,
        tuberculosis_mode,
        fracture_suspected,
        fracture_mode,
    )
    return {"decision_support": support, "alerts": alerts}
