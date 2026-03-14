"""Reports router"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import Analysis, User
from routers.auth import get_current_user
from datetime import datetime

router = APIRouter()


@router.get("/generate/{analysis_id}")
async def generate_report(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a clinical report for an analysis."""
    analysis = (
        db.query(Analysis)
        .filter(Analysis.analysis_id == analysis_id, Analysis.user_id == current_user.id)
        .first()
    )
    if not analysis:
        raise HTTPException(status_code=404, detail="Analyse non trouvée")

    predictions = analysis.predictions or []
    top_findings = sorted(predictions, key=lambda x: x.get("probability", 0), reverse=True)[:3]

    report = {
        "report_id": f"RPT-{analysis_id}",
        "analysis_id": analysis_id,
        "generated_at": datetime.utcnow().isoformat(),
        "physician": {"name": current_user.name, "role": current_user.role},
        "summary": f"Analyse radiographique thoracique - Résultat principal: {analysis.primary_finding}",
        "findings": [
            {
                "pathology": p.get("pathology"),
                "probability": p.get("probability"),
                "severity": p.get("severity"),
                "interpretation": _interpret_finding(p),
            }
            for p in top_findings
        ],
        "recommendations": analysis.recommendations or [],
        "disclaimer": "Ce rapport est généré par assistance IA et doit être validé par un médecin qualifié.",
        "model_info": {
            "version": analysis.model_version,
            "confidence": analysis.confidence,
            "processing_time_ms": analysis.processing_time_ms,
        },
    }
    return report


def _interpret_finding(prediction: dict) -> str:
    prob = prediction.get("probability", 0)
    pathology = prediction.get("pathology", "")
    if prob > 0.7:
        return f"Forte probabilité de {pathology} détectée. Consultation spécialisée urgente."
    elif prob > 0.4:
        return f"Signe modéré compatible avec {pathology}. Suivi clinique recommandé."
    elif prob > 0.2:
        return f"Faibles signes de {pathology}. Surveillance à envisager."
    else:
        return f"Faible probabilité de {pathology}."
