"""
RadioX.AI - ML Inference Service
PyTorch + MONAI pipeline for chest X-ray analysis
"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Optional
from datetime import datetime

from pipelines.inference import InferencePipeline
from pipelines.gradcam import GradCAMPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RadioX.AI ML Service",
    description="Service d'inférence IA pour l'analyse de radiographies thoraciques",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize pipelines at startup
inference_pipeline: Optional[InferencePipeline] = None
gradcam_pipeline: Optional[GradCAMPipeline] = None

# Statistics storage
stats = {
    "analyses_today": 0,
    "analyses_total": 0,
    "pathologies_detected": 0,
    "avg_confidence": 0,
    "today_pathologies": {},
    "total_confidence": 0,
    "confidence_count": 0,
}

# Analysis history storage
analysis_history = []
patient_counter = 1
analysis_images = {}  # Stockage des images par analysis_id


@app.on_event("startup")
async def startup():
    global inference_pipeline, gradcam_pipeline
    logger.info("Loading ML models...")
    try:
        inference_pipeline = InferencePipeline()
        gradcam_pipeline = GradCAMPipeline(inference_pipeline.model)
        logger.info(f"ML models loaded successfully. Version: {"DenseNet121-NIH-xrv"}")
    except Exception as e:
        logger.error(f"Failed to load ML models: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Log des routes disponibles
    logger.info("Available routes:")
    for route in app.routes:
        logger.info(f"  {route.methods} {route.path}")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    analysis_id: str = Form(default=""),
):
    """Run inference on a chest X-ray image."""
    content = await file.read()

    if inference_pipeline is None:
        return _fallback_prediction(analysis_id)

    try:
        # Preprocess
        tensor = inference_pipeline.preprocess(content)

        # Inference
        predictions = inference_pipeline.predict(tensor)
        
        # Mettre à jour les statistiques
        _update_stats(predictions, analysis_id, content)
        
        # Log des prédictions pour debug
        logger.info(f"Predictions: {predictions[:3]}...")  # Premieres 3 prédictions

        # Grad-CAM heatmap
        heatmap_b64 = None
        if gradcam_pipeline:
            try:
                heatmap_b64 = gradcam_pipeline.generate(tensor, predictions)
                logger.info(f"Grad-CAM heatmap generated successfully, length: {len(heatmap_b64) if heatmap_b64 else 0}")
            except Exception as e:
                logger.error(f"Grad-CAM generation failed: {e}")
                heatmap_b64 = None

        return {
            "analysis_id": analysis_id,
            "predictions": predictions,
            "primary_finding": max(predictions, key=lambda x: x["probability"])["pathology"],
            "confidence": max(p["probability"] for p in predictions),
            "model_version": "DenseNet121-NIH-xrv",
            "heatmap_base64": heatmap_b64,
            "recommendations": _generate_recommendations(predictions),
        }
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return _fallback_prediction(analysis_id)


@app.get("/analysis/{analysis_id}/image")
async def get_analysis_image(analysis_id: str):
    """Get the image for a specific analysis."""
    global analysis_images
    
    logger.info(f"=== IMAGE REQUEST - Analysis ID: {analysis_id} ===")
    
    if analysis_id in analysis_images:
        import base64
        image_data = analysis_images[analysis_id]
        
        # Retourner l'image en base64
        return {
            "success": True,
            "image_base64": image_data,
            "format": "jpeg"
        }
    else:
        return {
            "success": False,
            "error": "Image not found for this analysis"
        }


@app.get("/history")
async def get_history():
    """Get analysis history."""
    global analysis_history
    
    logger.info(f"=== HISTORY REQUEST - Returning {len(analysis_history)} analyses ===")
    
    # Formatter les données pour le frontend
    formatted_history = []
    for record in analysis_history:
        formatted_record = {
            "id": record["id"],
            "date": record["date"],
            "patient": record["patient_id"],  # Utiliser patient_id au lieu de générer
            "primaryFinding": record["primary_finding"],
            "confidence": record["confidence"],
            "status": record["status"],
            "imageCount": record["image_count"],
            "predictions": record["predictions"],
            "detectedPathologies": record["detected_pathologies"]
        }
        formatted_history.append(formatted_record)
    
    return {
        "analyses": formatted_history,
        "total": len(formatted_history),
        "stats": {
            "avg_confidence": stats["avg_confidence"],
            "total_analyses": stats["analyses_total"]
        }
    }


@app.get("/reports")
async def get_reports():
    """Get reports based on analysis history."""
    global analysis_history
    
    logger.info(f"=== REPORTS REQUEST - Generating reports from {len(analysis_history)} analyses ===")
    
    # Générer des rapports à partir de l'historique
    reports = []
    for i, record in enumerate(analysis_history[:10]):  # Limiter aux 10 plus récents
        report = {
            "id": f"RPT-{datetime.now().strftime('%Y')}-{i+1:03d}",
            "title": f"Analyse thoracique - {record['primary_finding']}",
            "patient": record["patient_id"],  # Utiliser patient_id
            "date": record["date"].split()[0],  # Just la date
            "type": "analyse_complete",
            "findings": [p["pathology"] for p in record["detected_pathologies"][:3]],  # Top 3 pathologies
            "confidence": record["confidence"],
            "images": record["image_count"],
            "status": "completed" if record["confidence"] > 50 else "review",
            "analysis_id": record["id"],
            "predictions": record["predictions"]
        }
        reports.append(report)
    
    return {
        "reports": reports,
        "total": len(reports)
    }


@app.get("/stats")
async def get_stats():
    """Get real-time statistics from the ML service."""
    try:
        global stats
        
        logger.info("=== STATS REQUEST RECEIVED ===")
        logger.info(f"Current stats: {stats}")
        
        # Calculer la confiance moyenne
        avg_confidence = 0
        if stats["confidence_count"] > 0:
            avg_confidence = round((stats["total_confidence"] / stats["confidence_count"]) * 100)
        
        # Compter les pathologies uniques détectées aujourd'hui
        unique_pathologies = len([p for p, count in stats["today_pathologies"].items() if count > 0])
        
        result = {
            "analyses_today": stats["analyses_today"],
            "analyses_total": stats["analyses_total"],
            "pathologies_detected": unique_pathologies,
            "avg_confidence": avg_confidence,
            "model_version": "DenseNet121-NIH-xrv" if inference_pipeline else "not_loaded",
            "model_accuracy": 0.81,
        }
        
        logger.info(f"Stats result: {result}")
        logger.info("=== STATS REQUEST FINISHED ===")
        return result
        
    except Exception as e:
        logger.error(f"Error in /stats: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "analyses_today": 0,
            "analyses_total": 0,
            "pathologies_detected": 0,
            "avg_confidence": 0,
            "model_version": "error",
            "model_accuracy": 0.0,
        }


@app.get("/health")
async def health():
    return {
        "status": "operational",
        "model_loaded": inference_pipeline is not None,
        "model_version": "DenseNet121-NIH-xrv" if inference_pipeline else "not_loaded",
    }


def _fallback_prediction(analysis_id: str) -> dict:
    """Return a structured error response."""
    return {
        "analysis_id": analysis_id,
        "error": "Model not loaded",
        "predictions": [],
        "primary_finding": "Indéterminé",
        "confidence": 0.0,
        "model_version": "not_loaded",
        "heatmap_base64": None,
        "recommendations": ["Modèle IA non disponible. Veuillez réessayer ultérieurement."],
    }


def _update_stats(predictions: list, analysis_id: str = None, image_data: bytes = None):
    """Met à jour les statistiques après chaque analyse."""
    global stats, analysis_history, patient_counter, analysis_images
    
    logger.info("=== UPDATE STATS CALLED ===")
    logger.info(f"Current stats before update: {stats}")
    logger.info(f"Predictions received: {len(predictions)}")
    
    # Incrémenter les compteurs d'analyses
    stats["analyses_today"] += 1
    stats["analyses_total"] += 1
    
    # Calculer la confiance de cette analyse
    max_confidence = 0
    primary_finding = "Normal"
    if predictions:
        max_confidence = max(p["probability"] for p in predictions)
        primary_pred = max(predictions, key=lambda x: x["probability"])
        primary_finding = primary_pred["pathology"]
        stats["total_confidence"] += max_confidence
        stats["confidence_count"] += 1
        logger.info(f"Max confidence: {max_confidence}")
    
    # Compter les pathologies détectées (probabilité > 0.1)
    detected_count = 0
    detected_pathologies = []
    for pred in predictions:
        if pred["probability"] > 0.1 and pred["pathology"] != "Normal":
            pathology = pred["pathology"]
            if pathology not in stats["today_pathologies"]:
                stats["today_pathologies"][pathology] = 0
            stats["today_pathologies"][pathology] += 1
            detected_count += 1
            detected_pathologies.append({
                "pathology": pathology,
                "probability": pred["probability"],
                "severity": pred.get("severity", "low")
            })
    
    # Ajouter à l'historique
    current_patient = f"P{patient_counter:04d}"  # P0001, P0002, etc.
    analysis_record = {
        "id": analysis_id or f"AX-{datetime.now().strftime('%Y%m%d')}-{len(analysis_history) + 1:03d}",
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "patient_id": current_patient,
        "primary_finding": primary_finding,
        "confidence": round(max_confidence * 100),
        "predictions": predictions,
        "detected_pathologies": detected_pathologies,
        "status": "completed",
        "image_count": 1
    }
    
    analysis_history.insert(0, analysis_record)  # Ajouter au début (plus récent d'abord)
    patient_counter += 1  # Incrémenter pour la prochaine analyse
    
    # Sauvegarder l'image si fournie
    if analysis_id and image_data:
        import base64
        analysis_images[analysis_id] = base64.b64encode(image_data).decode('utf-8')
        logger.info(f"Image sauvegardée pour l'analyse {analysis_id}")
    
    logger.info(f"Pathologies detected this analysis: {detected_count}")
    logger.info(f"Stats updated: {stats['analyses_today']} analyses today")
    logger.info(f"History size: {len(analysis_history)} analyses")
    logger.info("=== UPDATE STATS FINISHED ===")


def _generate_recommendations(predictions: list) -> list:
    """Generate clinical recommendations based on predictions."""
    recommendations = []
    for pred in sorted(predictions, key=lambda x: x["probability"], reverse=True):
        prob = pred["probability"]
        pathology = pred["pathology"]

        if pathology == "Normal" and prob > 0.7:
            recommendations.append("Aucune anomalie détectée. Suivi de routine recommandé.")
            break
        elif pathology == "Pneumonie" and prob > 0.5:
            recommendations.append("Pneumonie probable - Antibiothérapie à évaluer selon clinique.")
            recommendations.append("Suivi radiographique à 4-6 semaines recommandé.")
        elif pathology == "Tuberculose" and prob > 0.3:
            recommendations.append("Suspicion de tuberculose - Prélèvements bacillaires urgents (BK crachats x3).")
            recommendations.append("Isolement respiratoire à envisager selon protocole.")
        elif pathology == "Cancer du poumon" and prob > 0.3:
            recommendations.append("Lésion suspecte - Scanner thoracique injecté urgent.")
            recommendations.append("Avis oncologique à solliciter rapidement.")
        elif pathology == "Épanchement pleural" and prob > 0.4:
            recommendations.append("Épanchement pleural - Évaluation étiologique nécessaire.")
        elif pathology == "Cardiomégalie" and prob > 0.4:
            recommendations.append("Cardiomégalie - Bilan cardiologique recommandé (ECG, écho).")

    if not recommendations:
        recommendations.append("Résultats à corréler avec la clinique et l'anamnèse du patient.")

    recommendations.append(
        "⚠️ Ces résultats sont indicatifs et doivent être validés par un médecin qualifié."
    )
    return recommendations[:4]
