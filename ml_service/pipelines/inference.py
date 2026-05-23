"""
RadioX.AI — Inference Pipeline
Modèle : DenseNet-121 pré-entraîné NIH ChestX-ray14 (torchxrayvision)
14 pathologies NIH directes — mapping 1:1 sans approximation
AUC moyen ~0.81
"""
import torch
import numpy as np
from io import BytesIO
from PIL import Image
import logging
import os
from typing import List, Dict, Any

from pipelines.heuristic_predictors import (
    compute_fracture_risk_raw,
    compute_tuberculosis_risk_raw,
    make_additional_result,
)
from pipelines.xrv_all_support import XRVAllSupport, DISPLAY_MIN_SCORE
from pipelines.decision_support import build_decision_support

logger = logging.getLogger(__name__)
IMAGE_SIZE = 224

# ── 14 pathologies NIH + Normal + 2 pathologies additionnelles ─────────────
PATHOLOGY_LABELS = [
    "Normal",
    "Atelectasie",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Oedeme",
    "Emphyseme",
    "Fibrose",
    "Epanchement pleural",
    "Pneumonie",
    "Epaississement pleural",
    "Cardiomegalie",
    "Nodule",
    "Masse",
    "Hernie",
    "Tuberculose",
    "Fracture costale",
]

# Mapping direct NIH label -> label RadioX (1:1)
XRV_TO_RADIOX = {
    "Atelectasis":        "Atelectasie",
    "Consolidation":      "Consolidation",
    "Infiltration":       "Infiltration",
    "Pneumothorax":       "Pneumothorax",
    "Edema":              "Oedeme",
    "Emphysema":          "Emphyseme",
    "Fibrosis":           "Fibrose",
    "Effusion":           "Epanchement pleural",
    "Pneumonia":          "Pneumonie",
    "Pleural_Thickening": "Epaississement pleural",
    "Cardiomegaly":       "Cardiomegalie",
    "Nodule":             "Nodule",
    "Mass":               "Masse",
    "Hernia":             "Hernie",
    # Pathologies additionnelles (mapping pour modèles futurs)
    "Tuberculosis":      "Tuberculose",
    "Fracture":           "Fracture costale",
}

# Labels fiables — inclure Fibrosis pour détecter cette pathologie importante
RELIABLE_LABELS = {
    "Atelectasis", 
    "Consolidation", 
    "Infiltration", 
    "Pneumothorax",
    "Edema", 
    "Effusion",
    "Pneumonia", 
    "Pleural_Thickening",
    "Cardiomegaly", 
    "Nodule", 
    "Mass", 
    "Fibrosis",  # Fibrosis ajouté
}
# Seul Emphysema et Hernia restent exclus car retournent ~0.5 fixe

# Seuil minimum par pathologie (valeurs originales)
THRESHOLDS_PER_PATHOLOGY = {
    "Atelectasis":        0.0391,
    "Consolidation":      0.0035,
    "Infiltration":       0.1140,
    "Pneumothorax":       0.0057,
    "Edema":              0.00046,
    "Effusion":           0.0387,
    "Pneumonia":          0.0037,
    "Pleural_Thickening": 0.0147,
    "Cardiomegaly":       0.0161,
    "Nodule":             0.0542,
    "Mass":               0.0372,
    "Fibrosis":           0.0120,
    "Hernia":             0.00044,
    # Seuils dédiés pour pathologies additionnelles
    "Tuberculosis":      0.60,
    "Fracture":           0.35,
}

SEVERITY_THRESHOLDS = {"high": 0.65, "moderate": 0.50, "low": 0.0}
PATHOLOGY_DESCRIPTIONS = {
    "Normal":                 "Aucune anomalie détectée",
    "Atelectasie":            "Collapse partiel ou total d'un lobe pulmonaire",
    "Consolidation":          "Remplissage des alvéoles — signe de pneumonie typique",
    "Infiltration":           "Opacité diffuse du parenchyme pulmonaire",
    "Pneumothorax":           "Présence d'air dans la cavité pleurale",
    "Oedeme":                 "Accumulation de liquide dans les poumons",
    "Emphyseme":              "Destruction progressive des alvéoles pulmonaires",
    "Fibrose":                "Cicatrisation et durcissement du tissu pulmonaire",
    "Epanchement pleural":    "Accumulation de liquide autour du poumon",
    "Pneumonie":              "Infection et inflammation du parenchyme pulmonaire",
    "Epaississement pleural": "Épaississement de la membrane pleurale",
    "Cardiomegalie":          "Augmentation anormale de la taille du cœur",
    "Nodule":                 "Petite lésion arrondie < 3cm dans le poumon",
    "Masse":                  "Lésion de grande taille > 3cm dans le poumon",
    "Hernie":                 "Saillie d'un organe à travers le diaphragme",
    "Tuberculose":            "Infection bactérienne chronique (Mycobacterium tuberculosis)",
    "Fracture costale":       "Rupture d'un ou plusieurs os de la cage thoracique",
}

PATHOLOGY_COLORS = {
    "Normal":                 "#00ffc8",
    "Atelectasie":            "#ff6b6b",
    "Consolidation":          "#ff9f43",
    "Infiltration":           "#ffd32a",
    "Pneumothorax":           "#ff4757",
    "Oedeme":                 "#5352ed",
    "Emphyseme":              "#ff6348",
    "Fibrose":                "#a29bfe",
    "Epanchement pleural":    "#00d2d3",
    "Pneumonie":              "#ff4466",
    "Epaississement pleural": "#54a0ff",
    "Cardiomegalie":          "#48dbfb",
    "Nodule":                 "#ff9ff3",
    "Masse":                  "#ee5a24",
    "Hernie":                 "#c8d6e5",
    "Tuberculose":            "#ff6b81",
    "Fracture costale":       "#ff7f50",
}


class PreprocessingPipeline:
    """Preprocessing officiel torchxrayvision."""

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        is_dicom = len(image_bytes) > 132 and image_bytes[128:132] == b"DICM"
        logger.info(f"File type detection: DICOM={is_dicom}, file size={len(image_bytes)} bytes")
        
        if is_dicom:
            try:
                import pydicom
                from pydicom.filebase import DicomBytesIO
                logger.info("Processing DICOM file...")
                
                # Essayer de lire avec différentes options
                try:
                    ds = pydicom.dcmread(DicomBytesIO(image_bytes))
                except Exception as e1:
                    logger.warning(f"Standard DICOM read failed: {e1}")
                    # Essayer sans forcer la lecture des pixels
                    ds = pydicom.dcmread(DicomBytesIO(image_bytes), stop_before_pixels=True)
                    logger.info("DICOM header loaded successfully")
                
                logger.info(f"DICOM loaded: {ds.PatientName if hasattr(ds, 'PatientName') else 'Unknown patient'}")
                
                # Essayer différentes méthodes pour obtenir les pixels
                try:
                    arr = ds.pixel_array.astype(np.float32)
                    logger.info(f"Pixel array shape: {arr.shape}, dtype: {arr.dtype}")
                except Exception as e2:
                    logger.error(f"Pixel array extraction failed: {e2}")
                    # Si ça échoue, essayer de convertir en image standard
                    try:
                        # Sauvegarder temporairement et recharger avec PIL
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as tmp:
                            tmp.write(image_bytes)
                            tmp.flush()
                            
                        # Utiliser une bibliothèque externe ou fallback
                        logger.error("Cannot extract DICOM pixels, falling back to standard processing")
                        raise Exception("DICOM pixel extraction failed")
                    except Exception as e3:
                        logger.error(f"All DICOM extraction methods failed: {e3}")
                        raise
                
                if hasattr(ds, "RescaleSlope"):
                    arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                    logger.info(f"Applied rescaling: slope={ds.RescaleSlope}, intercept={ds.RescaleIntercept}")
                    
                arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255).astype(np.uint8)
                if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                    arr = 255 - arr
                    logger.info("Applied MONOCHROME1 inversion")
                    
                if arr.ndim == 3:
                    arr = arr[0]
                    logger.info("Extracted first channel from 3D array")
                    
                img = Image.fromarray(arr)
                logger.info(f"DICOM processing successful: image size={img.size}")
                
                # S'assurer que l'image est en RGB pour l'affichage web
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    logger.info(f"Converted image to RGB mode")
                
                return img
                
            except Exception as e:
                logger.error(f"DICOM processing failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
        logger.info("Falling back to standard image processing")
        return Image.open(BytesIO(image_bytes))

    def process(self, image_bytes: bytes) -> torch.Tensor:
        import torchxrayvision as xrv
        import skimage.transform

        img = self._load_image(image_bytes)
        
        # Convertir en niveaux de gris
        if img.mode == "RGBA":
            img = img.convert("RGB")
        if img.mode == "RGB":
            arr = np.array(img).astype(np.float32)
            arr = 0.2989*arr[:,:,0] + 0.5870*arr[:,:,1] + 0.1140*arr[:,:,2]
        else:
            arr = np.array(img.convert("L")).astype(np.float32)

        # Vérifier si l'image vient d'un DICOM déjà normalisé
        is_dicom = len(image_bytes) > 132 and image_bytes[128:132] == b"DICM"
        if is_dicom:
            logger.info("DICOM detected")
        else:
            logger.info("Standard image detected")
        
        # Preprocessing DICOM unifié - identique pour DICOM et JPEG
        arr = xrv.datasets.normalize(arr, maxval=255, reshape=False)
        arr = skimage.transform.resize(arr, (224, 224),
              anti_aliasing=True, preserve_range=True).astype(np.float32)
        # identique DICOM et JPEG

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        logger.info(f"Preprocessing OK: min={tensor.min():.0f}, max={tensor.max():.0f}, mean={tensor.mean():.0f}")
        return tensor


class InferencePipeline:
    """Pipeline principal utilisant torchxrayvision DenseNet-121 NIH + modèles additionnels."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = PreprocessingPipeline()
        self.xrv_labels = None
        self.model = self._load_model()
        self.model.eval()

        self.xrv_all = XRVAllSupport(self.device)
        self.last_specialized_output = _empty_specialized_output()

        logger.info(f"InferencePipeline prêt sur {self.device}")

    def _load_model(self):
        try:
            import torchxrayvision as xrv
            model = xrv.models.DenseNet(weights="densenet121-res224-nih")
            self.xrv_labels = list(model.pathologies)
            logger.info(f"Modèle NIH chargé — {len(self.xrv_labels)} labels")
            return model.to(self.device)
        except ImportError:
            raise RuntimeError("torchxrayvision non installé. Lancez : pip install torchxrayvision")
        except Exception as e:
            raise RuntimeError(f"Erreur chargement modèle: {e}")

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        self.processed_image = self.preprocessor.process(image_bytes)
        return self.processed_image

    def get_processed_image(self, image_bytes: bytes) -> Image.Image:
        """Retourne l'image traitée pour l'affichage."""
        try:
            return self.preprocessor._load_image(image_bytes)
        except Exception as e:
            logger.error(f"Failed to get processed image: {e}")
            return None

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor) -> List[Dict]:
        output = self.model(tensor.to(self.device))
        return self._build_results(output, tensor)

    def _build_results(self, output: torch.Tensor, tensor: torch.Tensor) -> List[Dict]:
        probs = np.clip(output.squeeze().detach().cpu().numpy(), 0.0, 1.0)
        
        raw_scores = {
            self.xrv_labels[i]: round(float(probs[i]), 4)
            for i in range(len(self.xrv_labels))
        }
        logger.info("NIH raw scores: %s", raw_scores)

        results = []
        max_prob = 0.0

        # Pathologies NIH standard
        for xrv_lbl, radiox_lbl in XRV_TO_RADIOX.items():
            # Ne traiter que les labels présents ET valides dans le modèle
            if xrv_lbl not in self.xrv_labels:
                prob = 0.0
                continue
            
            try:
                idx = self.xrv_labels.index(xrv_lbl)
                prob = round(float(probs[idx]), 4)
            except ValueError:
                prob = 0.0

            # Mettre à 0 les labels non fiables (valeur fixe ~0.5)
            if xrv_lbl not in RELIABLE_LABELS:
                prob = 0.0

            # Appliquer seuil minimum par pathologie
            threshold = THRESHOLDS_PER_PATHOLOGY.get(xrv_lbl, 0.45)
            if prob < threshold:
                prob = 0.0

            max_prob = max(max_prob, prob)
            results.append({
                "pathology":    radiox_lbl,
                "probability":  prob,
                "severity":     self._severity(prob),
                "description":  PATHOLOGY_DESCRIPTIONS.get(radiox_lbl, ""),
                "color":        PATHOLOGY_COLORS.get(radiox_lbl, "#ffffff"),
            })

        # TB + Fracture via XRV-ALL (fallback heuristique NIH si indisponible)
        max_prob = self._append_specialized_pathologies(results, max_prob, tensor)

        # Trier par probabilité décroissante
        results.sort(key=lambda x: x["probability"], reverse=True)

        # Normal = inverse du score max
        normal_prob = round(max(0.02, min(0.98, 1.0 - max_prob)), 4)
        results.insert(0, {
            "pathology":   "Normal",
            "probability": normal_prob,
            "severity":    self._severity(normal_prob),
            "description": PATHOLOGY_DESCRIPTIONS["Normal"],
            "color":       PATHOLOGY_COLORS["Normal"],
        })

        return results

    def _append_specialized_pathologies(
        self,
        results: List[Dict],
        max_prob: float,
        tensor: torch.Tensor,
    ) -> float:
        fracture_detections: List[Dict[str, Any]] = []
        xrv_out = self.xrv_all.predict_scores(tensor)

        if xrv_out is not None:
            tb_probability = float(xrv_out["tuberculosis_probability"])
            fracture_probability = float(xrv_out["fracture_risk_score"])
            tuberculosis_mode = "xrv-all"
            fracture_mode = "xrv-all"
            tb_source = "xrv-all"
            fracture_source = "xrv-all"
            logger.info(
                "Clinical enrichment XRV-ALL: TB=%.4f fracture=%.4f",
                tb_probability,
                fracture_probability,
            )
        else:
            tb_probability = compute_tuberculosis_risk_raw(results)
            fracture_probability = compute_fracture_risk_raw(results)
            tuberculosis_mode = "heuristic"
            fracture_mode = "heuristic" if fracture_probability > 0 else "none"
            tb_source = "heuristic"
            fracture_source = "nih_risk_indicator"
            logger.warning(
                "XRV-ALL unavailable — NIH heuristic fallback TB=%.4f fracture=%.4f",
                tb_probability,
                fracture_probability,
            )

        tb_display = XRVAllSupport.display_score(tb_probability, DISPLAY_MIN_SCORE)
        fracture_display = XRVAllSupport.display_score(
            fracture_probability, DISPLAY_MIN_SCORE
        )

        decision = build_decision_support(
            tuberculosis_probability=tb_probability,
            tuberculosis_display_score=tb_display,
            tuberculosis_mode=tuberculosis_mode,
            fracture_risk_score=fracture_probability,
            fracture_display_score=fracture_display,
            fracture_mode=fracture_mode,
            fracture_detections=fracture_detections,
        )

        self.last_specialized_output = {
            "tuberculosis_probability": tb_probability,
            "tuberculosis_display_score": tb_display,
            "tuberculosis_mode": tuberculosis_mode,
            "tuberculosis_source": tb_source,
            "fracture_detections": fracture_detections,
            "fracture_risk_score": fracture_probability,
            "fracture_display_score": fracture_display,
            "fracture_mode": fracture_mode,
            "fracture_source": fracture_source,
            "alerts": decision["alerts"],
            "decision_support": decision["decision_support"],
            "tuberculose_score": tb_display,
            "fracture_score": fracture_display,
            "derived_source": "xrv-all" if tuberculosis_mode == "xrv-all" else "heuristic",
        }

        for label, prob in (
            ("Tuberculose", tb_display),
            ("Fracture costale", fracture_display),
        ):
            entry = make_additional_result(
                label,
                prob,
                self._severity,
                PATHOLOGY_DESCRIPTIONS,
                PATHOLOGY_COLORS,
            )
            if entry:
                results.append(entry)
                max_prob = max(max_prob, entry["probability"])
        return max_prob

    def _severity(self, p: float) -> str:
        if p >= SEVERITY_THRESHOLDS["high"]:     return "high"
        if p >= SEVERITY_THRESHOLDS["moderate"]: return "moderate"
        return "low"


def _empty_specialized_output() -> Dict:
    return {
        "tuberculosis_probability": 0.0,
        "tuberculosis_display_score": 0.0,
        "tuberculosis_source": "none",
        "fracture_detections": [],
        "fracture_risk_score": 0.0,
        "fracture_source": "none",
        "alerts": [],
        "tuberculose_score": 0.0,
        "fracture_score": 0.0,
        "derived_source": "none",
        "tuberculosis_mode": "heuristic",
        "fracture_mode": "none",
        "decision_support": {
            "tb_suspected": False,
            "fracture_suspected": False,
            "tuberculosis_mode": "heuristic",
            "fracture_mode": "none",
            "recommended_action": "Analyse non disponible.",
        },
    }


# Alias pour train.py
class ChestXRayModel(torch.nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        import torch.nn as nn
        from torchvision.models import densenet121, DenseNet121_Weights
        base = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        base.classifier = nn.Sequential(
            nn.Linear(base.classifier.in_features, 512),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self.backbone = base

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))
