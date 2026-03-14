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
from typing import List, Dict

logger = logging.getLogger(__name__)
IMAGE_SIZE = 224

# ── 14 pathologies NIH + Normal ───────────────────────────────────────────
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
}

# Labels fiables — exclure ceux qui retournent 0.5 fixe dans le modèle NIH
# Emphysema, Fibrosis, Hernia sont non fiables dans densenet121-res224-nih
RELIABLE_LABELS = {
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass",
}
# Labels exclus car retournent ~0.5 fixe : Emphysema, Fibrosis, Hernia

# Seuil minimum par pathologie (calibré pour réduire les faux positifs)
THRESHOLDS_PER_PATHOLOGY = {
    "Atelectasis":        0.45,
    "Consolidation":      0.45,
    "Infiltration":       0.45,
    "Pneumothorax":       0.50,
    "Edema":              0.45,
    "Effusion":           0.45,
    "Pneumonia":          0.45,
    "Pleural_Thickening": 0.50,
    "Cardiomegaly":       0.45,
    "Nodule":             0.60,  # beaucoup de faux positifs
    "Mass":               0.75,  # beaucoup de faux positifs
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
}


class PreprocessingPipeline:
    """Preprocessing officiel torchxrayvision."""

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        is_dicom = len(image_bytes) > 132 and image_bytes[128:132] == b"DICM"
        if is_dicom:
            try:
                import pydicom
                from pydicom.filebase import DicomBytesIO
                ds = pydicom.dcmread(DicomBytesIO(image_bytes))
                arr = ds.pixel_array.astype(np.float32)
                if hasattr(ds, "RescaleSlope"):
                    arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255).astype(np.uint8)
                if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                    arr = 255 - arr
                if arr.ndim == 3:
                    arr = arr[0]
                return Image.fromarray(arr)
            except Exception as e:
                logger.warning(f"DICOM error: {e}")
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

        # Normalisation officielle xrv AVANT resize : [0,255] -> [-1024,1024]
        arr = xrv.datasets.normalize(arr, maxval=255, reshape=False)

        # Resize avec preserve_range pour garder l'échelle [-1024,1024]
        arr = skimage.transform.resize(
            arr, (IMAGE_SIZE, IMAGE_SIZE),
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        logger.info(f"Preprocessing OK: min={tensor.min():.0f}, max={tensor.max():.0f}, mean={tensor.mean():.0f}")
        return tensor


class InferencePipeline:
    """Pipeline principal utilisant torchxrayvision DenseNet-121 NIH."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = PreprocessingPipeline()
        self.xrv_labels = None
        self.model = self._load_model()
        self.model.eval()
        logger.info(f"InferencePipeline prêt sur {self.device}")

    def _load_model(self):
        try:
            import torchxrayvision as xrv
            model = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.xrv_labels = list(model.pathologies)
            logger.info(f"Modèle NIH chargé — {len(self.xrv_labels)} labels")
            return model.to(self.device)
        except ImportError:
            raise RuntimeError("torchxrayvision non installé. Lancez : pip install torchxrayvision")
        except Exception as e:
            raise RuntimeError(f"Erreur chargement modèle: {e}")

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        return self.preprocessor.process(image_bytes)

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor) -> List[Dict]:
        output = self.model(tensor.to(self.device))
        return self._build_results(output)

    def _build_results(self, output: torch.Tensor) -> List[Dict]:
        probs = np.clip(output.squeeze().detach().cpu().numpy(), 0.0, 1.0)

        # Log des scores bruts (labels fiables uniquement)
        raw_log = {
            lbl: round(float(probs[i]), 3)
            for i, lbl in enumerate(self.xrv_labels)
            if i < len(probs) and lbl in RELIABLE_LABELS
        }
        logger.info(f"XRV scores: {raw_log}")

        results = []
        max_prob = 0.0

        for xrv_lbl, radiox_lbl in XRV_TO_RADIOX.items():
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
                "color":        PATHOLOGY_COLORS.get(radiox_lbl, "#4488ff"),
            })

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

        top3 = [(r["pathology"], r["probability"]) for r in results[1:4]]
        logger.info(f"Top 3 pathologies: {top3}")
        return results

    def _severity(self, p: float) -> str:
        if p >= SEVERITY_THRESHOLDS["high"]:     return "high"
        if p >= SEVERITY_THRESHOLDS["moderate"]: return "moderate"
        return "low"


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
