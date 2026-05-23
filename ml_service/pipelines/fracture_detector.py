"""
Fracture costale — deux modes exclusifs :

Mode YOLO (si fracture_yolov8.pt) :
  → bounding boxes réelles + fracture_risk_score

Mode fallback (sans modèle) :
  → fracture_risk_score via NIH uniquement (pipelines.heuristic_predictors)
  → fracture_detections TOUJOURS [] — pas de fausse détection
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

CUSTOM_DIR = Path(__file__).resolve().parent.parent / "models" / "custom"
CHECKPOINT_FILES = (
    "fracture_yolov8.pt",
    "fracture_yolo.pt",
    "rib_fracture_yolov8.pt",
)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


class FractureDetectorService:
    """Singleton YOLOv8 — pas de rechargement par requête."""

    def __init__(self, device: torch.device):
        self.device = device
        self._model = None
        self._checkpoint_path: Optional[Path] = None
        self._load_error: Optional[str] = None
        self._ultralytics_available = False

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
                "Fracture YOLO weights not found in %s — NIH risk indicator mode (no bboxes)",
                CUSTOM_DIR,
            )
            return False

        try:
            from ultralytics import YOLO

            self._ultralytics_available = True
            self._model = YOLO(str(path))
            self._checkpoint_path = path
            logger.info("Fracture YOLOv8 loaded from %s", path.name)
            return True
        except ImportError:
            self._load_error = "ultralytics not installed"
            logger.warning("ultralytics missing — pip install ultralytics for fracture detection")
            return False
        except Exception as e:
            self._load_error = str(e)
            logger.error("Fracture YOLO load failed: %s", e)
            return False

    @staticmethod
    def _resolve_checkpoint() -> Optional[Path]:
        CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
        for name in CHECKPOINT_FILES:
            p = CUSTOM_DIR / name
            if p.is_file():
                return p
        return None

    def predict(
        self,
        image_rgb: np.ndarray,
        img_size: Tuple[int, int] = (224, 224),
    ) -> Optional[Dict[str, Any]]:
        """
        image_rgb : H×W uint8 ou float grayscale/RGB (sera converti en 3 canaux si besoin).
        """
        if not self.load():
            return None

        try:
            arr = np.asarray(image_rgb)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)

            device_str = "0" if self.device.type == "cuda" else "cpu"
            results = self._model.predict(
                source=arr,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
                device=device_str,
            )

            detections: List[Dict[str, Any]] = []
            h, w = arr.shape[0], arr.shape[1]

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = xyxy
                    detections.append({
                        "bbox": [
                            round(x1 / w, 4),
                            round(y1 / h, 4),
                            round(x2 / w, 4),
                            round(y2 / h, 4),
                        ],
                        "bbox_pixels": [round(x1), round(y1), round(x2), round(y2)],
                        "confidence": round(conf, 4),
                        "class": "rib_fracture",
                    })

            from pipelines.heuristic_predictors import normalize_probability

            risk = normalize_probability(
                max((d["confidence"] for d in detections), default=0.0)
            )
            logger.info(
                "Fracture YOLO: %d detection(s), fracture_risk_score=%.4f",
                len(detections),
                risk,
            )
            return {
                "fracture_detections": detections,
                "fracture_risk_score": risk,
            }
        except Exception as e:
            logger.error("Fracture YOLO inference failed: %s", e, exc_info=True)
            return None
