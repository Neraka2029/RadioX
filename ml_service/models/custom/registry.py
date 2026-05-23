"""
Registre des modèles spécialisés — ajouter une entrée ici pour une nouvelle pathologie.
"""
from pathlib import Path

_CUSTOM_DIR = Path(__file__).resolve().parent


def get_custom_models_dir() -> Path:
    return _CUSTOM_DIR


# Clé interne -> métadonnées (fichiers candidats, labels API, seuil XRV)
CUSTOM_MODEL_REGISTRY = {
    "tuberculosis": {
        "radiox_label": "Tuberculose",
        "xrv_key": "Tuberculosis",
        "candidate_files": (
            "tuberculosis.pt",
            "tuberculosis.pth",
            "tb.pt",
            "tb.pth",
        ),
    },
    "fracture": {
        "radiox_label": "Fracture costale",
        "xrv_key": "Fracture",
        "candidate_files": (
            "fracture.pt",
            "fracture.pth",
            "fracture_costale.pt",
            "fracture_costale.pth",
        ),
    },
}
