"""
Modèles spécialisés plug-and-play (fichiers .pt / .pth).

Placez vos checkpoints ici. Noms de fichiers reconnus (premier trouvé utilisé) :
  - Tuberculose : tuberculosis.pt, tuberculosis.pth, tb.pt, tb.pth
  - Fracture    : fracture.pt, fracture.pth, fracture_costale.pt, fracture_costale.pth
"""

from .registry import CUSTOM_MODEL_REGISTRY, get_custom_models_dir

__all__ = [
    "CUSTOM_MODEL_REGISTRY",
    "get_custom_models_dir",
    "tuberculosis_inference",
]
