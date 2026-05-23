"""
ModelManager léger — chargement lazy des modèles spécialisés (.pt/.pth) avec fallback.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Set

import torch
import torch.nn as nn

from models.custom.registry import CUSTOM_MODEL_REGISTRY, get_custom_models_dir

logger = logging.getLogger(__name__)


class ModelManager:
    """Gestion plug-and-play des modèles dans models/custom/."""

    def __init__(self, device: torch.device):
        self.device = device
        self._models: Dict[str, nn.Module] = {}
        self._checkpoint_paths: Dict[str, Path] = {}
        self._used_this_predict: Set[str] = set()
        if not torch.cuda.is_available() and device.type == "cpu":
            logger.info("GPU unavailable, using CPU")

    def load_models(self) -> None:
        """Découvre les checkpoints sans les charger en mémoire (lazy)."""
        custom_dir = get_custom_models_dir()
        custom_dir.mkdir(parents=True, exist_ok=True)
        for key, spec in CUSTOM_MODEL_REGISTRY.items():
            path = self._resolve_checkpoint(custom_dir, spec["candidate_files"])
            if path is not None:
                self._checkpoint_paths[key] = path
                logger.info(
                    "%s model file found: %s (lazy load on first use)",
                    spec["radiox_label"],
                    path.name,
                )
            else:
                logger.info(
                    "No checkpoint for %s in %s — heuristic fallback will be used",
                    spec["radiox_label"],
                    custom_dir,
                )

    def get_model(self, name: str) -> Optional[nn.Module]:
        """Charge un modèle spécialisé à la demande."""
        if name in self._models:
            return self._models[name]

        path = self._checkpoint_paths.get(name)
        if path is None:
            return None

        try:
            if name == "tuberculosis":
                from models.custom.tuberculosis_inference import load_tuberculosis_checkpoint

                module = load_tuberculosis_checkpoint(path)
            else:
                module = self._load_checkpoint(path, name)
            module.to(self.device)
            module.eval()
            self._models[name] = module
            label = CUSTOM_MODEL_REGISTRY[name]["radiox_label"]
            logger.info("%s model loaded from %s", label, path.name)
            return module
        except Exception as e:
            logger.warning(
                "Failed to load specialized model '%s' from %s: %s",
                name,
                path,
                e,
            )
            return None

    def unload_unused_models(self, keep: Optional[Set[str]] = None) -> None:
        """Libère la mémoire GPU des modèles non utilisés lors du dernier predict."""
        keep_set = keep if keep is not None else self._used_this_predict
        to_remove = [k for k in list(self._models.keys()) if k not in keep_set]
        for key in to_remove:
            spec = CUSTOM_MODEL_REGISTRY.get(key, {})
            label = spec.get("radiox_label", key)
            del self._models[key]
            logger.debug("Unloaded specialized model: %s", label)
        if to_remove and self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._used_this_predict = set()

    def begin_predict(self) -> None:
        self._used_this_predict = set()

    @torch.no_grad()
    def predict(self, name: str, tensor: torch.Tensor) -> Optional[float]:
        """
        Inférence modèle spécialisé. Retourne une probabilité [0, 1] ou None si indisponible.
        """
        model = self.get_model(name)
        if model is None:
            return None

        self._used_this_predict.add(name)
        if name == "tuberculosis":
            from models.custom.tuberculosis_inference import run_tuberculosis_inference

            return run_tuberculosis_inference(model, tensor, self.device)

        x = tensor.to(self.device)
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.dim() > 1:
            out = out.squeeze()
        if out.numel() > 1:
            prob = float(torch.sigmoid(out).max().item())
        else:
            prob = float(torch.sigmoid(out).item())
        return round(max(0.0, min(1.0, prob)), 4)

    def has_checkpoint(self, name: str) -> bool:
        return name in self._checkpoint_paths

    @staticmethod
    def _resolve_checkpoint(directory: Path, candidates: tuple) -> Optional[Path]:
        for filename in candidates:
            path = directory / filename
            if path.is_file():
                return path
        return None

    def _load_checkpoint(self, path: Path, registry_key: str) -> nn.Module:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, nn.Module):
            return checkpoint

        if isinstance(checkpoint, dict):
            if "model" in checkpoint and isinstance(checkpoint["model"], nn.Module):
                return checkpoint["model"]
            state = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
            if state is not None:
                num_classes = int(checkpoint.get("num_classes", 1))
                model = self._build_binary_classifier(num_classes)
                model.load_state_dict(state, strict=False)
                return model

        if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
            model = self._build_binary_classifier(1)
            model.load_state_dict(checkpoint, strict=False)
            return model

        raise ValueError(
            f"Unsupported checkpoint format in {path.name}. "
            "Expected nn.Module, state_dict, or dict with 'state_dict'."
        )

    @staticmethod
    def _build_binary_classifier(num_classes: int = 1) -> nn.Module:
        from pipelines.inference import ChestXRayModel

        return ChestXRayModel(num_classes=num_classes)
