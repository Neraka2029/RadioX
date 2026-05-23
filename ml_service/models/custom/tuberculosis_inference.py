"""
Wrapper d'inférence sécurisé — modèle Tuberculose (.pt / .pth).

Entrée compatible pipeline NIH :
  - shape attendue : (1, 1, 224, 224)
  - même normalisation torchxrayvision (xrv.datasets.normalize) que PreprocessingPipeline

En cas d'erreur (chargement, shape, forward) : retourne None → fallback heuristique dans inference.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

REGISTRY_KEY = "tuberculosis"
EXPECTED_SHAPE: Tuple[int, ...] = (1, 1, 224, 224)


class TuberculosisInferenceWrapper(nn.Module):
    """Encapsule le checkpoint TB et les métadonnées d'activation de sortie."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        apply_sigmoid: bool = False,
        source_file: str = "",
    ):
        super().__init__()
        self.backbone = backbone
        self.apply_sigmoid = apply_sigmoid
        self.source_file = source_file

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_tuberculosis_checkpoint(path: Path) -> TuberculosisInferenceWrapper:
    """Charge un checkpoint TB compatible RadioX (ChestXRayModel binaire ou nn.Module)."""
    logger.info("Loading Tuberculosis checkpoint: %s", path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    apply_sigmoid = False
    backbone: nn.Module

    if isinstance(checkpoint, TuberculosisInferenceWrapper):
        logger.info("Checkpoint is already a TuberculosisInferenceWrapper")
        return checkpoint

    if isinstance(checkpoint, nn.Module):
        backbone = checkpoint
        apply_sigmoid = _module_needs_sigmoid(backbone)
        logger.info(
            "Loaded nn.Module (%s), apply_sigmoid=%s",
            backbone.__class__.__name__,
            apply_sigmoid,
        )
    elif isinstance(checkpoint, dict):
        meta = checkpoint.get("meta") or checkpoint.get("metadata") or {}
        if meta.get("apply_sigmoid") is not None:
            apply_sigmoid = bool(meta["apply_sigmoid"])
        elif meta.get("output_activation") == "logits":
            apply_sigmoid = True
        elif meta.get("output_activation") in ("sigmoid", "probability"):
            apply_sigmoid = False

        if "model" in checkpoint and isinstance(checkpoint["model"], nn.Module):
            backbone = checkpoint["model"]
            if not meta:
                apply_sigmoid = _module_needs_sigmoid(backbone)
        else:
            state = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
            if state is None and all(isinstance(k, str) for k in checkpoint.keys()):
                state = checkpoint
            if state is None:
                raise ValueError(f"No loadable weights in {path.name}")
            num_classes = int(checkpoint.get("num_classes", 1))
            backbone = _build_tb_classifier(num_classes)
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            logger.info(
                "state_dict loaded (num_classes=%d, missing=%d, unexpected=%d)",
                num_classes,
                len(missing),
                len(unexpected),
            )
            if not meta:
                apply_sigmoid = False
    else:
        raise ValueError(f"Unsupported Tuberculosis checkpoint type: {type(checkpoint)}")

    wrapper = TuberculosisInferenceWrapper(
        backbone,
        apply_sigmoid=apply_sigmoid,
        source_file=path.name,
    )
    wrapper.eval()
    logger.info(
        "Tuberculosis model ready (file=%s, apply_sigmoid=%s, expected_input=%s)",
        path.name,
        wrapper.apply_sigmoid,
        EXPECTED_SHAPE,
    )
    return wrapper


def prepare_tb_input(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Valide la sortie PreprocessingPipeline NIH : (1, 1, 224, 224).
    Ne modifie pas la normalisation (déjà appliquée par xrv.datasets.normalize).
    """
    x = tensor.to(device)
    original_shape = tuple(x.shape)
    logger.info("TB input received: shape=%s dtype=%s device=%s", original_shape, x.dtype, x.device)

    if x.dim() == 2 and x.shape == (224, 224):
        x = x.unsqueeze(0).unsqueeze(0)
        logger.info("TB input reshaped from (224,224) -> (1,1,224,224)")
    elif x.dim() == 3:
        if x.shape[0] == 1:
            x = x.unsqueeze(0)
            logger.info("TB input reshaped from %s -> (1,1,224,224)", original_shape)
        elif x.shape[0] in (1, 3) and x.shape[1] == 224:
            if x.shape[0] == 3:
                x = x.mean(dim=0, keepdim=True).unsqueeze(0)
                logger.warning("TB input was 3-channel; converted to grayscale (1,1,224,224)")
            else:
                x = x.unsqueeze(0)
    elif x.dim() == 4 and x.shape[0] != 1:
        x = x[:1]
        logger.warning("TB input batch trimmed to size 1")

    if x.dim() != 4 or x.shape[2:] != (224, 224):
        raise ValueError(
            f"TB spatial size incompatible: got {tuple(x.shape)}, expected (*, *, 224, 224)"
        )

    if x.shape[1] not in (1, 3):
        raise ValueError(f"TB channel count must be 1 or 3, got {x.shape[1]}")

    if x.shape[1] == 3:
        x = x.mean(dim=1, keepdim=True)
        logger.info("TB NIH pipeline expects 1ch; averaged 3ch input to (1,1,224,224)")

    if tuple(x.shape) != EXPECTED_SHAPE:
        raise ValueError(f"TB NIH input must be {EXPECTED_SHAPE}, got {tuple(x.shape)}")

    logger.info(
        "TB NIH input validated: shape=%s min=%.4f max=%.4f mean=%.4f",
        tuple(x.shape),
        float(x.min()),
        float(x.max()),
        float(x.mean()),
    )
    return x


def detect_model_in_channels(backbone: nn.Module) -> int:
    """Premier Conv2d du backbone (1 = torchxrayvision / NIH, 3 = DenseNet ImageNet)."""
    root = backbone.backbone if hasattr(backbone, "backbone") else backbone
    for module in root.modules():
        if isinstance(module, nn.Conv2d):
            logger.info("TB model first Conv2d in_channels=%d", module.in_channels)
            return module.in_channels
    logger.warning("TB model: no Conv2d found, assuming 1 input channel")
    return 1


def adapt_tb_input_for_model(x: torch.Tensor, model_in_channels: int) -> torch.Tensor:
    """Adapte (1,1,224,224) NIH vers le nombre de canaux attendu par le .pt."""
    if x.shape[1] == model_in_channels:
        return x
    if x.shape[1] == 1 and model_in_channels == 3:
        adapted = x.repeat(1, 3, 1, 1)
        logger.info(
            "TB adapter: (1,1,224,224) -> (1,3,224,224) (grayscale repeat for 3ch model)"
        )
        return adapted
    if x.shape[1] == 3 and model_in_channels == 1:
        adapted = x.mean(dim=1, keepdim=True)
        logger.info("TB adapter: (1,3,224,224) -> (1,1,224,224) (mean for 1ch model)")
        return adapted
    raise ValueError(
        f"Cannot adapt TB input channels {x.shape[1]} to model in_channels={model_in_channels}"
    )


def extract_tb_probability(raw: torch.Tensor, apply_sigmoid: bool) -> float:
    """Convertit la sortie réseau en probabilité [0, 1] (sigmoid si logits)."""
    out = raw
    if isinstance(out, (list, tuple)):
        out = out[0]
    if out.dim() > 1:
        out = out.squeeze()

    if out.numel() > 1:
        values = out.flatten().float()
        logger.info("TB multi-output (%d values), using max activation", values.numel())
        out = values.max()
    else:
        out = out.float().squeeze()

    val = float(out.item())
    if apply_sigmoid:
        prob = float(torch.sigmoid(out).item())
        logger.info("TB logits=%.4f -> sigmoid prob=%.4f", val, prob)
    elif 0.0 <= val <= 1.0:
        prob = val
        logger.info("TB output treated as probability: %.4f", prob)
    else:
        prob = float(torch.sigmoid(out).item())
        logger.info("TB output out of [0,1] (%.4f), applied sigmoid -> %.4f", val, prob)

    return round(max(0.0, min(1.0, prob)), 4)


@torch.no_grad()
def run_tuberculosis_inference(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> Optional[float]:
    """
    Inférence TB sécurisée. Retourne None en cas d'erreur (déclenche fallback heuristique).
    """
    try:
        if not isinstance(model, TuberculosisInferenceWrapper):
            wrapper = TuberculosisInferenceWrapper(
                model,
                apply_sigmoid=_module_needs_sigmoid(model),
            )
        else:
            wrapper = model
        wrapper.eval()

        x = prepare_tb_input(tensor, device)
        in_ch = detect_model_in_channels(wrapper.backbone)
        x_model = adapt_tb_input_for_model(x, in_ch)
        logger.info("TB model forward input shape=%s", tuple(x_model.shape))
        raw = wrapper(x_model)
        prob = extract_tb_probability(raw, wrapper.apply_sigmoid)
        logger.info(
            "TB inference OK (file=%s, prob=%.4f)",
            getattr(wrapper, "source_file", "?"),
            prob,
        )
        return prob
    except Exception as e:
        logger.error("TB inference failed, heuristic fallback will be used: %s", e, exc_info=True)
        return None


def _module_needs_sigmoid(module: nn.Module) -> bool:
    """ChestXRayModel applique déjà sigmoid — éviter double sigmoid."""
    name = module.__class__.__name__
    if name == "ChestXRayModel":
        return False
    if isinstance(module, TuberculosisInferenceWrapper):
        return module.apply_sigmoid
    return True


def _build_tb_classifier(num_classes: int) -> nn.Module:
    from pipelines.inference import ChestXRayModel

    return ChestXRayModel(num_classes=num_classes)
