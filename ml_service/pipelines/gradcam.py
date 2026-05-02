"""
Grad-CAM Pipeline - Compatible torchxrayvision
Clone les outputs pour éviter l'erreur inplace view.
"""
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class GradCAMPipeline:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.target_layer = self._find_target_layer()
        # Pas de logs pour le nettoyage

    def _find_target_layer(self):
        self._layer_name = ""
        backbone = getattr(self.model, "features", None) or self.model
        last_conv, last_name = None, ""
        for name, module in backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv, last_name = module, name
        if last_conv:
            self._layer_name = last_name
        return last_conv

    def generate(self, tensor: torch.Tensor, predictions: List[Dict],
                 target_class_idx: Optional[int] = None) -> Optional[str]:
        if self.target_layer is None:
            return None
        try:
            # Classe cible = probabilité la plus haute (hors Normal)
            if target_class_idx is None:
                non_normal = [(i, p) for i, p in enumerate(predictions) if p["pathology"] != "Normal"]
                if non_normal:
                    target_class_idx = max(non_normal, key=lambda x: x[1]["probability"])[0]
                else:
                    target_class_idx = 0

            activations, gradients = {}, {}

            def fwd_hook(module, input, output):
                activations["v"] = output.clone()  # ← clone obligatoire

            def bwd_hook(module, grad_in, grad_out):
                gradients["v"] = grad_out[0].clone()  # ← clone obligatoire

            h_fwd = self.target_layer.register_forward_hook(fwd_hook)
            h_bwd = self.target_layer.register_full_backward_hook(bwd_hook)

            try:
                self.model.train()
                out = self.model(tensor.to(self.device).float())
                if out.dim() == 2:
                    score = out[0, min(target_class_idx, out.shape[1]-1)]
                elif out.dim() == 1:
                    score = out[min(target_class_idx, out.shape[0]-1)]
                else:
                    score = out.mean()
                self.model.zero_grad()
                score.backward()             
            finally:
                self.model.eval()
                h_fwd.remove()
                h_bwd.remove()

            if "v" not in gradients or "v" not in activations:
                logger.error(f"Grad-CAM missing gradients or activations")
                return None

            # Calcul Grad-CAM
            grads = gradients["v"]    # [1, C, H, W]
            acts  = activations["v"]  # [1, C, H, W]
            
            # Poids = moyenne globale des gradients (avec valeur absolue pour éviter l'annulation)
            weights = torch.abs(grads).mean(dim=(2, 3), keepdim=True)
            
            # Produit poids * activations
            weighted_acts = weights * acts
            
            # Somme sur les canaux + ReLU
            cam = torch.relu(weighted_acts.sum(dim=1, keepdim=True))

            cam_np = cam.squeeze().detach().cpu().numpy()
            if cam_np.ndim == 0:
                logger.error(f"Grad-CAM CAM is scalar after squeeze")
                return None

            # Normaliser [0, 1]
            cmin, cmax = cam_np.min(), cam_np.max()
            cam_np = (cam_np - cmin) / (cmax - cmin + 1e-8)

            # Coloriser en jet et encoder PNG (carte seule)
            heatmap = self._colorize(cam_np, size=(224, 224))
            buf = BytesIO()
            heatmap.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")
            return None

    def _colorize(self, cam: np.ndarray, size: tuple) -> Image.Image:
        cam_r = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
        ) / 255.0
        h, w = cam_r.shape
        
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Colormap "hot" (rouge/orange) au lieu de "jet" pour éviter le bleu
        # Seules les zones chaudes sont colorées
        intensity = cam_r  # 0 à 1
        
        # Rouge pour les zones chaudes, orange pour moyennes, transparent pour froides
        r = np.ones_like(intensity)  # Toujours rouge
        g = np.where(intensity > 0.4, 0.6, np.where(intensity > 0.2, 0.3, 0))  # Orange/rouge plus visible
        b = np.zeros_like(intensity)  # Pas de bleu
        
        rgba[:,:,0] = (r * 255).astype(np.uint8)
        rgba[:,:,1] = (g * 255).astype(np.uint8)
        rgba[:,:,2] = (b * 255).astype(np.uint8)
        
        # Transparence : les zones faibles sont transparentes
        # Seuils ajustés pour ne montrer que les zones importantes
        threshold = 0.01  # Seuil très bas pour les valeurs CAM faibles
        alpha = np.where(cam_r > threshold, ((cam_r - threshold) / (1 - threshold) * 255).astype(np.uint8), 0)
        rgba[:,:,3] = alpha
        
        print(f"Alpha non-zero pixels: {np.sum(alpha > 0)} / {alpha.size}")
        print(f"Alpha min/max: {alpha.min()} / {alpha.max()}")
        print(f"===================")
        
        return Image.fromarray(rgba, 'RGBA')
