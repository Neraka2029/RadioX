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
        if self.target_layer:
            logger.info(f"Grad-CAM target layer: {self._layer_name}")

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
                self.model.eval()
            finally:
                h_fwd.remove()
                h_bwd.remove()

            if "v" not in gradients or "v" not in activations:
                print(f"GRADCAM ERROR: Missing gradients or activations")
                print(f"Gradients keys: {list(gradients.keys())}")
                print(f"Activations keys: {list(activations.keys())}")
                return None

            # Logs pour debug des gradients et activations
            print(f"=== GRADCAM HOOKS DEBUG ===")
            print(f"Gradients shape: {gradients['v'].shape if 'v' in gradients else 'None'}")
            print(f"Activations shape: {activations['v'].shape if 'v' in activations else 'None'}")
            print(f"Gradients min/max: {gradients['v'].min():.6f} / {gradients['v'].max():.6f}" if 'v' in gradients else "None")
            print(f"Activations min/max: {activations['v'].min():.6f} / {activations['v'].max():.6f}" if 'v' in activations else "None")
            print(f"Target class index: {target_class_idx}")
            print(f"========================")

            # Calcul Grad-CAM
            grads = gradients["v"]    # [1, C, H, W]
            acts  = activations["v"]  # [1, C, H, W]
            
            print(f"Gradients shape: {grads.shape}, min/max: {grads.min():.6f} / {grads.max():.6f}")
            print(f"Activations shape: {acts.shape}, min/max: {acts.min():.6f} / {acts.max():.6f}")
            
            # Poids = moyenne globale des gradients (avec valeur absolue pour éviter l'annulation)
            weights = torch.abs(grads).mean(dim=(2, 3), keepdim=True)
            print(f"Weights shape: {weights.shape}, min/max: {weights.min():.6f} / {weights.max():.6f}")
            
            # Produit poids * activations
            weighted_acts = weights * acts
            print(f"Weighted acts min/max: {weighted_acts.min():.6f} / {weighted_acts.max():.6f}")
            
            # Somme sur les canaux + ReLU
            cam = torch.relu(weighted_acts.sum(dim=1, keepdim=True))
            print(f"CAM after sum+relu min/max: {cam.min():.6f} / {cam.max():.6f}")

            cam_np = cam.squeeze().detach().cpu().numpy()
            if cam_np.ndim == 0:
                print(f"CAM is scalar after squeeze")
                return None

            print(f"CAM after squeeze: {cam_np.shape}")
            print(f"CAM min/max after squeeze: {cam_np.min():.6f} / {cam_np.max():.6f}")

            # Normaliser [0, 1]
            cmin, cmax = cam_np.min(), cam_np.max()
            cam_np = (cam_np - cmin) / (cmax - cmin + 1e-8)

            # Coloriser en jet et encoder PNG
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
        
        # Logs pour debug
        print(f"=== GRADCAM DEBUG ===")
        print(f"CAM shape: {cam.shape}")
        print(f"CAM min/max: {cam.min():.3f} / {cam.max():.3f}")
        print(f"CAM_r min/max: {cam_r.min():.3f} / {cam_r.max():.3f}")
        print(f"Pixels > 0.2: {np.sum(cam_r > 0.2)} / {cam_r.size}")
        print(f"Pixels > 0.5: {np.sum(cam_r > 0.5)} / {cam_r.size}")
        
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
