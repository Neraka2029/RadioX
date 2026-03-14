#!/usr/bin/env python3
"""
Convertit le modèle NIH téléchargé vers notre format.
"""

import os
import torch
import torchxrayvision as xrv

def convert_nih_model():
    """Convertit le modèle NIH vers notre architecture."""
    
    print("🔄 Conversion du modèle NIH...")
    
    # Charger le modèle NIH
    nih_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    
    # Créer notre modèle
    from pipelines.inference import ChestXRayModel, PATHOLOGY_LABELS
    
    our_model = ChestXRayModel(num_classes=len(PATHOLOGY_LABELS))
    
    # Mapping des pathologies
    nih_pathologies = nih_model.pathologies
    print(f"Pathologies NIH ({len(nih_pathologies)}): {nih_pathologies}")
    print(f"Pathologies RadioX ({len(PATHOLOGY_LABELS)}): {PATHOLOGY_LABELS}")
    
    # Créer le mapping
    our_to_nih = {}
    for i, our_path in enumerate(PATHOLOGY_LABELS):
        for j, nih_path in enumerate(nih_pathologies):
            if (our_path.lower() == nih_path.lower() or
                (our_path == 'Normal' and nih_path == 'No Finding') or
                (our_path == 'Oedeme' and nih_path == 'Edema') or
                (our_path == 'Epanchement pleural' and nih_path == 'Effusion') or
                (our_path == 'Epaississement pleural' and nih_path == 'Pleural_Thickening')):
                our_to_nih[i] = j
                print(f"  {our_path} -> {nih_path} ({i} -> {j})")
                break
    
    # Sauvegarder les poids du modèle NIH dans notre format
    nih_state = nih_model.state_dict()
    our_state = our_model.state_dict()
    
    # Copier les poids du backbone
    for key, value in nih_state.items():
        if key in our_state:
            our_state[key] = value
    
    # Mapper le classifier
    if 'classifier.weight' in nih_state:
        nih_weight = nih_state['classifier.weight']
        nih_bias = nih_state['classifier.bias']
        
        new_weight = torch.zeros(len(PATHOLOGY_LABELS), nih_weight.shape[1])
        new_bias = torch.zeros(len(PATHOLOGY_LABELS))
        
        for our_idx, nih_idx in our_to_nih.items():
            if nih_idx < len(nih_weight):
                new_weight[our_idx] = nih_weight[nih_idx]
                new_bias[our_idx] = nih_bias[nih_idx]
        
        our_state['backbone.classifier.1.weight'] = new_weight
        our_state['backbone.classifier.1.bias'] = new_bias
    
    # Sauvegarder
    model_path = "models/nih_densenet121_converted.pt"
    os.makedirs("models", exist_ok=True)
    torch.save(our_state, model_path)
    
    print(f"✅ Modèle converti sauvegardé: {model_path}")
    print(f"📊 Taille: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
    
    return model_path

if __name__ == "__main__":
    convert_nih_model()
