#!/usr/bin/env python3
"""
Télécharge et configure un modèle pré-entraîné NIH ChestX-ray14.
Supporte plusieurs sources : torchxrayvision, modèles Hugging Face, ou modèles locaux.
"""

import os
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def download_torchxrayvision_model():
    """Télécharge un modèle pré-entraîné depuis torchxrayvision."""
    try:
        import torchxrayvision as xrv
        logger.info("Téléchargement du modèle DenseNet-121 depuis torchxrayvision...")
        
        # Modèle DenseNet-121 pré-entraîné sur NIH ChestX-ray14
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        
        # Sauvegarder le modèle
        model_path = "models/nih_densenet121.pth"
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        
        logger.info(f"Modèle sauvegardé dans {model_path}")
        return model_path
    except ImportError:
        logger.warning("torchxrayvision n'est pas installé. Installation...")
        os.system("pip install torchxrayvision")
        return download_torchxrayvision_model()
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement: {e}")
        return None

def download_huggingface_model():
    """Télécharge un modèle depuis Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download
        
        logger.info("Téléchargement du modèle depuis Hugging Face...")
        
        # Modèle pré-entraîné sur NIH ChestX-ray14
        model_path = hf_hub_download(
            repo_id="medicalai/CheXNet",
            filename="model.pth",
            cache_dir="models/"
        )
        
        logger.info(f"Modèle téléchargé dans {model_path}")
        return model_path
    except ImportError:
        logger.warning("huggingface_hub n'est pas installé. Installation...")
        os.system("pip install huggingface_hub")
        return download_huggingface_model()
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement: {e}")
        return None

def create_nih_compatible_model():
    """Crée un modèle compatible avec les poids NIH."""
    try:
        import torchxrayvision as xrv
        
        # Télécharger le modèle
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        
        # Adapter pour notre architecture
        from pipelines.inference import ChestXRayModel, PATHOLOGY_LABELS
        
        our_model = ChestXRayModel(num_classes=len(PATHOLOGY_LABELS))
        
        # Mapper les poids du modèle NIH vers notre modèle
        nih_to_our_mapping = {
            # Pathologies NIH vers nos pathologies
            'Atelectasis': 'Atelectasie',
            'Consolidation': 'Consolidation', 
            'Infiltration': 'Infiltration',
            'Pneumothorax': 'Pneumothorax',
            'Edema': 'Oedeme',
            'Emphysema': 'Emphyseme',
            'Fibrosis': 'Fibrose',
            'Effusion': 'Epanchement pleural',
            'Pneumonia': 'Pneumonie',
            'Pleural_Thickening': 'Epaississement pleural',
            'Cardiomegaly': 'Cardiomegalie',
            'Nodule': 'Nodule',
            'Mass': 'Masse',
            'Hernia': 'Hernie',
            'No Finding': 'Normal'
        }
        
        # Charger les poids du modèle NIH
        nih_pathologies = list(model.pathologies)
        nih_weights = model.state_dict()
        
        # Créer un mapping des indices
        our_to_nih_indices = {}
        for i, our_path in enumerate(PATHOLOGY_LABELS):
            for j, nih_path in enumerate(nih_pathologies):
                if our_path.lower() == nih_path.lower() or \
                   (our_path == 'Normal' and nih_path == 'No Finding') or \
                   (our_path == 'Oedeme' and nih_path == 'Edema') or \
                   (our_path == 'Epanchement pleural' and nih_path == 'Effusion') or \
                   (our_path == 'Epaississement pleural' and nih_path == 'Pleural_Thickening'):
                    our_to_nih_indices[i] = j
                    break
        
        # Mapper les poids du classifier
        if 'backbone.classifier.weight' in nih_weights:
            classifier_weight = nih_weights['backbone.classifier.weight']
            classifier_bias = nih_weights['backbone.classifier.bias']
            
            # Créer nouveau poids pour notre classifier
            new_weight = torch.zeros(len(PATHOLOGY_LABELS), classifier_weight.shape[1])
            new_bias = torch.zeros(len(PATHOLOGY_LABELS))
            
            for our_idx, nih_idx in our_to_nih_indices.items():
                new_weight[our_idx] = classifier_weight[nih_idx]
                new_bias[our_idx] = classifier_bias[nih_idx]
            
            # Mettre à jour les poids de notre modèle
            our_state_dict = our_model.state_dict()
            our_state_dict['backbone.classifier.1.weight'] = new_weight
            our_state_dict['backbone.classifier.1.bias'] = new_bias
            
            # Copier les autres poids du backbone
            for key, value in nih_weights.items():
                if key.startswith('backbone.') and 'classifier' not in key:
                    our_state_dict[key] = value
            
            our_model.load_state_dict(our_state_dict)
            
            # Sauvegarder
            model_path = "models/nih_densenet121_mapped.pth"
            os.makedirs("models", exist_ok=True)
            torch.save(our_model.state_dict(), model_path)
            
            logger.info(f"Modèle NIH mappé et sauvegardé dans {model_path}")
            logger.info(f"Mapping: {our_to_nih_indices}")
            
            return model_path
        
    except Exception as e:
        logger.error(f"Erreur lors de la création du modèle compatible: {e}")
        return None

def main():
    """Fonction principale pour télécharger et configurer le modèle."""
    logging.basicConfig(level=logging.INFO)
    
    print("🏥 Configuration du modèle NIH ChestX-ray14 pour RadioX")
    print("=" * 60)
    
    # Option 1: torchxrayvision (recommandé)
    print("\n1. Téléchargement depuis torchxrayvision...")
    model_path = create_nih_compatible_model()
    
    if model_path and os.path.exists(model_path):
        print(f"✅ Modèle NIH configuré: {model_path}")
        print(f"📊 Taille: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
        
        # Mettre à jour la variable d'environnement
        os.environ['MODEL_PATH'] = model_path
        print(f"🔧 MODEL_PATH mis à jour: {model_path}")
        
        print("\n🎯 Instructions:")
        print("1. Redémarrez le service ML")
        print("2. Le modèle utilisera maintenant les poids NIH réels")
        print("3. Les prédictions seront basées sur un modèle entraîné sur 112,000+ images")
        
    else:
        print("❌ Échec du téléchargement du modèle")
        print("\n🔄 Alternatives:")
        print("- Installez torchxrayvision: pip install torchxrayvision")
        print("- Ou utilisez un modèle local dans models/")

if __name__ == "__main__":
    main()
