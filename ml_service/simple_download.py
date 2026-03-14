#!/usr/bin/env python3
"""
Téléchargement simple du modèle NIH ChestX-ray14.
"""

import os
import torch
import urllib.request
from pathlib import Path

def download_model():
    """Télécharge le modèle NIH depuis une URL directe."""
    
    # URL du modèle pré-entraîné (alternative plus fiable)
    model_url = "https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
    
    model_path = "models/nih_densenet121.pt"
    os.makedirs("models", exist_ok=True)
    
    try:
        print("📥 Téléchargement du modèle NIH...")
        print(f"URL: {model_url}")
        
        # Téléchargement avec barre de progression
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                print(f"\r[{'█' * (percent // 5)}{' ' * (20 - percent // 5)}] {percent}%", end='')
            else:
                print(f"\rTéléchargé: {downloaded / 1024 / 1024:.1f} MB", end='')
        
        urllib.request.urlretrieve(model_url, model_path, show_progress)
        print(f"\n✅ Modèle téléchargé: {model_path}")
        
        # Vérifier la taille
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"📊 Taille: {size_mb:.1f} MB")
        
        return model_path
        
    except Exception as e:
        print(f"\n❌ Erreur de téléchargement: {e}")
        return None

def create_simple_model():
    """Crée un modèle simple avec des poids pré-entraînés de base."""
    try:
        import torchxrayvision as xrv
        
        print("🔧 Création du modèle NIH...")
        
        # Charger le modèle pré-entraîné
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        
        # Sauvegarder dans notre format
        model_path = "models/nih_densenet121.pt"
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        
        print(f"✅ Modèle sauvegardé: {model_path}")
        
        # Afficher les pathologies supportées
        print(f"🏥 Pathologies supportées ({len(model.pathologies)}):")
        for i, path in enumerate(model.pathologies):
            print(f"  {i+1:2d}. {path}")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def main():
    print("🏥 Téléchargement du modèle NIH ChestX-ray14")
    print("=" * 50)
    
    # Essayer le téléchargement direct
    model_path = download_model()
    
    if not model_path:
        print("\n🔄 Tentative avec torchxrayvision...")
        model_path = create_simple_model()
    
    if model_path and os.path.exists(model_path):
        print(f"\n✅ Succès! Modèle disponible: {model_path}")
        print("\n🎯 Prochaines étapes:")
        print("1. Redémarrez le service ML")
        print("2. Le modèle utilisera les poids NIH réels")
    else:
        print("\n❌ Échec. Vérifiez votre connexion internet.")

if __name__ == "__main__":
    main()
