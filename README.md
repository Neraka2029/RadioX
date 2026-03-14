# RadioX.AI 🏥

> Application web d'analyse automatisée de radiographies thoraciques par Intelligence Artificielle

![Python](https://img.shields.io/badge/Python-3.11-blue)
![React](https://img.shields.io/badge/React-18-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

## Aperçu

RadioX.AI détecte automatiquement **14 pathologies pulmonaires** sur des radiographies thoraciques grâce à un modèle DenseNet-121 pré-entraîné sur plus de 700 000 images (NIH ChestX-ray14, CheXpert, MIMIC-CXR, PadChest).

## Fonctionnalités

- Upload de radiographies (PNG, JPEG, DICOM)
- Détection de 14 pathologies NIH avec score de probabilité
- Visualisation Grad-CAM (carte de chaleur des zones analysées)
- Authentification sécurisée JWT
- Gestion des patients et historique des analyses

## Démarrage rapide (Windows)

### Prérequis (à installer une seule fois)
1. [Python 3.11+](https://www.python.org/downloads/) — cocher **"Add Python to PATH"**
2. [Node.js 18+](https://nodejs.org/)

### Lancement
```
Double-clic sur START_RADIOX.bat
```
L'application s'ouvre automatiquement sur http://localhost:3000

**Compte démo :** `demo@radiox.ai` / `demo123`

## Architecture

```
RadioX/
├── START_RADIOX.bat        ← Lanceur Windows (double-clic)
├── frontend/               ← React 18 + Vite  (port 3000)
├── backend/                ← FastAPI + SQLite  (port 8000)
├── ml_service/             ← PyTorch + torchxrayvision (port 8001)
└── docker-compose.yml      ← Déploiement Docker
```

## Déploiement Docker

```bash
docker-compose up -d
```

## Stack technique

| Couche | Technologie |
|--------|-------------|
| Frontend | React 18, Vite |
| Backend | FastAPI, SQLAlchemy, JWT |
| IA | PyTorch, torchxrayvision, DenseNet-121 |
| Base de données | SQLite / PostgreSQL |

## Modèle IA

- **Architecture** : DenseNet-121
- **Pré-entraînement** : NIH ChestX-ray14 + CheXpert + MIMIC-CXR + PadChest (~725 000 images)
- **Bibliothèque** : [torchxrayvision](https://github.com/mlmed/torchxrayvision)
- **Téléchargement automatique** au premier démarrage (~28 MB)

## Pathologies détectées

Atélectasie • Consolidation • Infiltration • Pneumothorax • Œdème • Épanchement pleural • Pneumonie • Épaississement pleural • Cardiomégalie • Nodule • Masse

## Avertissement

> RadioX.AI est un outil d'aide au diagnostic et **ne remplace pas l'avis d'un médecin radiologue**. Les prédictions doivent être interprétées par un professionnel de santé qualifié.

## Licence

MIT — voir [LICENSE](LICENSE)
