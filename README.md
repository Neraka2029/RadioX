# RadioX.AI 🏥

> Application web d'analyse automatisée de radiographies thoraciques par Intelligence Artificielle

![Python](https://img.shields.io/badge/Python-3.11-blue)
![React](https://img.shields.io/badge/React-18-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

## Aperçu

RadioX.AI détecte automatiquement **16 pathologies pulmonaires** sur des radiographies thoraciques grâce à un modèle DenseNet-121 NIH (torchxrayvision) pour les prédictions officielles et Grad-CAM, enrichi par un second modèle **DenseNet-ALL** pour l'aide à la décision clinique (tuberculose, risque traumatique thoracique).

## Fonctionnalités

- Upload de radiographies (PNG, JPEG, DICOM)
- Détection de 16 pathologies (14 NIH + Tuberculose + Fracture costale) avec scores de probabilité
- Visualisation Grad-CAM (carte de chaleur sur le modèle NIH)
- **Priorité clinique** du résultat principal (TB → trauma → pathologie NIH dominante)
- Conclusion clinique et export PDF structuré
- Authentification sécurisée JWT
- Historique des analyses et rapports

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

| Modèle | Poids | Rôle |
|--------|-------|------|
| **NIH** | `densenet121-res224-nih` | Prédictions officielles des 14 pathologies, Grad-CAM |
| **ALL** | `densenet121-res224-all` | Enrichissement TB, risque traumatique (aide à la décision) |

- **Bibliothèque** : [torchxrayvision](https://github.com/mlmed/torchxrayvision)
- **Téléchargement automatique** au premier démarrage (~28 MB par modèle)

### Pipeline ML (`ml_service/pipelines/`)

| Fichier | Rôle |
|---------|------|
| `inference.py` | Pipeline principal NIH + scores TB/fracture |
| `xrv_all_support.py` | Singleton TorchXRayVision ALL (lazy load) |
| `gradcam.py` | Heatmap sur le modèle NIH |
| `heuristic_predictors.py` | Fallback NIH si ALL indisponible |
| `decision_support.py` | Alertes et aide à la décision |

### Priorité clinique (frontend)

Le **résultat principal** n'est plus le simple score maximal. Logique dans `frontend/src/utils/clinicalPriority.js` :

1. **Tuberculose** si score TB ≥ 30 % et dominant sur Infiltration / Pneumonie
2. **Risque traumatique thoracique** si score trauma ≥ 30 % et dominant sur Infiltration / Pneumonie / Œdème (jamais « Fracture costale » sauf mode YOLO avec détections)
3. Sinon : pathologie NIH la plus élevée

## Pathologies détectées

**14 pathologies NIH :** Atélectasie • Consolidation • Infiltration • Pneumothorax • Œdème • Épanchement pleural • Pneumonie • Épaississement pleural • Cardiomégalie • Nodule • Masse • Emphysème • Fibrose • Hernie

**Pathologies additionnelles (scores dérivés XRV-ALL) :** Tuberculose • Fracture costale (proxy de risque traumatique)

## API `/predict` (champs principaux)

| Champ | Description |
|-------|-------------|
| `predictions` | Liste complète (NIH + Normal + TB + fracture) |
| `nih_predictions` | Dict des 14 pathologies NIH + Normal |
| `tuberculosis_probability` | Score TB brut (XRV-ALL) |
| `fracture_risk_score` | Score risque traumatique (XRV-ALL) |
| `tuberculosis_mode` / `fracture_mode` | `"xrv-all"` ou `"heuristic"` (fallback) |
| `heatmap_base64` | Grad-CAM (modèle NIH) |
| `recommendations` | Recommandations textuelles |
| `confidence` | Score de confiance global |

## Export PDF

Le rapport PDF inclut :

- Images (originale + Grad-CAM)
- **Conclusion clinique principale** (priorité clinique + anomalies contributrices)
- Tableau détaillé des pathologies

Les recommandations cliniques restent affichées dans l'interface web, pas dans le PDF.

## Avertissement

> RadioX.AI est un outil d'aide au diagnostic et **ne remplace pas l'avis d'un médecin radiologue**. Les prédictions doivent être interprétées par un professionnel de santé qualifié. Les scores TB et fracture sont des indicateurs dérivés, non confirmatoires.

## Licence

MIT — voir [LICENSE](LICENSE)
