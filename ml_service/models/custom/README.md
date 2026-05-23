# Modèles spécialisés RadioX

Placez les poids entraînés ici :

| Fichier | Module | Description |
|---------|--------|-------------|
| `tb_classifier.pt` | `pipelines/tb_model.py` | Classifieur TB binaire (Montgomery / Shenzhen) |
| `fracture_yolov8.pt` | `pipelines/fracture_detector.py` | YOLOv8 détection fracture costale |

Entraînement TB : `python train_tb.py` depuis `ml_service/`.

Fracture YOLO : exporter un modèle Ultralytics entraîné sur radiographies thoraciques annotées (format YOLO).

Si absent → fallback heuristique (logs + champ `alerts`).
