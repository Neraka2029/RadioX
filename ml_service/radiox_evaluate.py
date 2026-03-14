"""
RadioX. — Évaluation et visualisation du modèle pré-entraîné
Génère des graphiques de performance pour le rapport

Usage:
    python radiox_evaluate.py \
        --csv D:/datasets/nih/labels/val_labels.csv \
        --img_dir D:/datasets/nih/images/val \
        --subset 0.05
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────
BG      = "#080d1a"
SURFACE = "#0d1528"
CARD    = "#162040"
CYAN    = "#00d4ff"
TEXT    = "#e8f4ff"
MUTED   = "#6b7fa3"

XRV_TO_NIH = {
    "Atelectasis":        "Atelectasis",
    "Consolidation":      "Consolidation",
    "Infiltration":       "Infiltration",
    "Pneumothorax":       "Pneumothorax",
    "Edema":              "Edema",
    "Effusion":           "Effusion",
    "Pneumonia":          "Pneumonia",
    "Pleural_Thickening": "Pleural_Thickening",
    "Cardiomegaly":       "Cardiomegaly",
    "Nodule":             "Nodule",
    "Mass":               "Mass",
}

LABELS_FR = {
    "Atelectasis":        "Atelectasie",
    "Consolidation":      "Consolidation",
    "Infiltration":       "Infiltration",
    "Pneumothorax":       "Pneumothorax",
    "Edema":              "Oedeme",
    "Effusion":           "Epanchement pleural",
    "Pneumonia":          "Pneumonie",
    "Pleural_Thickening": "Epaississement pleural",
    "Cardiomegaly":       "Cardiomegalie",
    "Nodule":             "Nodule",
    "Mass":               "Masse",
}

COLORS = [
    "#00d4ff","#ff4466","#00ffc8","#ffaa00","#4488ff",
    "#ff6348","#a29bfe","#00d2d3","#ff9f43","#ffd32a","#ee5a24"
]

def style_ax(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(CARD)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(color=CARD, linestyle="--", linewidth=0.6, alpha=0.8)

def load_model():
    import torchxrayvision as xrv
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model, list(model.pathologies)

def preprocess_image(img_path):
    import torchxrayvision as xrv
    import skimage.transform
    from PIL import Image

    img = Image.open(img_path)
    if img.mode == "RGB":
        arr = np.array(img).astype(np.float32)
        arr = 0.2989*arr[:,:,0] + 0.5870*arr[:,:,1] + 0.1140*arr[:,:,2]
    else:
        arr = np.array(img.convert("L")).astype(np.float32)

    arr = xrv.datasets.normalize(arr, maxval=255, reshape=False)
    arr = skimage.transform.resize(arr, (224, 224), anti_aliasing=True, preserve_range=True).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

def evaluate(csv_path, img_dir, subset=0.05):
    print(f"Chargement du modèle...")
    model, xrv_labels = load_model()

    print(f"Chargement du CSV...")
    df = pd.read_csv(csv_path)
    df = df.sample(frac=subset, random_state=42).reset_index(drop=True)
    print(f"Évaluation sur {len(df)} images ({subset*100:.0f}% du dataset)")

    all_probs = {lbl: [] for lbl in XRV_TO_NIH}
    all_labels = {lbl: [] for lbl in XRV_TO_NIH}
    errors = 0

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_path = Path(img_dir) / row["Image Index"]
            if not img_path.exists():
                errors += 1
                continue
            try:
                tensor = preprocess_image(img_path)
                output = model(tensor).squeeze().numpy()
                output = np.clip(output, 0, 1)

                for xrv_lbl in XRV_TO_NIH:
                    try:
                        idx = xrv_labels.index(xrv_lbl)
                        prob = float(output[idx])
                    except ValueError:
                        prob = 0.0
                    all_probs[xrv_lbl].append(prob)
                    all_labels[xrv_lbl].append(int(row.get(xrv_lbl, 0)))
            except Exception:
                errors += 1

    if errors > 0:
        print(f"⚠️ {errors} images ignorées")

    return all_probs, all_labels

def compute_metrics(all_probs, all_labels):
    metrics = {}
    for lbl in XRV_TO_NIH:
        y_true = np.array(all_labels[lbl])
        y_score = np.array(all_probs[lbl])
        if len(np.unique(y_true)) < 2:
            continue
        try:
            auc = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            metrics[lbl] = {"auc": auc, "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                            "n_pos": int(y_true.sum()), "n_total": len(y_true)}
        except Exception:
            pass
    return metrics

def plot_report(metrics, out_path="radiox_report.png"):
    labels = list(metrics.keys())
    aucs   = [metrics[l]["auc"] for l in labels]
    labels_fr = [LABELS_FR.get(l, l) for l in labels]

    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle("RadioX. — Rapport de performance DenseNet-121\n(NIH+CheXpert+MIMIC — densenet121-res224-all)",
                 color=TEXT, fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.92, bottom=0.07)

    # 1 — AUC par pathologie (barres)
    ax1 = fig.add_subplot(gs[0, :2])
    colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
    bars = ax1.barh(labels_fr, aucs, color=colors, edgecolor=BG, linewidth=0.5)
    ax1.set_xlim(0, 1)
    ax1.axvline(x=0.5, color=MUTED,    linestyle="--", linewidth=1, alpha=0.7, label="Aléatoire (0.50)")
    ax1.axvline(x=0.7, color="#ffaa00", linestyle="--", linewidth=1, alpha=0.8, label="Seuil 0.70")
    ax1.axvline(x=0.8, color="#00ffc8", linestyle="--", linewidth=1, alpha=0.8, label="Seuil 0.80")
    for bar, auc in zip(bars, aucs):
        ax1.text(min(auc + 0.01, 0.96), bar.get_y() + bar.get_height()/2,
                 f"{auc:.3f}", va="center", color=TEXT, fontsize=9, fontweight="bold")
    ax1.legend(facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT, fontsize=8)
    style_ax(ax1, "🎯 AUC par Pathologie (Area Under ROC Curve)")

    # 2 — Résumé stats
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(SURFACE)
    ax2.axis("off")
    mean_auc = np.mean(aucs)
    best_lbl = labels_fr[np.argmax(aucs)]
    worst_lbl = labels_fr[np.argmin(aucs)]
    n_above_07 = sum(1 for a in aucs if a >= 0.7)

    stats = [
        ("Modèle",           "DenseNet-121"),
        ("Dataset entraîn.", "NIH+CheXpert+MIMIC"),
        ("Pathologies",      f"{len(labels)}"),
        ("AUC moyen",        f"{mean_auc:.4f}"),
        ("Meilleure",        f"{best_lbl} ({max(aucs):.3f})"),
        ("Plus difficile",   f"{worst_lbl} ({min(aucs):.3f})"),
        ("AUC ≥ 0.70",       f"{n_above_07}/{len(labels)}"),
    ]

    ax2.text(0.5, 0.97, "📋 Résumé", transform=ax2.transAxes,
             color=CYAN, fontsize=12, fontweight="bold", ha="center", va="top")
    for i, (k, v) in enumerate(stats):
        y = 0.84 - i * 0.11
        ax2.text(0.05, y, k, transform=ax2.transAxes, color=MUTED, fontsize=8.5, va="top")
        ax2.text(0.97, y, v, transform=ax2.transAxes, color=TEXT, fontsize=8.5,
                 fontweight="bold", ha="right", va="top")
        if i < len(stats)-1:
            line = plt.Line2D([0.03, 0.97], [y-0.025, y-0.025],
                              transform=ax2.transAxes, color=CARD, linewidth=0.5)
            ax2.add_line(line)
    style_ax(ax2, "")
    ax2.set_title("📋 Résumé", color=CYAN, fontsize=11, fontweight="bold", pad=10)

    # 3 — Courbes ROC (top 6)
    ax3 = fig.add_subplot(gs[1, :2])
    top6 = sorted(metrics.keys(), key=lambda l: metrics[l]["auc"], reverse=True)[:6]
    for i, lbl in enumerate(top6):
        m = metrics[lbl]
        ax3.plot(m["fpr"], m["tpr"], color=COLORS[i], linewidth=2,
                 label=f"{LABELS_FR.get(lbl, lbl)} (AUC={m['auc']:.3f})")
    ax3.plot([0,1],[0,1], color=MUTED, linestyle="--", linewidth=1, alpha=0.5)
    ax3.set_xlabel("Taux de faux positifs (FPR)")
    ax3.set_ylabel("Taux de vrais positifs (TPR)")
    ax3.legend(facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT, fontsize=8,
               loc="lower right")
    style_ax(ax3, "📈 Courbes ROC — Top 6 Pathologies")

    # 4 — Distribution des scores
    ax4 = fig.add_subplot(gs[1, 2])
    sorted_aucs = sorted(aucs)
    sorted_lbls = [labels_fr[aucs.index(a)] for a in sorted_aucs]
    ax4.barh(sorted_lbls, sorted_aucs,
             color=["#00ffc8" if a >= 0.7 else "#ffaa00" if a >= 0.6 else "#ff4466" for a in sorted_aucs],
             edgecolor=BG, linewidth=0.5)
    ax4.axvline(x=0.7, color="#ffaa00", linestyle="--", linewidth=1)
    ax4.set_xlim(0.4, 1.0)
    for i, (lbl, auc) in enumerate(zip(sorted_lbls, sorted_aucs)):
        ax4.text(auc + 0.005, i, f"{auc:.2f}", va="center", color=TEXT, fontsize=8)
    style_ax(ax4, "📊 Classement AUC")
    ax4.tick_params(axis="y", labelsize=7.5)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n✅ Graphique sauvegardé : {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--subset",  type=float, default=0.05)
    parser.add_argument("--out",     default="radiox_report.png")
    args = parser.parse_args()

    all_probs, all_labels = evaluate(args.csv, args.img_dir, args.subset)
    metrics = compute_metrics(all_probs, all_labels)

    print(f"\n{'='*50}")
    print(f"{'Pathologie':<25} {'AUC':>6}  {'N+':>5}/{' N total':>7}")
    print(f"{'='*50}")
    for lbl, m in sorted(metrics.items(), key=lambda x: x[1]["auc"], reverse=True):
        print(f"{LABELS_FR.get(lbl,lbl):<25} {m['auc']:.4f}  {m['n_pos']:>5}/{m['n_total']:>7}")
    print(f"{'='*50}")
    print(f"{'AUC MOYEN':<25} {np.mean([m['auc'] for m in metrics.values()]):.4f}")

    with open("radiox_metrics.json", "w") as f:
        json.dump({k: {"auc": v["auc"], "n_pos": v["n_pos"], "n_total": v["n_total"]}
                   for k, v in metrics.items()}, f, indent=2)
    print(f"✅ Métriques sauvegardées : radiox_metrics.json")

    plot_report(metrics, args.out)

if __name__ == "__main__":
    main()
