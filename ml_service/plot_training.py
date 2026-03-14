"""
RadioX.AI - Visualisation des résultats d'entraînement
Génère des graphiques à partir de training_history.json

Usage :
    python plot_training.py
    python plot_training.py --history models/training_history.json
"""
import json
import argparse
import os
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # Sans interface graphique
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("matplotlib non installé. Lancez : pip install matplotlib")


PATHOLOGY_LABELS = [
    "Normal", "Pneumonie", "Tuberculose",
    "Cancer du poumon", "Epanchement pleural", "Cardiomegalie",
]

COLORS = {
    "Normal":              "#00ffc8",
    "Pneumonie":           "#ff4466",
    "Tuberculose":         "#ff7733",
    "Cancer du poumon":    "#ff4466",
    "Epanchement pleural": "#ffaa00",
    "Cardiomegalie":       "#4488ff",
}

BG      = "#080d1a"
SURFACE = "#0d1528"
CARD    = "#162040"
CYAN    = "#00d4ff"
TEXT    = "#e8f4ff"
MUTED   = "#6b7fa3"


def load_history(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def style_ax(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.spines[:].set_color(CARD)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(color=CARD, linestyle="--", linewidth=0.6, alpha=0.8)


def plot_all(history: list, out_dir: str = "models"):
    Path(out_dir).mkdir(exist_ok=True)
    epochs      = [h["epoch"]      for h in history]
    train_loss  = [h["train_loss"] for h in history]
    val_loss    = [h["loss"]       for h in history]
    mean_auc    = [h["mean_auc"]   for h in history]

    # AUC par pathologie
    per_class = {label: [] for label in PATHOLOGY_LABELS}
    for h in history:
        auc_dict = h.get("per_class_auc", {})
        for label in PATHOLOGY_LABELS:
            per_class[label].append(auc_dict.get(label, None))

    # ── Figure principale ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle(
        "RadioX.AI — Rapport d'entraînement DenseNet-121",
        color=TEXT, fontsize=16, fontweight="bold", y=0.97
    )

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.97, top=0.91, bottom=0.08)

    # 1. Loss courbes
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, color=CYAN,     linewidth=2, marker="o", markersize=5, label="Train Loss")
    ax1.plot(epochs, val_loss,   color="#ff4466", linewidth=2, marker="s", markersize=5, label="Val Loss")
    ax1.fill_between(epochs, train_loss, val_loss, alpha=0.08, color=CYAN)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend(facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT, fontsize=9)
    style_ax(ax1, "📉 Courbes de Loss")

    # 2. AUC moyen
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(epochs, mean_auc, color=[CYAN if v == max(mean_auc) else "#4488ff" for v in mean_auc],
                   edgecolor=BG, linewidth=0.5, width=0.6)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC moyen")
    ax2.axhline(y=0.7, color="#ffaa00", linestyle="--", linewidth=1, alpha=0.7, label="Seuil 0.70")
    ax2.axhline(y=0.8, color="#00ffc8", linestyle="--", linewidth=1, alpha=0.7, label="Seuil 0.80")
    for bar, val in zip(bars, mean_auc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
    ax2.legend(facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT, fontsize=8)
    style_ax(ax2, "🎯 AUC Moyen par Epoch")

    # 3. AUC par pathologie (dernière epoch)
    ax3 = fig.add_subplot(gs[0, 2])
    last_auc = history[-1].get("per_class_auc", {})
    labels_present = [l for l in PATHOLOGY_LABELS if l in last_auc]
    values         = [last_auc[l] for l in labels_present]
    colors_bars    = [COLORS.get(l, CYAN) for l in labels_present]
    short_labels   = [l.replace(" du ", "\n").replace(" pleural", "\npleural") for l in labels_present]

    bars3 = ax3.barh(short_labels, values, color=colors_bars, edgecolor=BG, linewidth=0.5)
    ax3.set_xlim(0, 1)
    ax3.axvline(x=0.7, color="#ffaa00", linestyle="--", linewidth=1, alpha=0.8)
    for bar, val in zip(bars3, values):
        ax3.text(min(val + 0.01, 0.97), bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", color=TEXT, fontsize=9, fontweight="bold")
    style_ax(ax3, f"🫁 AUC par Pathologie (Epoch {history[-1]['epoch']})")
    ax3.tick_params(axis="y", labelsize=8)

    # 4. Évolution AUC par pathologie
    ax4 = fig.add_subplot(gs[1, :2])
    for label in PATHOLOGY_LABELS:
        vals = per_class[label]
        valid_epochs = [e for e, v in zip(epochs, vals) if v is not None]
        valid_vals   = [v for v in vals if v is not None]
        if valid_vals:
            color = COLORS.get(label, CYAN)
            ax4.plot(valid_epochs, valid_vals, color=color, linewidth=2,
                     marker="o", markersize=5, label=label)
            ax4.annotate(f"{valid_vals[-1]:.2f}", (valid_epochs[-1], valid_vals[-1]),
                         textcoords="offset points", xytext=(6, 0),
                         color=color, fontsize=8, fontweight="bold")
    ax4.set_ylim(0, 1)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("AUC")
    ax4.axhline(y=0.7, color="#ffaa00", linestyle="--", linewidth=0.8, alpha=0.5)
    ax4.legend(facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT, fontsize=8,
               loc="lower right", ncol=2)
    style_ax(ax4, "📈 Évolution AUC par Pathologie")

    # 5. Résumé statistiques
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(SURFACE)
    ax5.axis("off")

    best_epoch  = epochs[mean_auc.index(max(mean_auc))]
    best_auc    = max(mean_auc)
    final_loss  = val_loss[-1]
    total_ep    = len(epochs)

    stats = [
        ("Epochs complétées",  f"{total_ep}"),
        ("Meilleur AUC moyen", f"{best_auc:.4f}"),
        ("Meilleure epoch",    f"{best_epoch}"),
        ("Loss finale (val)",  f"{final_loss:.4f}"),
        ("Dataset",            "NIH ChestX-ray14"),
        ("Modèle",             "DenseNet-121"),
        ("Pathologies",        f"{len(labels_present)}"),
    ]

    ax5.text(0.5, 0.95, "📋 Résumé", transform=ax5.transAxes,
             color=CYAN, fontsize=12, fontweight="bold", ha="center", va="top")

    for i, (key, val) in enumerate(stats):
        y = 0.82 - i * 0.11
        ax5.text(0.05, y, key, transform=ax5.transAxes,
                 color=MUTED, fontsize=9, va="top")
        ax5.text(0.95, y, val, transform=ax5.transAxes,
                 color=TEXT, fontsize=9, fontweight="bold", ha="right", va="top")
        ax5.axhline(y=y - 0.02, xmin=0.02, xmax=0.98,
                    color=CARD, linewidth=0.5)

    ax5.set_title("", color=TEXT)
    ax5.spines[:].set_visible(False)

    # Sauvegarde
    out_path = os.path.join(out_dir, "training_report.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n✅ Graphique sauvegardé : {out_path}")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="RadioX.AI — Visualisation entraînement")
    parser.add_argument("--history", default="models/training_history.json")
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    if not MATPLOTLIB_OK:
        return

    if not os.path.exists(args.history):
        print(f"❌ Fichier introuvable : {args.history}")
        print("   L'entraînement doit être terminé pour générer le rapport.")
        return

    history = load_history(args.history)
    print(f"✅ {len(history)} epochs chargées")
    plot_all(history, args.out_dir)


if __name__ == "__main__":
    main()
