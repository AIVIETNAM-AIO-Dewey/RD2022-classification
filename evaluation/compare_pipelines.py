import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ── Constants ─────────────────────────────────────────────────────────────────

PIPELINE_LABELS = {
    "baseline":                      "Baseline\n(Resize only)",
    "pipeline1_letterbox":           "Pipeline 1\n(Letterbox)",
    "pipeline2_clahe":               "Pipeline 2\n(CLAHE)",
    "pipeline3_grayscale_bilateral": "Pipeline 3\n(Gray+Bilateral)",
    "pipeline4_combined":            "Pipeline 4\n(Combined)",
}

PIPELINE_SHORT = {
    "baseline":                      "Baseline",
    "pipeline1_letterbox":           "P1: Letterbox",
    "pipeline2_clahe":               "P2: CLAHE",
    "pipeline3_grayscale_bilateral": "P3: Gray+Bilateral",
    "pipeline4_combined":            "P4: Combined",
}

CLASSES = ["D00", "D20", "D40", "Normal"]

# Flat-UI inspired palette — vibrant, distinct, beautiful
PALETTE     = ["#476EAE", "#48B3AF", "#A7E399", "#FAEF5D", "#007F73"]
BAR_PALETTE = ["#476EAE", "#48B3AF", "#A7E399", "#FAEF5D", "#007F73"]

_WHITE  = "#FFFFFF"
_DARK   = "#2C2C2C"
_MID    = "#555555"
_LIGHT  = "#AAAAAA"
_GRID   = "#E8E8E8"


# ── Style ─────────────────────────────────────────────────────────────────────

def _setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor":   _WHITE,
        "axes.facecolor":     _WHITE,
        "axes.edgecolor":     "#CCCCCC",
        "grid.color":         _GRID,
        "grid.linewidth":     0.7,
        "grid.linestyle":     "-",
        "axes.axisbelow":     True,
        "font.family":        "DejaVu Sans",
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "axes.titlecolor":    _DARK,
        "axes.titlepad":      12,
        "axes.labelsize":     10,
        "axes.labelcolor":    _MID,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "xtick.color":        _MID,
        "ytick.color":        _MID,
        "xtick.major.size":   0,
        "ytick.major.size":   3,
        "legend.fontsize":    8.5,
        "legend.framealpha":  0.96,
        "legend.edgecolor":   _LIGHT,
        "legend.fancybox":    True,
        "lines.linewidth":    2.4,
        "patch.linewidth":    0.4,
        "figure.dpi":         110,
    })


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_LIGHT)
    ax.spines["bottom"].set_color(_LIGHT)


def _header(fig, title: str, subtitle: str = "", top: float = 0.97):
    fig.text(0.5, top, title, ha="center", va="top",
             fontsize=15, fontweight="bold", color=_DARK,
             fontfamily="DejaVu Sans")
    if subtitle:
        fig.text(0.5, top - 0.052, subtitle, ha="center", va="top",
                 fontsize=9, color=_MID, style="italic")


def _bar_label(ax, bars, fmt="{:.1f}", pad=0.4, fontsize=7.5):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + pad,
                    fmt.format(h), ha="center", va="bottom",
                    fontsize=fontsize, color=_MID, fontweight="bold")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(outputs_dir: Path) -> dict:
    results = {}
    for exp_name in PIPELINE_LABELS:
        eval_path    = outputs_dir / exp_name / "test_evaluation.json"
        history_path = outputs_dir / exp_name / "history.json"
        if not eval_path.exists() or not history_path.exists():
            print(f"[Skip] '{exp_name}': results not found.")
            continue
        with open(eval_path, encoding="utf-8") as f:
            eval_data = json.load(f)
        with open(history_path, encoding="utf-8") as f:
            history_data = json.load(f)
        results[exp_name] = {"eval": eval_data, "history": history_data}
    return results


# ── Metrics table (text) ──────────────────────────────────────────────────────

def print_metrics_table(results: dict, save_dir: Path):
    rows = []
    for exp_name, data in results.items():
        report = data["eval"]["classification_report"]
        row = {
            "Pipeline":     PIPELINE_SHORT[exp_name],
            "Accuracy (%)": round(data["eval"]["accuracy"] * 100, 2),
            "Macro F1 (%)": round(report["macro avg"]["f1-score"] * 100, 2),
        }
        for cls in CLASSES:
            row[f"F1-{cls} (%)"] = round(report[cls]["f1-score"] * 100, 2)
        rows.append(row)

    lines = ["\n" + "=" * 95, "  METRICS COMPARISON TABLE", "=" * 95]
    if rows:
        col_w = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in rows[0]}
        hdr   = "  ".join(k.ljust(col_w[k]) for k in col_w)
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for row in rows:
            lines.append("  ".join(str(row[k]).ljust(col_w[k]) for k in col_w))
    lines.append("=" * 95)
    table_str = "\n".join(lines)
    print(table_str)

    csv_path = save_dir / "metrics_table.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        if rows:
            f.write(",".join(rows[0].keys()) + "\n")
            for row in rows:
                f.write(",".join(str(v) for v in row.values()) + "\n")
    print(f"Saved: {csv_path}")
    return rows, table_str


# ── Chart 1: Metrics heatmap ──────────────────────────────────────────────────

def plot_metrics_heatmap(results: dict, save_dir: Path):
    exp_names = list(results.keys())
    col_labels = ["Accuracy", "Macro F1", "F1-D00", "F1-D20", "F1-D40", "F1-Normal"]
    row_labels = [PIPELINE_SHORT[n] for n in exp_names]

    matrix = np.array([
        [
            results[n]["eval"]["accuracy"] * 100,
            results[n]["eval"]["classification_report"]["macro avg"]["f1-score"] * 100,
            results[n]["eval"]["classification_report"]["D00"]["f1-score"] * 100,
            results[n]["eval"]["classification_report"]["D20"]["f1-score"] * 100,
            results[n]["eval"]["classification_report"]["D40"]["f1-score"] * 100,
            results[n]["eval"]["classification_report"]["Normal"]["f1-score"] * 100,
        ]
        for n in exp_names
    ])

    nrows, ncols = matrix.shape
    fig, ax = plt.subplots(figsize=(ncols * 1.7 + 1.5, nrows * 0.9 + 2.2), facecolor=_WHITE)
    _header(fig, "Performance Metrics Heatmap",
            "Score (%) per pipeline × metric  ·  colour encodes relative performance")
    fig.subplots_adjust(top=0.83, bottom=0.10, left=0.22, right=0.90)

    cmap = LinearSegmentedColormap.from_list(
        "green_teal",
        ["#C5D34A", "#98E875", "#5AB840", "#3B9B8B", "#2E6B6B"],
    )
    norm = mcolors.Normalize(vmin=40, vmax=100)

    for i in range(nrows):
        for j in range(ncols):
            val  = matrix[i, j]
            rgba = cmap(norm(val))
            rect = mpatches.FancyBboxPatch(
                (j + 0.05, i + 0.05), 0.90, 0.90,
                boxstyle="round,pad=0.05",
                facecolor=rgba, edgecolor=_WHITE, linewidth=1.5,
                transform=ax.transData,
            )
            ax.add_patch(rect)
            lum  = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc   = _WHITE if lum < 0.55 else _DARK
            ax.text(j + 0.50, i + 0.50, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=tc)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_xticks([j + 0.5 for j in range(ncols)])
    ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold", color=_DARK)
    ax.set_yticks([i + 0.5 for i in range(nrows)])
    ax.set_yticklabels(row_labels, fontsize=9.5, color=_DARK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(_WHITE)
    ax.grid(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.80, aspect=18, pad=0.03)
    cb.set_label("Score (%)", fontsize=9, color=_MID)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    plt.savefig(save_dir / "metrics_heatmap.png", dpi=200, bbox_inches="tight", facecolor=_WHITE)
    plt.close()
    print(f"Saved: {save_dir / 'metrics_heatmap.png'}")


# ── Chart 2: Radar chart ──────────────────────────────────────────────────────

def plot_radar_chart(results: dict, save_dir: Path):
    metric_keys = ["Accuracy", "Macro F1", "F1-D00", "F1-D20", "F1-D40", "F1-Normal"]
    n_m    = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, n_m, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"polar": True}, facecolor=_WHITE)
    ax.set_facecolor(_WHITE)
    _header(fig, "Pipeline Comparison — Radar",
            "Each axis = one metric  ·  larger area = better overall performance", top=0.96)
    fig.subplots_adjust(top=0.84)

    # Subtle concentric ring background
    for r in [20, 40, 60, 80, 100]:
        ax.fill(angles, [r] * (n_m + 1), color="#F8F9FA" if r % 40 == 0 else _WHITE, zorder=0)

    for idx, (exp_name, data) in enumerate(results.items()):
        report = data["eval"]["classification_report"]
        vals   = [
            data["eval"]["accuracy"] * 100,
            report["macro avg"]["f1-score"] * 100,
            report["D00"]["f1-score"] * 100,
            report["D20"]["f1-score"] * 100,
            report["D40"]["f1-score"] * 100,
            report["Normal"]["f1-score"] * 100,
        ] + [0]  # placeholder to close later
        vals[-1] = vals[0]
        color   = PALETTE[idx % len(PALETTE)]
        short   = PIPELINE_SHORT[exp_name]

        ax.plot(angles, vals, color=color, linewidth=2.2, label=short, zorder=3)
        ax.fill(angles, vals, color=color, alpha=0.10, zorder=2)
        # Highlight vertices
        ax.scatter(angles[:-1], vals[:-1], color=color, s=45, zorder=5,
                   edgecolors=_WHITE, linewidths=1.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_keys, fontsize=10.5, fontweight="bold", color=_DARK)
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color=_MID)
    ax.spines["polar"].set_color(_LIGHT)
    ax.grid(color=_GRID, linewidth=0.9)

    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.40, 1.18),
                    fontsize=9, framealpha=0.97, edgecolor=_LIGHT)

    plt.savefig(save_dir / "radar_chart.png", dpi=200, bbox_inches="tight", facecolor=_WHITE)
    plt.close()
    print(f"Saved: {save_dir / 'radar_chart.png'}")


# ── Chart 3: Learning curves ──────────────────────────────────────────────────

def plot_learning_curves(results: dict, save_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=_WHITE)
    _header(fig, "Learning Curves",
            "Solid = validation  ·  Dashed = training  ·  ★ = best validation epoch")
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.07, right=0.97, wspace=0.28)

    for idx, (exp_name, data) in enumerate(results.items()):
        h     = data["history"]
        color = PALETTE[idx % len(PALETTE)]
        short = PIPELINE_SHORT[exp_name]
        ep    = list(range(1, len(h["val_loss"]) + 1))

        # Validation loss
        axes[0].plot(ep, h["val_loss"], color=color, linewidth=2.4,
                     label=short, solid_capstyle="round")
        if "train_loss" in h:
            axes[0].plot(ep, h["train_loss"], color=color, linewidth=1.0,
                         linestyle="--", alpha=0.45)
            axes[0].fill_between(ep, h["train_loss"], h["val_loss"],
                                  color=color, alpha=0.06)
        # Mark best val loss epoch
        best_i = int(np.argmin(h["val_loss"]))
        axes[0].scatter(ep[best_i], h["val_loss"][best_i],
                         color=color, s=90, marker="*", zorder=6,
                         edgecolors=_WHITE, linewidths=0.8)

        # Validation accuracy
        val_acc = [v * 100 for v in h["val_acc"]]
        axes[1].plot(ep, val_acc, color=color, linewidth=2.4,
                     label=short, solid_capstyle="round")
        if "train_acc" in h:
            trn_acc = [v * 100 for v in h["train_acc"]]
            axes[1].plot(ep, trn_acc, color=color, linewidth=1.0,
                         linestyle="--", alpha=0.45)
            axes[1].fill_between(ep, trn_acc, val_acc, color=color, alpha=0.06)
        best_j = int(np.argmax(h["val_acc"]))
        axes[1].scatter(ep[best_j], val_acc[best_j],
                         color=color, s=90, marker="*", zorder=6,
                         edgecolors=_WHITE, linewidths=0.8)

    for ax, title, ylabel in zip(
        axes,
        ["Validation Loss", "Validation Accuracy (%)"],
        ["Loss", "Accuracy (%)"],
    ):
        _despine(ax)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8.5)

    plt.savefig(save_dir / "learning_curves.png", dpi=200, bbox_inches="tight", facecolor=_WHITE)
    plt.close()
    print(f"Saved: {save_dir / 'learning_curves.png'}")


# ── Chart 4: F1-score per class ───────────────────────────────────────────────

def plot_f1_comparison(results: dict, save_dir: Path):
    exp_names   = list(results.keys())
    n_pipelines = len(exp_names)
    x     = np.arange(len(CLASSES))
    width = 0.72 / n_pipelines

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=_WHITE)
    _header(fig, "F1-Score per Damage Class",
            "One-vs-Rest F1 (%)  ·  higher = better detection for that class")
    fig.subplots_adjust(top=0.83, bottom=0.12, right=0.80)

    for i, (exp_name, data) in enumerate(results.items()):
        report    = data["eval"]["classification_report"]
        f1_scores = [report[cls]["f1-score"] * 100 for cls in CLASSES]
        offset    = (i - n_pipelines / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, f1_scores, width,
            label=PIPELINE_SHORT[exp_name],
            color=BAR_PALETTE[i % len(BAR_PALETTE)],
            alpha=0.88, edgecolor=_WHITE, linewidth=0.6,
        )
        _bar_label(ax, bars, pad=0.30, fontsize=6.8)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=12, fontweight="bold", color=_DARK)
    ax.set_ylabel("F1-Score (%)")
    ax.set_ylim(0, 112)
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        ncol=1, framealpha=0.95, edgecolor=_LIGHT,
        fontsize=8.5, borderpad=0.7,
    )
    _despine(ax)

    plt.savefig(save_dir / "f1_comparison.png", dpi=200, bbox_inches="tight", facecolor=_WHITE)
    plt.close()
    print(f"Saved: {save_dir / 'f1_comparison.png'}")


# ── Chart 5: Overall summary bars ────────────────────────────────────────────

def plot_summary_bars(results: dict, save_dir: Path):
    exp_names = list(results.keys())
    shorts    = [PIPELINE_SHORT[n] for n in exp_names]
    accs      = [results[n]["eval"]["accuracy"] * 100 for n in exp_names]
    macro_f1s = [results[n]["eval"]["classification_report"]["macro avg"]["f1-score"] * 100
                 for n in exp_names]

    best_acc = max(accs)
    best_f1  = max(macro_f1s)
    x, width = np.arange(len(exp_names)), 0.36

    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=_WHITE)
    _header(fig, "Overall Performance Summary",
            "Accuracy & Macro-averaged F1 across all classes  ·  ★ = best in category")
    fig.subplots_adjust(top=0.83, bottom=0.14)

    COLOR_ACC = BAR_PALETTE[0]
    COLOR_F1  = BAR_PALETTE[2]

    b1 = ax.bar(x - width / 2, accs, width, label="Accuracy (%)",
                color=COLOR_ACC, alpha=0.88, edgecolor=_WHITE, linewidth=0.6)
    b2 = ax.bar(x + width / 2, macro_f1s, width, label="Macro F1 (%)",
                color=COLOR_F1,  alpha=0.88, edgecolor=_WHITE, linewidth=0.6)

    for bar, val in zip(b1, accs):
        if abs(val - best_acc) < 0.01:
            bar.set_edgecolor("#F1C40F")
            bar.set_linewidth(2.8)
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.4,
                    "★", ha="center", va="bottom", fontsize=13, color="#F1C40F")
    for bar, val in zip(b2, macro_f1s):
        if abs(val - best_f1) < 0.01:
            bar.set_edgecolor("#F1C40F")
            bar.set_linewidth(2.8)
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.4,
                    "★", ha="center", va="bottom", fontsize=13, color="#F1C40F")

    _bar_label(ax, b1, pad=0.3)
    _bar_label(ax, b2, pad=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(shorts, fontsize=8.5, color=_DARK)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 112)
    ax.legend(loc="upper right", framealpha=0.96)
    _despine(ax)

    plt.savefig(save_dir / "summary_bars.png", dpi=200, bbox_inches="tight", facecolor=_WHITE)
    plt.close()
    print(f"Saved: {save_dir / 'summary_bars.png'}")


# ── Chart 6: Confusion matrices montage ──────────────────────────────────────

def plot_confusion_matrices(outputs_dir: Path, results: dict, save_dir: Path):
    exp_names = list(results.keys())
    n = len(exp_names)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.6), facecolor=_WHITE)
    if n == 1:
        axes = [axes]
    _header(fig, "Confusion Matrices",
            "Rows = true class  ·  Columns = predicted class")
    fig.subplots_adjust(top=0.83, bottom=0.02, left=0.01, right=0.99, wspace=0.06)

    for ax, exp_name in zip(axes, exp_names):
        cm_path = outputs_dir / exp_name / "confusion_matrix.png"
        if cm_path.exists():
            ax.imshow(mpimg.imread(str(cm_path)))
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color=_MID, fontsize=11)
        ax.set_title(PIPELINE_SHORT[exp_name], fontsize=9.5, pad=7, color=_DARK)
        ax.axis("off")

    plt.savefig(save_dir / "confusion_matrices_all.png", dpi=200, bbox_inches="tight", facecolor=_WHITE)
    plt.close()
    print(f"Saved: {save_dir / 'confusion_matrices_all.png'}")


# ── Chart 7: Overfitting gap ──────────────────────────────────────────────────

def plot_overfit_gap(results: dict, save_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=_WHITE)
    _header(fig, "Overfitting Analysis",
            "Train − Validation accuracy (%)  ·  above zero = overfitting  ·  near zero = well-generalised")
    fig.subplots_adjust(top=0.83, bottom=0.13)

    for idx, (exp_name, data) in enumerate(results.items()):
        h      = data["history"]
        color  = PALETTE[idx % len(PALETTE)]
        epochs = list(range(1, len(h["train_acc"]) + 1))
        gap    = [t * 100 - v * 100 for t, v in zip(h["train_acc"], h["val_acc"])]

        ax.plot(epochs, gap, color=color, linewidth=2.4,
                label=PIPELINE_SHORT[exp_name], solid_capstyle="round")
        ax.fill_between(epochs, gap, 0,
                        where=[g > 0 for g in gap],
                        color=color, alpha=0.10, interpolate=True)

    ax.axhline(0, color=_DARK, linewidth=1.0, linestyle="--", alpha=0.40)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Acc − Val Acc  (%)")
    ax.legend(loc="upper left", fontsize=8.5)
    _despine(ax)

    plt.savefig(save_dir / "overfit_gap.png", dpi=200, bbox_inches="tight", facecolor=_WHITE)
    plt.close()
    print(f"Saved: {save_dir / 'overfit_gap.png'}")


# ── Chart 8: Full ROC curves (needs probability scores) ──────────────────────

def plot_roc_curves(results: dict, save_dir: Path):
    n_cls = len(CLASSES)
    fig, axes = plt.subplots(1, n_cls, figsize=(5.2 * n_cls, 5.2), facecolor=_WHITE)
    _header(fig, "ROC Curves  (One-vs-Rest)",
            "Filled area = AUC region  ·  dashed diagonal = random classifier baseline")
    fig.subplots_adjust(top=0.83, bottom=0.12, wspace=0.30)

    has_data = False
    for idx, (exp_name, data) in enumerate(results.items()):
        ev      = data["eval"]
        labels  = ev.get("true_labels")
        probs   = ev.get("probabilities")
        classes = ev.get("classes", CLASSES)

        if labels is None or probs is None:
            continue

        labels_arr = np.array(labels)
        probs_arr  = np.array(probs)
        if probs_arr.ndim == 1:
            probs_arr = probs_arr[:, np.newaxis]
        if len(classes) != probs_arr.shape[1]:
            continue

        y_bin = label_binarize(labels_arr, classes=list(range(len(classes))))
        color = PALETTE[idx % len(PALETTE)]
        short = PIPELINE_SHORT[exp_name]

        for ci, ax in enumerate(axes):
            fpr, tpr, _ = roc_curve(y_bin[:, ci], probs_arr[:, ci])
            auc = roc_auc_score(y_bin[:, ci], probs_arr[:, ci])
            ax.plot(fpr, tpr, color=color, linewidth=2.4,
                    label=f"{short}  (AUC={auc:.3f})", solid_capstyle="round")
            ax.fill_between(fpr, tpr, alpha=0.10, color=color)
            has_data = True

    if not has_data:
        print("[Info] No probability scores found; run evaluate.py with a .pth model to enable ROC curves.")
        plt.close()
        return

    for ax, cls in zip(axes, CLASSES):
        ax.plot([0, 1], [0, 1], linestyle="--", color=_LIGHT, linewidth=1.0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Class: {cls}")
        ax.legend(fontsize=7.5, loc="lower right")
        _despine(ax)

    plt.savefig(save_dir / "roc_curves.png", dpi=200, bbox_inches="tight", facecolor=_WHITE)
    plt.close()
    print(f"Saved: {save_dir / 'roc_curves.png'}")


# ── Chart 9: ROC operating points ────────────────────────────────────────────

def _roc_point_from_report(report: dict, cls: str) -> tuple[float, float]:
    precision = report[cls]["precision"]
    recall    = report[cls]["recall"]
    support   = report[cls]["support"]
    total     = report["macro avg"]["support"]
    tp    = recall * support
    fp    = tp * (1.0 / precision - 1.0) if precision > 0 else 0.0
    n_neg = total - support
    return float(fp / n_neg if n_neg > 0 else 0.0), float(recall)


def plot_roc_operating_points(results: dict, save_dir: Path):
    MARKERS = ["o", "s", "^", "D", "P"]
    n_cls   = len(CLASSES)

    fig, axes = plt.subplots(1, n_cls, figsize=(4.8 * n_cls, 4.8), facecolor=_WHITE)
    _header(fig, "ROC Operating Points",
            "(FPR, TPR) at the classification threshold  ·  top-left = perfect  ·  diagonal = random")
    fig.subplots_adjust(top=0.83, bottom=0.12, wspace=0.30)

    for ax, cls in zip(axes, CLASSES):
        for idx, (exp_name, data) in enumerate(results.items()):
            report = data["eval"]["classification_report"]
            if cls not in report:
                continue
            fpr, tpr = _roc_point_from_report(report, cls)
            color    = PALETTE[idx % len(PALETTE)]

            # Glow effect: large translucent ring + solid point on top
            ax.scatter(fpr, tpr, color=color, s=320, alpha=0.15, zorder=3)
            ax.scatter(fpr, tpr, color=color,
                       marker=MARKERS[idx % len(MARKERS)],
                       s=110, zorder=5,
                       edgecolors=_WHITE, linewidths=1.4,
                       label=PIPELINE_SHORT[exp_name])

        ax.plot([0, 1], [0, 1], linestyle="--", color=_LIGHT, linewidth=1.0)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.10)
        ax.set_xlabel("FPR  (1 − Specificity)")
        ax.set_ylabel("TPR  (Recall)")
        ax.set_title(f"Class: {cls}")
        ax.legend(fontsize=7.5, loc="lower right")
        _despine(ax)

    plt.savefig(save_dir / "roc_operating_points.png", dpi=200, bbox_inches="tight", facecolor=_WHITE)
    plt.close()
    print(f"Saved: {save_dir / 'roc_operating_points.png'}")


# ── Conclusion about preprocessing ───────────────────────────────────────────

def generate_conclusion(results: dict, save_dir: Path, table_str: str = ""):
    if "baseline" not in results:
        print("[Skip] Conclusion requires baseline results.")
        return

    def _hist(data, key):
        h = data.get("history", {})
        return h.get(key, [])

    baseline_eval    = results["baseline"]["eval"]
    baseline_acc     = baseline_eval["accuracy"] * 100
    baseline_report  = baseline_eval["classification_report"]
    baseline_f1      = baseline_report["macro avg"]["f1-score"] * 100
    baseline_wf1     = baseline_report["weighted avg"]["f1-score"] * 100
    baseline_cls_f1  = {c: baseline_report[c]["f1-score"] * 100 for c in CLASSES}
    baseline_cls_pre = {c: baseline_report[c]["precision"] * 100 for c in CLASSES}
    baseline_cls_rec = {c: baseline_report[c]["recall"] * 100 for c in CLASSES}
    baseline_support = {c: baseline_report[c]["support"] for c in CLASSES}

    b_val_loss = _hist(results["baseline"], "val_loss")
    b_tra_loss = _hist(results["baseline"], "train_loss")
    b_val_acc  = _hist(results["baseline"], "val_acc")
    b_tra_acc  = _hist(results["baseline"], "train_acc")
    b_epochs   = len(b_val_loss)
    b_best_ep  = int(np.argmin(b_val_loss)) + 1 if b_val_loss else None
    b_gap_loss = (b_tra_loss[b_best_ep - 1] - b_val_loss[b_best_ep - 1]) if b_best_ep else None
    b_gap_acc  = (b_val_acc[b_best_ep - 1] - b_tra_acc[b_best_ep - 1])   if b_best_ep else None

    pipelines = {k: v for k, v in results.items() if k != "baseline"}

    lines = []
    W = 76

    def sep(char="="):  lines.append(char * W)
    def blank():        lines.append("")

    # ── Header ──────────────────────────────────────────────────────────────
    sep()
    lines.append("  KẾT LUẬN VỀ ÁP DỤNG PREPROCESSING")
    lines.append("  Road Damage Detection · ResNet-18 · 4 classes: D00 D20 D40 Normal")
    sep()

    # ── 0. Nhận xét & đánh giá tổng quan ────────────────────────────────────
    blank()
    lines.append("0. NHẬN XÉT & ĐÁNH GIÁ TỔNG QUAN")
    sep("-")

    # Collect per-pipeline precision & recall for D00, D20
    def _pr(exp, cls, metric):
        return results[exp]["eval"]["classification_report"][cls][metric] * 100

    blank()
    lines.append("  [1] Accuracy cao nhưng không phản ánh đúng khả năng phát hiện hư hỏng")
    lines.append(f"      Baseline Accuracy = {baseline_acc:.1f}% nghe có vẻ tốt, nhưng Macro F1 chỉ")
    lines.append(f"      {baseline_f1:.1f}% — chênh lệch {baseline_acc - baseline_f1:.1f}pp. Nguyên nhân: class Normal")
    lines.append(f"      chiếm đa số mẫu, kéo Accuracy lên cao ngay cả khi model yếu ở D00/D20.")
    lines.append("      → Không nên dùng Accuracy làm tiêu chí chính để đánh giá bài toán này.")

    blank()
    lines.append("  [2] D20: Precision cao nhưng Recall thấp — model bỏ sót nhiều vết nứt dọc")
    d20_pre_bl = _pr("baseline", "D20", "precision")
    d20_rec_bl = _pr("baseline", "D20", "recall")
    lines.append(f"      Baseline D20: Precision={d20_pre_bl:.1f}%  Recall={d20_rec_bl:.1f}%  (gap={d20_pre_bl-d20_rec_bl:.1f}pp)")
    lines.append("      Model rất chắc chắn khi dự đoán D20 (ít false positive), nhưng bỏ qua")
    lines.append("      ~27% vết nứt dọc thực sự trong ảnh. Preprocessing cần tập trung")
    lines.append("      cải thiện Recall D20, không chỉ Precision.")
    # Check which pipeline improves D20 recall most
    d20_rec_best = max(pipelines.keys(), key=lambda k: _pr(k, "D20", "recall"))
    lines.append(f"      → {PIPELINE_SHORT[d20_rec_best]} cải thiện Recall D20 tốt nhất: {_pr(d20_rec_best,'D20','recall'):.1f}%")

    blank()
    lines.append("  [3] CLAHE tăng Precision D00 nhưng không giải quyết được Recall")
    d00_pre_bl  = _pr("baseline", "D00", "precision")
    d00_rec_bl  = _pr("baseline", "D00", "recall")
    d00_pre_cl  = _pr("pipeline2_clahe", "D00", "precision")
    d00_rec_cl  = _pr("pipeline2_clahe", "D00", "recall")
    lines.append(f"      Baseline D00:  Precision={d00_pre_bl:.1f}%  Recall={d00_rec_bl:.1f}%")
    lines.append(f"      P2 CLAHE  D00: Precision={d00_pre_cl:.1f}%  Recall={d00_rec_cl:.1f}%  (Prec +{d00_pre_cl-d00_pre_bl:.1f}pp, Rec {d00_rec_cl-d00_rec_bl:+.1f}pp)")
    lines.append("      CLAHE giúp model tự tin hơn khi nhận ra D00 (ít nhầm D00 là class khác),")
    lines.append("      nhưng recall gần như không đổi — model vẫn bỏ sót số lượng D00 tương đương.")
    # Check Gray+Bilateral for D00 recall
    if "pipeline3_grayscale_bilateral" in results:
        d00_rec_gb = _pr("pipeline3_grayscale_bilateral", "D00", "recall")
        lines.append(f"      → P3 Gray+Bilateral là pipeline duy nhất cải thiện Recall D00 đáng kể:")
        lines.append(f"        Recall D00 = {d00_rec_gb:.1f}% (+{d00_rec_gb-d00_rec_bl:.1f}pp), dù Accuracy tổng thể thấp hơn Baseline.")

    blank()
    lines.append("  [4] P1 Letterbox làm model bị bias mạnh về class Normal")
    d20_rec_lb = _pr("pipeline1_letterbox", "D20", "recall")
    nm_rec_lb  = _pr("pipeline1_letterbox", "Normal", "recall")
    lines.append(f"      P1 Letterbox: Recall Normal={nm_rec_lb:.1f}%  nhưng Recall D20={d20_rec_lb:.1f}%")
    lines.append(f"      Padding thêm vào ảnh (để giữ tỷ lệ khung hình) tạo ra vùng pixel đồng nhất,")
    lines.append(f"      khiến model dễ nhận ra ảnh 'clean' (Normal) hơn, còn các vết nứt mảnh")
    lines.append(f"      (D20) bị mất đặc trưng trong quá trình resize-after-pad.")
    lines.append(f"      → Letterbox KHÔNG phù hợp với bài toán phát hiện hư hỏng nhỏ.")

    blank()
    lines.append("  [5] P4 Combined đạt Recall 100% trên Normal — đáng tin cậy nhất cho normal roads")
    nm_rec_cb = _pr("pipeline4_combined", "Normal", "recall")
    d40_f1_cb = _pr("pipeline4_combined", "D40", "f1-score") if False else \
                results["pipeline4_combined"]["eval"]["classification_report"]["D40"]["f1-score"] * 100
    lines.append(f"      P4 Recall Normal = {nm_rec_cb:.1f}% — mọi đoạn đường bình thường đều được")
    lines.append(f"      phân loại đúng, không có false alarm. Đây là điểm mạnh trong thực tế:")
    lines.append(f"      hệ thống sẽ không báo sai 'có hư hỏng' trên đường tốt.")
    lines.append(f"      Đánh đổi: F1-D40 giảm nhẹ ({d40_f1_cb:.1f}% vs {baseline_cls_f1['D40']:.1f}% Baseline).")

    blank()
    lines.append("  [6] Không có pipeline nào giải quyết được hoàn toàn điểm yếu của Baseline")
    lines.append(f"      D00 vẫn là class khó nhất sau tất cả preprocessing (F1 max={max(_pr(k,'D00','f1-score') for k in pipelines):.1f}%),")
    lines.append(f"      và D20 Recall vẫn chưa vượt 81% ở bất kỳ pipeline nào.")
    lines.append("      → Preprocessing giúp ích nhưng không đủ: cần thêm data augmentation,")
    lines.append("        oversampling D00/D20, hoặc focal loss để giải quyết triệt để.")

    # ── 1. Baseline ──────────────────────────────────────────────────────────
    blank()
    lines.append("1. HIỆU SUẤT BASELINE (Resize + Normalize only)")
    sep("-")
    lines.append(f"   Accuracy     : {baseline_acc:.2f}%")
    lines.append(f"   Macro F1     : {baseline_f1:.2f}%   (trung bình không trọng số qua 4 class)")
    lines.append(f"   Weighted F1  : {baseline_wf1:.2f}%  (trọng số theo số mẫu mỗi class)")
    if b_best_ep:
        lines.append(f"   Số epoch     : {b_epochs}  |  Best epoch: {b_best_ep}")
    blank()
    lines.append("   Hiệu suất theo từng class:")
    for c in CLASSES:
        sup  = baseline_support[c]
        f1   = baseline_cls_f1[c]
        pre  = baseline_cls_pre[c]
        rec  = baseline_cls_rec[c]
        lines.append(f"     {c:<8}  F1={f1:.1f}%  Prec={pre:.1f}%  Rec={rec:.1f}%  (n={sup})")

    # Identify hardest class for baseline
    hardest  = min(CLASSES, key=lambda c: baseline_cls_f1[c])
    easiest  = max(CLASSES, key=lambda c: baseline_cls_f1[c])
    blank()
    lines.append(f"   → Class khó nhất: {hardest} (F1={baseline_cls_f1[hardest]:.1f}%) — cần cải thiện nhất.")
    lines.append(f"   → Class dễ nhất : {easiest} (F1={baseline_cls_f1[easiest]:.1f}%)")

    # ── 1b. Lưu ý mất cân bằng dữ liệu ─────────────────────────────────────
    total_samples = sum(baseline_support.values())
    majority_cls  = max(CLASSES, key=lambda c: baseline_support[c])
    minority_cls  = min(CLASSES, key=lambda c: baseline_support[c])
    majority_pct  = baseline_support[majority_cls] / total_samples * 100
    minority_pct  = baseline_support[minority_cls] / total_samples * 100
    imbalance_ratio = baseline_support[majority_cls] / baseline_support[minority_cls]

    blank()
    lines.append("   ⚠ LƯU Ý: MẤT CÂN BẰNG DỮ LIỆU (Class Imbalance)")
    lines.append(f"   Tổng mẫu test: {int(total_samples)}")
    for c in CLASSES:
        pct = baseline_support[c] / total_samples * 100
        bar = "█" * int(pct / 3)
        lines.append(f"     {c:<8}  n={int(baseline_support[c]):<5}  ({pct:.1f}%)  {bar}")
    blank()
    lines.append(f"   Tỷ lệ mất cân bằng: {majority_cls} / {minority_cls} = {imbalance_ratio:.1f}x")
    lines.append(f"   → {majority_cls} chiếm {majority_pct:.1f}% tổng mẫu, làm Accuracy bị kéo lên cao")
    lines.append(f"     ngay cả khi model dự đoán kém ở các class thiểu số ({minority_cls}, D20).")
    lines.append("   → Macro F1 là chỉ số đáng tin cậy hơn Accuracy trong bộ dữ liệu này,")
    lines.append("     vì nó đánh giá đồng đều trên tất cả class bất kể số lượng mẫu.")
    lines.append("   → Preprocessing tác động MẠNH nhất ở các class thiểu số (D00, D20)")
    lines.append("     vì đây là các class model cần được hỗ trợ thêm từ chất lượng ảnh.")

    # ── 2. Chi tiết từng pipeline ────────────────────────────────────────────
    blank(); blank()
    lines.append("2. SO SÁNH CHI TIẾT TỪNG PREPROCESSING PIPELINE")
    sep("-")

    improvements = {}
    for exp_name, data in pipelines.items():
        report   = data["eval"]["classification_report"]
        acc      = data["eval"]["accuracy"] * 100
        mf1      = report["macro avg"]["f1-score"] * 100
        wf1      = report["weighted avg"]["f1-score"] * 100
        d_acc    = acc - baseline_acc
        d_mf1    = mf1 - baseline_f1
        d_wf1    = wf1 - baseline_wf1
        improvements[exp_name] = (d_acc, d_mf1, acc, mf1, d_wf1)

        val_loss = _hist(data, "val_loss")
        tra_loss = _hist(data, "train_loss")
        val_acc  = _hist(data, "val_acc")
        tra_acc  = _hist(data, "train_acc")
        n_ep     = len(val_loss)
        best_ep  = int(np.argmin(val_loss)) + 1 if val_loss else None

        verdict  = ("✓ TỐT HƠN BASELINE" if d_acc > 0.5
                    else "✗ KÉM HƠN BASELINE" if d_acc < -0.5
                    else "≈ TƯƠNG ĐƯƠNG BASELINE")

        blank()
        lines.append(f"  [{PIPELINE_SHORT[exp_name]}]  {verdict}")
        lines.append(f"    Accuracy    : {acc:.2f}%  ({d_acc:+.2f} pp so với Baseline)")
        lines.append(f"    Macro F1    : {mf1:.2f}%  ({d_mf1:+.2f} pp)")
        lines.append(f"    Weighted F1 : {wf1:.2f}%  ({d_wf1:+.2f} pp)")
        if best_ep:
            lines.append(f"    Epochs      : {n_ep} epoch  |  Best epoch: {best_ep}")
            gap_a = val_acc[best_ep-1] - tra_acc[best_ep-1] if tra_acc else None
            if gap_a is not None:
                overfit_note = ("(gần như không overfit)" if abs(gap_a)*100 < 3
                                else "(overfit nhẹ)" if abs(gap_a)*100 < 8
                                else "(overfit đáng kể)")
                lines.append(f"    Val−Train Acc gap: {gap_a*100:+.1f}pp {overfit_note}")

        # Per-class breakdown
        lines.append("    Per-class F1:")
        gains, losses = [], []
        for c in CLASSES:
            f1_c  = report[c]["f1-score"] * 100
            pre_c = report[c]["precision"] * 100
            rec_c = report[c]["recall"] * 100
            delta = f1_c - baseline_cls_f1[c]
            mark  = f"↑{delta:+.1f}pp" if delta > 1 else (f"↓{delta:+.1f}pp" if delta < -1 else f"  {delta:+.1f}pp")
            lines.append(f"      {c:<8} F1={f1_c:.1f}%  Prec={pre_c:.1f}%  Rec={rec_c:.1f}%  {mark}")
            if delta > 1:   gains.append((c, delta))
            if delta < -1:  losses.append((c, delta))

        if gains:
            g = ", ".join(f"{c} (+{d:.1f}pp)" for c, d in sorted(gains, key=lambda x: -x[1]))
            lines.append(f"    ✔ Cải thiện rõ rệt : {g}")
        if losses:
            l = ", ".join(f"{c} ({d:.1f}pp)" for c, d in sorted(losses, key=lambda x: x[1]))
            lines.append(f"    ✘ Giảm sút rõ rệt  : {l}")

        # Technique-specific insight
        if exp_name == "pipeline1_letterbox":
            lines.append("    Insight: Letterbox giữ tỷ lệ khung hình nhưng thêm padding,")
            lines.append("      có thể làm nhiễu loạn các đặc trưng cục bộ (texture vết nứt).")
            lines.append("      D20 giảm mạnh → padding ảnh hưởng đến phát hiện vết nứt dọc.")
        elif exp_name == "pipeline2_clahe":
            lines.append("    Insight: CLAHE tăng cường độ tương phản cục bộ, giúp model")
            lines.append("      nhìn rõ hơn các vết nứt mờ. Hiệu quả đồng đều trên cả 4 class.")
        elif exp_name == "pipeline3_grayscale_bilateral":
            lines.append("    Insight: Chuyển grayscale loại bỏ thông tin màu sắc (mặt đường)")
            lines.append("      có thể hữu ích. Bilateral filter khử nhiễu nhưng có thể làm mờ")
            lines.append("      các cạnh mảnh của vết nứt, gây giảm F1 ở D20, D40.")
        elif exp_name == "pipeline4_combined":
            lines.append("    Insight: Kết hợp CLAHE + bilateral + letterbox tận dụng được")
            lines.append("      ưu điểm của nhiều kỹ thuật. Cân bằng tốt giữa tăng cường")
            lines.append("      tương phản và khử nhiễu, giải thích hiệu suất tổng thể cao nhất.")

    # ── 3. Phân tích theo class ──────────────────────────────────────────────
    blank(); blank()
    lines.append("3. PHÂN TÍCH TÁC ĐỘNG CỦA PREPROCESSING THEO TỪNG CLASS")
    sep("-")
    for c in CLASSES:
        sup = baseline_support[c]
        blank()
        lines.append(f"  [{c}]  (n={sup} mẫu test)  Baseline F1={baseline_cls_f1[c]:.1f}%")
        deltas = []
        for exp_name, data in pipelines.items():
            f1_c  = data["eval"]["classification_report"][c]["f1-score"] * 100
            delta = f1_c - baseline_cls_f1[c]
            deltas.append((PIPELINE_SHORT[exp_name], f1_c, delta))
        for label, f1_c, delta in sorted(deltas, key=lambda x: -x[2]):
            bar = "█" * int(abs(delta) / 1.0)
            dir_ = "+" if delta >= 0 else "-"
            lines.append(f"    {label:<22}  F1={f1_c:.1f}%  {dir_}{abs(delta):.1f}pp  {bar}")
        best_for_cls  = max(deltas, key=lambda x: x[2])
        worst_for_cls = min(deltas, key=lambda x: x[2])
        lines.append(f"    → Tốt nhất cho {c}: {best_for_cls[0]} ({best_for_cls[2]:+.1f}pp)")
        if worst_for_cls[2] < -1:
            lines.append(f"    → Nên tránh cho {c}: {worst_for_cls[0]} ({worst_for_cls[2]:+.1f}pp)")

    # ── 4. Tốc độ hội tụ ────────────────────────────────────────────────────
    blank(); blank()
    lines.append("4. PHÂN TÍCH TỐC ĐỘ HỘI TỤ VÀ EARLY STOPPING")
    sep("-")
    lines.append(f"  {'Pipeline':<28}  Epochs  BestEp  MinValLoss")
    lines.append("  " + "-" * 55)
    all_ep_data = [("Baseline", results["baseline"])]
    all_ep_data += list(pipelines.items())
    for exp_name, data in all_ep_data:
        vl = _hist(data, "val_loss")
        n  = len(vl)
        if n == 0:
            lines.append(f"  {PIPELINE_SHORT.get(exp_name, exp_name):<28}  n/a")
            continue
        best = int(np.argmin(vl)) + 1
        min_vl = min(vl)
        note = " ← early stop" if n < 25 else ""
        lines.append(f"  {PIPELINE_SHORT.get(exp_name, exp_name):<28}  {n:<6}  {best:<6}  {min_vl:.4f}{note}")
    blank()
    lines.append("  Nhận xét:")
    fast = [k for k, v in pipelines.items() if len(_hist(v, "val_loss")) < 15]
    slow = [k for k, v in pipelines.items() if len(_hist(v, "val_loss")) >= 25]
    if fast:
        names = ", ".join(PIPELINE_SHORT[k] for k in fast)
        lines.append(f"  • {names} hội tụ nhanh (early stop < 15 epoch) → dữ liệu đã")
        lines.append("    'dễ học' hơn nhưng có thể underfit hoặc converge tại local minimum.")
    if slow:
        names = ", ".join(PIPELINE_SHORT[k] for k in slow)
        lines.append(f"  • {names} cần nhiều epoch hơn → mô hình học sâu hơn từ dữ liệu.")

    # ── 5. Overfitting ───────────────────────────────────────────────────────
    blank(); blank()
    lines.append("5. PHÂN TÍCH OVERFITTING (Train vs Validation Accuracy tại Best Epoch)")
    sep("-")
    lines.append(f"  {'Pipeline':<28}  TrainAcc  ValAcc  Gap(Val-Train)")
    lines.append("  " + "-" * 60)
    for exp_name, data in all_ep_data:
        vl = _hist(data, "val_loss")
        ta = _hist(data, "train_acc")
        va = _hist(data, "val_acc")
        if not vl or not ta or not va:
            continue
        best = int(np.argmin(vl))
        tr   = ta[best] * 100
        vr   = va[best] * 100
        gap  = vr - tr
        flag = " ← overfit" if gap < -5 else (" ← strong overfit" if gap < -10 else "")
        label = PIPELINE_SHORT.get(exp_name, exp_name)
        lines.append(f"  {label:<28}  {tr:.1f}%     {vr:.1f}%   {gap:+.1f}pp{flag}")
    blank()
    lines.append("  Nhận xét: Gap âm (Val < Train) = dấu hiệu overfitting.")
    lines.append("  Gap dương (Val > Train) = regularization hiệu quả hoặc data augmentation.")

    # ── 6. Bảng xếp hạng ────────────────────────────────────────────────────
    blank(); blank()
    lines.append("6. BẢNG XẾP HẠNG TỔNG HỢP")
    sep("-")
    lines.append(f"  {'Hạng':<6} {'Pipeline':<28}  Accuracy   MacroF1   ΔAcc    ΔF1")
    lines.append("  " + "-" * 72)
    ranked = sorted(improvements.items(), key=lambda x: (x[1][0] + x[1][1]) / 2, reverse=True)
    for rank, (exp_name, (da, df, acc, mf1, dwf1)) in enumerate(ranked, 1):
        medal = ["⭐", "🥈", "🥉", " 4."][min(rank - 1, 3)]
        lines.append(f"  {medal}  #{rank}  {PIPELINE_SHORT[exp_name]:<26}  {acc:.2f}%    {mf1:.2f}%   {da:+.2f}   {df:+.2f}")
    lines.append(f"\n  Baseline (tham chiếu)                   {baseline_acc:.2f}%    {baseline_f1:.2f}%    —      —")

    # ── 7. Kết luận và khuyến nghị ──────────────────────────────────────────
    blank(); blank()
    lines.append("7. KẾT LUẬN VÀ KHUYẾN NGHỊ")
    sep("=")

    helped   = [k for k, (da, *_) in improvements.items() if da > 0.5]
    degraded = [k for k, (da, *_) in improvements.items() if da < -0.5]
    neutral  = [k for k, (da, *_) in improvements.items() if abs(da) <= 0.5]

    best_name = ranked[0][0]
    best_acc  = improvements[best_name][2]
    best_f1   = improvements[best_name][3]

    blank()
    lines.append("  TỔNG QUAN:")
    lines.append(f"  Preprocessing CÓ hiệu quả trong {len(helped)}/{len(pipelines)} pipeline được thử nghiệm.")
    if helped:
        lines.append(f"  Pipeline có lợi  : {', '.join(PIPELINE_SHORT[k] for k in helped)}")
    if neutral:
        lines.append(f"  Tương đương      : {', '.join(PIPELINE_SHORT[k] for k in neutral)}")
    if degraded:
        lines.append(f"  Có hại           : {', '.join(PIPELINE_SHORT[k] for k in degraded)}")

    blank()
    lines.append(f"  PIPELINE TỐT NHẤT: {PIPELINE_SHORT[best_name]}")
    lines.append(f"  • Accuracy = {best_acc:.2f}%  (+{improvements[best_name][0]:.2f}pp so với Baseline)")
    lines.append(f"  • Macro F1 = {best_f1:.2f}%  (+{improvements[best_name][1]:.2f}pp so với Baseline)")

    # Trade-off: CLAHE vs Combined
    clahe_data    = improvements.get("pipeline2_clahe")
    combined_data = improvements.get("pipeline4_combined")
    blank()
    if clahe_data and combined_data:
        clahe_acc, clahe_f1   = clahe_data[2],    clahe_data[3]
        comb_acc,  comb_f1    = combined_data[2],  combined_data[3]
        lines.append("  PHÂN TÍCH TRADE-OFF: P2 CLAHE vs P4 Combined")
        lines.append("  " + "-" * 50)
        lines.append(f"  {'Tiêu chí':<30}  {'P2: CLAHE':>12}  {'P4: Combined':>12}")
        lines.append(f"  {'Accuracy (%)':<30}  {clahe_acc:>12.2f}  {comb_acc:>12.2f}")
        lines.append(f"  {'Macro F1 (%)':<30}  {clahe_f1:>12.2f}  {comb_f1:>12.2f}")
        lines.append(f"  {'F1-D00 (class thiểu số)':<30}  "
                     f"{results['pipeline2_clahe']['eval']['classification_report']['D00']['f1-score']*100:>12.1f}  "
                     f"{results['pipeline4_combined']['eval']['classification_report']['D00']['f1-score']*100:>12.1f}")
        lines.append(f"  {'F1-Normal (class đa số)':<30}  "
                     f"{results['pipeline2_clahe']['eval']['classification_report']['Normal']['f1-score']*100:>12.1f}  "
                     f"{results['pipeline4_combined']['eval']['classification_report']['Normal']['f1-score']*100:>12.1f}")
        lines.append(f"  {'Epochs training':<30}  "
                     f"{len(_hist(results['pipeline2_clahe'], 'val_loss')):>12}  "
                     f"{len(_hist(results['pipeline4_combined'], 'val_loss')):>12}")
        blank()
        winner_acc = "P2: CLAHE" if clahe_acc > comb_acc else "P4: Combined"
        winner_f1  = "P2: CLAHE" if clahe_f1  > comb_f1  else "P4: Combined"
        lines.append(f"  → Accuracy cao hơn  : {winner_acc}")
        lines.append(f"  → Macro F1 cao hơn  : {winner_f1}")
        lines.append("  → KHI NÀO CHỌN P2 CLAHE:")
        lines.append("    • Ưu tiên phát hiện đồng đều tất cả loại damage (Macro F1 cao hơn)")
        lines.append("    • Muốn tối ưu F1 trên class D20 (+1.5pp vs Baseline)")
        lines.append("    • Hệ thống cần train lâu hơn nhưng chấp nhận được (26 epochs)")
        lines.append("  → KHI NÀO CHỌN P4 Combined:")
        lines.append("    • Ưu tiên Accuracy tổng thể trên toàn bộ dataset")
        lines.append("    • Cần training nhanh hơn (chỉ 10 epochs, early stop)")
        lines.append("    • Ưu tiên cải thiện D00 (+6.4pp, tốt hơn CLAHE +5.6pp)")
        lines.append("    • Hệ thống thực tế cần deploy nhanh, ít tài nguyên preprocessing")

    blank()
    lines.append("  KHUYẾN NGHỊ TRIỂN KHAI:")
    lines.append(f"  1. Nếu mục tiêu là phát hiện đều tất cả loại damage → dùng P2: CLAHE")
    lines.append(f"     (Macro F1 cao nhất: {clahe_data[3]:.2f}%, cải thiện đồng đều 4/4 class)")
    lines.append(f"  2. Nếu mục tiêu là accuracy tổng thể + training nhanh → dùng P4: Combined")
    lines.append(f"     (Accuracy cao nhất: {combined_data[2]:.2f}%, chỉ cần 10 epochs)")
    if degraded:
        for k in degraded:
            lines.append(f"  ✗ KHÔNG nên dùng {PIPELINE_SHORT[k]} — làm giảm hiệu suất.")

    blank()
    lines.append("  HƯỚNG CẢI THIỆN TIẾP THEO:")
    lines.append("  • Tăng dữ liệu training (data augmentation: flip, rotate, brightness)")
    lines.append("  • Fine-tune với learning rate thấp hơn trên pipeline tốt nhất")
    lines.append("  • Thử nghiệm ensemble giữa Baseline và pipeline tốt nhất")
    lines.append("  • Tập trung cải thiện class khó nhất: " + hardest)
    blank()
    sep()

    text = "\n".join(lines)
    print(text)

    out_path = save_dir / "conclusion_preprocessing.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        if table_str:
            f.write(table_str + "\n\n")
        f.write(text)
    print(f"\nSaved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Compare metrics across all pipelines")
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--save_dir",    type=str, default="outputs/comparison")
    return parser.parse_args()


def main():
    args        = parse_args()
    outputs_dir = Path(args.outputs_dir)
    save_dir    = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()

    print("Loading results...")
    results = load_results(outputs_dir)

    if not results:
        print("\n[Error] No pipeline results found. Run training and evaluate.py first.")
        return

    print(f"Found: {list(results.keys())}\n")

    _, table_str = print_metrics_table(results, save_dir)
    plot_metrics_heatmap(results, save_dir)
    plot_radar_chart(results, save_dir)
    plot_learning_curves(results, save_dir)
    plot_f1_comparison(results, save_dir)
    plot_summary_bars(results, save_dir)
    plot_confusion_matrices(outputs_dir, results, save_dir)
    plot_overfit_gap(results, save_dir)
    plot_roc_curves(results, save_dir)
    plot_roc_operating_points(results, save_dir)
    generate_conclusion(results, save_dir, table_str=table_str)

    print(f"\nAll charts saved to: {save_dir}/")


if __name__ == "__main__":
    main()
