import argparse
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing.transforms import get_transforms
from src.training.dataset import RDDDataset
from src.models.baseline_cnn import get_model


PIPELINE_LABELS = {
    "baseline":                      "Baseline (Resize only)",
    "pipeline1_letterbox":           "Pipeline 1 (Letterbox)",
    "pipeline2_clahe":               "Pipeline 2 (CLAHE)",
    "pipeline3_grayscale_bilateral": "Pipeline 3 (Gray+Bilateral)",
    "pipeline4_combined":            "Pipeline 4 (Combined)",
}

PIPELINE_CONFIGS = {
    "baseline":                      "configs/baseline.yaml",
    "pipeline1_letterbox":           "configs/pipeline1_letterbox.yaml",
    "pipeline2_clahe":               "configs/pipeline2_clahe.yaml",
    "pipeline3_grayscale_bilateral": "configs/pipeline3_grayscale_bilateral.yaml",
    "pipeline4_combined":            "configs/pipeline4_combined.yaml",
}

CLASSES = ["D00", "D20", "D40", "Normal"]
COLORS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# ── Inference ────────────────────────────────────────────────────────────────

def collect_errors(model, dataset, device, max_errors: int = 300) -> list:
    """Run inference on test set and return misclassified samples."""
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    errors = []
    global_idx = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            preds   = outputs.argmax(dim=1).cpu()

            for i in range(len(labels)):
                true = labels[i].item()
                pred = preds[i].item()
                if pred != true:
                    errors.append({
                        "img_path": dataset.image_paths[global_idx + i],
                        "true":     true,
                        "pred":     pred,
                    })
                    if len(errors) >= max_errors:
                        return errors
            global_idx += len(labels)

    return errors


# ── Chart 1: Grid of misclassified image samples ─────────────────────────────

def plot_error_samples(errors: list, exp_name: str, save_dir: Path, n_show: int = 20):
    n      = min(len(errors), n_show)
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.2 * n_rows))
    fig.suptitle(
        f"Mẫu phân loại sai — {PIPELINE_LABELS.get(exp_name, exp_name)}\n"
        f"({len(errors)} lỗi tổng cộng, hiển thị {n})",
        fontsize=11, fontweight="bold"
    )
    axes = np.array(axes).flatten()

    for i, err in enumerate(errors[:n]):
        img = cv2.imread(err["img_path"])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112))
        else:
            img = np.zeros((112, 112, 3), dtype=np.uint8)

        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {CLASSES[err['true']]}\nPred: {CLASSES[err['pred']]}",
            fontsize=8, color="red"
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = save_dir / f"error_samples_{exp_name}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 2: Error pattern heatmap (true → predicted, errors only) ───────────

def plot_error_heatmap(all_errors: dict, save_dir: Path):
    n           = len(all_errors)
    n_cls       = len(CLASSES)
    fig, axes   = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Error Pattern Heatmap — True vs Predicted (chỉ mẫu sai)",
        fontsize=12, fontweight="bold"
    )

    for ax, (exp_name, errors) in zip(axes, all_errors.items()):
        matrix = np.zeros((n_cls, n_cls), dtype=int)
        for err in errors:
            matrix[err["true"]][err["pred"]] += 1
        np.fill_diagonal(matrix, 0)   # chỉ hiện lỗi, không hiện đúng

        im = ax.imshow(matrix, cmap="Reds")
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        ax.set_xticklabels(CLASSES, rotation=45, fontsize=8)
        ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(PIPELINE_LABELS.get(exp_name, exp_name), fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

        for i in range(n_cls):
            for j in range(n_cls):
                if matrix[i, j] > 0:
                    ax.text(j, i, str(matrix[i, j]),
                            ha="center", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = save_dir / "error_heatmap_all.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Chart 3: Error rate per class, per pipeline ──────────────────────────────

def plot_error_rate_by_class(all_errors: dict, datasets: dict, save_dir: Path):
    """Bar chart: % bị phân loại sai trong mỗi class, mỗi pipeline."""
    exp_names   = list(all_errors.keys())
    n_pipelines = len(exp_names)
    n_cls       = len(CLASSES)
    x           = np.arange(n_cls)
    width       = 0.8 / n_pipelines

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Tỷ lệ lỗi (%) theo từng lớp — So sánh Pipeline",
                 fontsize=12, fontweight="bold")

    for i, exp_name in enumerate(exp_names):
        errors  = all_errors[exp_name]
        dataset = datasets[exp_name]

        # Count total samples per class
        total_per_class = defaultdict(int)
        for lbl in dataset.labels:
            total_per_class[lbl] += 1

        # Count errors per class
        error_per_class = defaultdict(int)
        for err in errors:
            error_per_class[err["true"]] += 1

        error_rates = [
            error_per_class[c] / total_per_class[c] * 100 if total_per_class[c] > 0 else 0
            for c in range(n_cls)
        ]

        offset = (i - n_pipelines / 2 + 0.5) * width
        label  = PIPELINE_LABELS.get(exp_name, exp_name)
        bars   = ax.bar(x + offset, error_rates, width,
                        label=label, color=COLORS[i % len(COLORS)], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.set_ylabel("Tỷ lệ lỗi (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = save_dir / "error_rate_by_class.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Text summary ──────────────────────────────────────────────────────────────

def print_error_summary(all_errors: dict, datasets: dict):
    print("\n" + "=" * 70)
    print("  ERROR ANALYSIS SUMMARY")
    print("=" * 70)

    for exp_name, errors in all_errors.items():
        label   = PIPELINE_LABELS.get(exp_name, exp_name)
        dataset = datasets[exp_name]
        total   = len(dataset)
        n_err   = len(errors)
        print(f"\n[{label}]")
        print(f"  Tổng sai: {n_err}/{total} mẫu  ({n_err/total*100:.1f}%)")

        # Top confusions
        confusion_count = defaultdict(int)
        for err in errors:
            key = f"{CLASSES[err['true']]} → {CLASSES[err['pred']]}"
            confusion_count[key] += 1

        top = sorted(confusion_count.items(), key=lambda x: -x[1])[:5]
        print("  Nhầm lẫn phổ biến nhất:")
        for pair, count in top:
            print(f"    {pair}: {count} lần")

    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Error analysis across all pipelines")
    parser.add_argument("--outputs_dir",  type=str, default="outputs",
                        help="Root outputs directory (default: outputs/)")
    parser.add_argument("--save_dir",     type=str, default="outputs/comparison",
                        help="Directory to save error analysis charts")
    parser.add_argument("--max_errors",   type=int, default=300,
                        help="Max misclassified samples collected per pipeline")
    parser.add_argument("--show_samples", type=int, default=20,
                        help="Number of sample images to show per pipeline grid")
    return parser.parse_args()


def main():
    args        = parse_args()
    outputs_dir = Path(args.outputs_dir)
    save_dir    = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_errors = {}
    all_datasets = {}

    for exp_name, config_path_str in PIPELINE_CONFIGS.items():
        config_path = Path(config_path_str)
        model_path  = outputs_dir / exp_name / "best_model.pth"

        if not config_path.exists():
            print(f"\n[Skip] {exp_name}: config not found ({config_path})")
            continue
        if not model_path.exists():
            print(f"\n[Skip] {exp_name}: model not found ({model_path})")
            continue

        print(f"\nAnalyzing: {exp_name}...")
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        transform = get_transforms(config)

        try:
            dataset = RDDDataset(config["data_dir"], "test", transform=transform)
        except FileNotFoundError as e:
            print(f"  [Error] {e}")
            continue

        model = get_model(
            model_name=config["model_name"],
            pretrained=False,
            num_classes=config["num_classes"]
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)

        errors = collect_errors(model, dataset, device, max_errors=args.max_errors)
        all_errors[exp_name]   = errors
        all_datasets[exp_name] = dataset

        total = len(dataset)
        print(f"  Sai: {len(errors)}/{total} mẫu ({len(errors)/total*100:.1f}%)")
        plot_error_samples(errors, exp_name, save_dir, n_show=args.show_samples)

    if not all_errors:
        print("\n[Error] No pipeline results found. Train and evaluate first.")
        return

    plot_error_heatmap(all_errors, save_dir)
    plot_error_rate_by_class(all_errors, all_datasets, save_dir)
    print_error_summary(all_errors, all_datasets)

    print(f"\nError analysis done. All charts saved to: {save_dir}/")


if __name__ == "__main__":
    main()
