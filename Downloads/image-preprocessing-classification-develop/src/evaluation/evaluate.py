import argparse
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.preprocessing.transforms import get_transforms
from src.training.dataset import RDDDataset
from src.models.baseline_cnn import get_model

import os

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN model on the test set")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model weight file (.pth)"
    )
    return parser.parse_args()

def plot_confusion_matrix(cm, classes, output_path, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots and saves the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Labeling the matrix cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Create parent folder if not exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        
    return np.array(all_labels), np.array(all_preds)

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    print(f"--- Evaluating model for: {config['exp_name']} ---")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "cuda") == "cuda" else "cpu")
    
    # Load dataset
    test_transform = get_transforms(config)
    test_dataset = RDDDataset(config["data_dir"], "test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    
    # Build model & load weights
    model = get_model(
        model_name=config["model_name"],
        pretrained=config["pretrained"],
        num_classes=config["num_classes"]
    )
    
    # Load checkpoint
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint file not found at: {args.model_path}")
        return
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    print("Model weights loaded successfully.")
    
    # Run evaluation
    true_labels, pred_labels = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    acc = accuracy_score(true_labels, pred_labels)
    report_dict = classification_report(
        true_labels, 
        pred_labels, 
        target_names=test_dataset.classes, 
        output_dict=True
    )
    report_str = classification_report(
        true_labels, 
        pred_labels, 
        target_names=test_dataset.classes
    )
    
    print("\n--- Test Set Classification Report ---")
    print(report_str)
    print(f"Overall Accuracy: {acc*100:.2f}%")
    
    # Save results
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    eval_results = {
        "accuracy": acc,
        "classification_report": report_dict
    }
    
    with open(output_dir / "test_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
        
    # Generate and save confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plot_confusion_matrix(
        cm, 
        classes=test_dataset.classes, 
        output_path=output_dir / "confusion_matrix.png"
    )
    
    print(f"\nEvaluation completed. Metrics and confusion matrix saved to {output_dir}")

if __name__ == "__main__":
    main()
