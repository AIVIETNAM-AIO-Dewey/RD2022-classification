import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from src.preprocessing.transforms import get_transforms
from src.training.dataset import RDDDataset
from src.models.baseline_cnn import get_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN model on RDD2022 dataset")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g., configs/baseline.yaml)"
    )
    return parser.parse_args()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"--- Starting Experiment: {config['exp_name']} ---")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "cuda") == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get transforms
    train_transform = get_transforms(config, is_train=True)
    val_transform = get_transforms(config, is_train=False)
    
    # Setup datasets & dataloaders
    data_dir = config["data_dir"]
    train_dataset = RDDDataset(data_dir, "train", transform=train_transform)
    val_dataset = RDDDataset(data_dir, "val", transform=val_transform)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Train or Val dataset is empty. Please run process_patches.py first.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    
    # Build model
    model = get_model(
        model_name=config["model_name"],
        pretrained=config["pretrained"],
        num_classes=config["num_classes"]
    )
    model = model.to(device)
    
    # Define Loss and Optimizer
    if config.get("use_class_weights", False):
        labels = train_dataset.labels
        class_counts = [labels.count(i) for i in range(config["num_classes"])]
        print(f"Class counts in training set: {class_counts}")
        
        # Calculate weights: total_samples / (num_classes * class_count)
        total_samples = len(labels)
        class_weights = [total_samples / (config["num_classes"] * count) if count > 0 else 1.0 for count in class_counts]
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        print(f"Calculated class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # History logs
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_acc = 0.0
    
    # Train loop
    epochs = config["epochs"]
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
              
        # Checkpoint: Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"  => Saved new best model checkpoint (Val Acc: {best_val_acc*100:.2f}%)")
            
    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    print(f"--- Training Completed. Results saved to {output_dir} ---")

if __name__ == "__main__":
    main()
