import os
import argparse
import cv2
import yaml
import shutil
from pathlib import Path
from tqdm import tqdm

import sys
# Add project root to sys.path in case it is run from subfolders
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.preprocessing.transforms import get_transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and save the dataset offline based on a config file")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"\n--- Offline Preprocessing: {config['exp_name']} ---")
    
    data_dir_root = Path(config["data_dir"])  # e.g., data/processed
    cropped_root = data_dir_root / "cropped"
    mode = config.get("preprocessing_mode", "baseline")
    preprocessed_root = data_dir_root / mode
    
    # 1. Dynamic folder reorganization:
    # If splits train/val/test are directly under data/processed/, move them to data/processed/cropped/
    if not cropped_root.exists():
        has_splits = any((data_dir_root / s).exists() for s in ["train", "val", "test"])
        if has_splits:
            print(f"Reorganizing raw cropped patches: moving train/val/test to {cropped_root}")
            cropped_root.mkdir(parents=True, exist_ok=True)
            for s in ["train", "val", "test"]:
                src_split = data_dir_root / s
                if src_split.exists():
                    shutil.move(str(src_split), str(cropped_root / s))
                    
    # 2. Determine source root folder
    src_root = cropped_root if cropped_root.exists() else data_dir_root
    
    print(f"Reading raw cropped images from: {src_root}")
    print(f"Saving preprocessed images to: {preprocessed_root}")
    
    # Instantiate transform pipeline
    transform = get_transforms(config)
    
    classes = ["D00", "D20", "D40", "Normal"]
    splits = ["train", "val", "test"]
    
    # Process each split and class
    for split in splits:
        for cls_name in classes:
            src_dir = src_root / split / cls_name
            if not src_dir.exists():
                continue
                
            dest_dir = preprocessed_root / split / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            img_paths = list(src_dir.glob("*.jpg"))
            if not img_paths:
                continue
                
            print(f"Processing split '{split}', class '{cls_name}' ({len(img_paths)} images)...")
            
            for img_path in tqdm(img_paths):
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                
                # Convert BGR to RGB (transforms expect RGB inputs)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply OpenCV preprocessing only
                preprocessed_np = transform.preprocess_image(image)
                
                # Convert back to BGR for cv2.imwrite
                if len(preprocessed_np.shape) == 2 or preprocessed_np.shape[2] == 1:
                    bgr_save = preprocessed_np
                else:
                    bgr_save = cv2.cvtColor(preprocessed_np, cv2.COLOR_RGB2BGR)
                    
                # Save preprocessed image
                dest_path = dest_dir / img_path.name
                cv2.imwrite(str(dest_path), bgr_save)
                
    print(f"Offline preprocessing completed successfully! Saved dataset to: {preprocessed_root}")

if __name__ == "__main__":
    main()
