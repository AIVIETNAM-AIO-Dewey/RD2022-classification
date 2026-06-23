import os
import cv2
from pathlib import Path
from torch.utils.data import Dataset

class RDDDataset(Dataset):
    """
    Custom PyTorch Dataset for loading preprocessed image patches.
    Strictly expects preprocessed images to exist on disk.
    """
    def __init__(self, data_dir, split, transform=None):
        """
        Args:
            data_dir (str): Base directory of processed dataset (e.g. data/processed).
            split (str): Split name ('train', 'val', or 'test').
            transform (OpenCVPreprocessingPipeline): Preprocessing transforms.
        """
        self.transform = transform
        self.split = split
        self.classes = ["D00", "D20", "D40", "Normal"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Determine preprocessing mode
        mode = transform.mode if (transform and hasattr(transform, 'mode')) else "baseline"
        self.raw_data_dir = Path(data_dir)
        self.preprocessed_root = self.raw_data_dir / mode
        
        # We MUST load from the preprocessed folder directly.
        # If it does not exist, raise an error asking the user to run offline preprocess script first.
        if not self.preprocessed_root.exists():
            raise FileNotFoundError(
                f"\n[Error] Preprocessed data directory not found at: {self.preprocessed_root}\n"
                f"You must preprocess the dataset offline before training. Please run:\n"
                f"python src/data/preprocess_dataset.py --config <your_config_file>\n"
            )
            
        self.split_dir = self.preprocessed_root / split
        print(f"[{split}] Loading preprocessed dataset from: {self.split_dir}")
        
        self.image_paths = []
        self.labels = []
        
        # Load all image paths and labels
        for cls_name in self.classes:
            cls_dir = self.split_dir / cls_name
            if not cls_dir.exists():
                continue
                
            for img_path in cls_dir.glob("*.jpg"):
                self.image_paths.append(str(img_path))
                self.labels.append(self.class_to_idx[cls_name])
                
        print(f"Loaded {len(self.image_paths)} images for split: {split}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read preprocessed image (BGR format)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Preprocessed image not found or could not be loaded: {img_path}")
            
        # Convert BGR to RGB (Standard for PyTorch models)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply PyTorch transforms (To Tensor & Normalize) only
        if self.transform:
            transformed = self.transform(image=image, is_preprocessed=True)
            image = transformed["image"]
            
        return image, label
