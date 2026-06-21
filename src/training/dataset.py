import os
import cv2
from pathlib import Path
from torch.utils.data import Dataset

class RDDDataset(Dataset):
    """
    Custom PyTorch Dataset for loading cropped RDD2022 image patches.
    """
    def __init__(self, data_dir, split, transform=None):
        """
        Args:
            data_dir (str): Base directory of processed dataset (e.g. data/processed).
            split (str): Split name ('train', 'val', or 'test').
            transform (albumentations.Compose): Transformations to apply to images.
        """
        self.split_dir = Path(data_dir) / split
        self.transform = transform
        
        self.classes = ["D00", "D20", "D40", "Normal"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
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
        
        # Read image using OpenCV (BGR format)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or could not be loaded: {img_path}")
            
        # Convert BGR to RGB (Standard for PyTorch models)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply Albumentations transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
            
        return image, label
