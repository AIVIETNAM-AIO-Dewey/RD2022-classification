import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_transforms(config, is_train=True):
    """
    Builds the image preprocessing pipeline based on configuration.
    
    Args:
        config (dict): Configuration dictionary containing preprocessing options.
        is_train (bool): If True, applies data augmentations (if enabled).
        
    Returns:
        albumentations.Compose: Preprocessing transform pipeline.
    """
    transforms_list = []
    
    # TODO: [Thực nghiệm 1] Thêm CLAHE (Contrast Limited Adaptive Histogram Equalization) vào đây
    # Ví dụ: transforms_list.append(A.CLAHE(...))
    
    # TODO: [Thực nghiệm 2] Thêm Gaussian Blur vào đây
    # Ví dụ: transforms_list.append(A.GaussianBlur(...))
    
    # TODO: [Thực nghiệm 3] Thêm Data Augmentation (chỉ chạy trong training mode) vào đây
    # Ví dụ:
    # if is_train:
    #     transforms_list.extend([A.HorizontalFlip(...), ...])
        
    # --- BASELINE TRANSFORMS (Mặc định luôn chạy) ---
    # 1. Resize ảnh
    image_size = config.get("image_size", 224)
    transforms_list.append(A.Resize(image_size, image_size))
    
    # 2. Chuẩn hóa ảnh (Standardization & Normalization)
    mean = config.get("mean", [0.485, 0.456, 0.406])
    std = config.get("std", [0.229, 0.224, 0.225])
    transforms_list.append(A.Normalize(mean=mean, std=std))
    
    # 3. Chuyển đổi thành PyTorch Tensor
    transforms_list.append(ToTensorV2())
    
    return A.Compose(transforms_list)
