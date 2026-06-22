import cv2
import numpy as np
from torchvision import transforms as T

# =============================================================================
# CÁC HÀM PREPROCESSING SỬ DỤNG OPENCV
# =============================================================================

# --- Pixel Preprocessing ---

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8), **kwargs):
    return image


def apply_grayscale_bilateral(image, d=9, sigma_color=75, sigma_space=75, **kwargs):
    return image


# --- Geometric Preprocessing ---

def apply_letterbox_resize(image, target_size=224, **kwargs):
    return image


# =============================================================================
# PIPELINE TRANSFORMS
# =============================================================================

class OpenCVPreprocessingPipeline:
    """
    Pipeline hoàn toàn bằng OpenCV (đối với bước 2 & 3) và PyTorch torchvision.transforms
    (đối với bước 4 & 5). Không sử dụng thư viện Albumentations.
    """
    def __init__(self, mode, image_size, params, mean, std):
        self.mode = mode
        self.image_size = image_size
        self.params = params
        
        # PyTorch transforms (To Tensor converting to 0.0-1.0, and Normalize)
        self.pytorch_transform = T.Compose([
            T.ToTensor(),                   # Bước 4: Chuyển hóa Tensor (0-255 → 0.0-1.0)
            T.Normalize(mean=mean, std=std) # Bước 5: Chuẩn hóa phân phối (luôn luôn là bước cuối)
        ])

    def __call__(self, image, **kwargs):
        # 1. Read: ảnh đã được đọc trong dataset.py dạng numpy RGB (0-255)
        
        # 2. Pixel Preprocessing (Vẫn giữ nguyên dải màu 0-255)
        if self.mode == "clahe":
            clahe_params = self.params.get("clahe", {})
            image = apply_clahe(image, **clahe_params)
            
        elif self.mode == "grayscale_bilateral":
            gb_params = self.params.get("grayscale_bilateral", {})
            image = apply_grayscale_bilateral(image, **gb_params)
            
        elif self.mode == "combined":
            clahe_params = self.params.get("clahe", {})
            gb_params = self.params.get("grayscale_bilateral", {})
            # Grayscale -> CLAHE -> Bilateral Filter
            image = apply_grayscale_bilateral(image, **gb_params)
            image = apply_clahe(image, **clahe_params)

        # 3. Geometric Preprocessing (Vẫn giữ nguyên dải màu 0-255)
        if self.mode == "letterbox" or self.mode == "combined":
            lb_params = self.params.get("letterbox", {})
            image = apply_letterbox_resize(image, target_size=self.image_size, **lb_params)
        else:
            # Baseline/Default: OpenCV Standard Resize
            image = cv2.resize(image, (self.image_size, self.image_size))

        # 4 & 5. ToTensor & Normalize
        image = self.pytorch_transform(image)
        
        return {"image": image}


def get_transforms(config):
    """Xây dựng pipeline tiền xử lý ảnh dựa trên preprocessing_mode trong config."""
    mode = config.get("preprocessing_mode", "baseline")
    image_size = config.get("image_size", 224)
    params = config.get("preprocessing_params", {})
    
    mean = config.get("mean", [0.485, 0.456, 0.406])
    std = config.get("std", [0.229, 0.224, 0.225])

    return OpenCVPreprocessingPipeline(
        mode=mode, 
        image_size=image_size, 
        params=params, 
        mean=mean, 
        std=std
    )
