import cv2
import numpy as np
from torchvision import transforms as T


# Pixel Preprocessing

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8), **kwargs):
    return image


def apply_grayscale_bilateral(image, d=9, sigma_color=75, sigma_space=75, **kwargs):
    return image


# Geometric Preprocessing

def apply_letterbox_resize(image, target_size=224, **kwargs):
    return image

# PIPELINE TRANSFORMS


class OpenCVPreprocessingPipeline:
    def __init__(self, mode, image_size, params, mean, std):
        self.mode = mode
        self.image_size = image_size
        self.params = params
        
        # PyTorch transforms (To Tensor converting to 0.0-1.0, and Normalize)
        self.pytorch_transform = T.Compose([
            T.ToTensor(),                   
            T.Normalize(mean=mean, std=std) 
        ])

    def __call__(self, image, **kwargs):

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

        if self.mode == "letterbox" or self.mode == "combined":
            lb_params = self.params.get("letterbox", {})
            image = apply_letterbox_resize(image, target_size=self.image_size, **lb_params)
        else:

            image = cv2.resize(image, (self.image_size, self.image_size))


        image = self.pytorch_transform(image)
        
        return {"image": image}


def get_transforms(config):
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
