import cv2
import numpy as np
from torchvision import transforms as T


# Pixel Preprocessing

def apply_clahe(image, **kwargs):

    clip_limit , tile_grid_size = kwargs.values()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clip_limit, tile_grid_size)
    cl = clahe.apply(l)
    clahe_image = cv2.merge((cl, a, b))
    clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_LAB2BGR)
    
    return clahe_image

def apply_bilateral (image, **kwargs):

    d, sigma_color, sigma_space = kwargs.values()
    bilateral_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    return bilateral_image

def apply_grayscale(image):

    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image_bw

def apply_grayscale_bilateral (image, **kwargs):
    
    image_bw = apply_grayscale(image)
    bilateral_img = apply_bilateral(image_bw, **kwargs)

    return bilateral_img



# Geometric Preprocessing
# Padding Color : Black
def apply_letterbox_resize(image, target_size=224, **kwargs):
    mode, value = kwargs.values()
    if mode == "reflect":
        border_mode = cv2.BORDER_REFLECT
    else: 
        border_mode = cv2.BORDER_CONSTANT
    h,w = image.shape[:2]
    scale = target_size / max(h,w)
    new_h, new_w = int(h * scale), int(w * scale)
    if scale < 1:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR
    resize_image = cv2.resize(image, (new_w, new_h), interpolation = interp)
    pad_h, pad_w = target_size - new_h, target_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    letterbox_image = cv2.copyMakeBorder(
        resize_image, 
        top, bottom, left, right,
        border_mode,
        value = value
    )

    return letterbox_image

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

    def preprocess_image (self, image, **kwargs):

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
            
            image = apply_clahe(image, **clahe_params)
            image = apply_grayscale_bilateral(image, **gb_params)

        if self.mode == "letterbox" or self.mode == "combined":
            lb_params = self.params.get("letterbox", {})
            image = apply_letterbox_resize(image, target_size=self.image_size, **lb_params)
        else:
            image = cv2.resize(image, (self.image_size, self.image_size))
            
        return image

    def __call__(self, image, **kwargs):
        image = self.preprocess_image(image)
        image = self.pytorch_transform(image)
        
        return image


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