import torch.nn as nn
import torchvision.models as models

def get_model(model_name="resnet18", pretrained=False, num_classes=4):
    """
    Build and return the CNN model.
    
    Args:
        model_name (str): Name of the architecture ('resnet18').
        pretrained (bool): Whether to load pre-trained ImageNet weights.
        num_classes (int): Number of output classes (default is 4 for RDD2022).
        
    Returns:
        torch.nn.Module: The configured PyTorch model.
    """
    if model_name.lower() == "resnet18":
        # Handle older vs newer torchvision versions for loading weights
        try:
            if pretrained:
                # Newer torchvision API
                weights = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=weights)
            else:
                model = models.resnet18(weights=None)
        except AttributeError:
            # Older torchvision API fallback
            model = models.resnet18(pretrained=pretrained)
            
        # Replace the last classification layer (fc)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Model {model_name} is not supported. Currently only 'resnet18' is implemented.")
