from __future__ import annotations
import torch.nn as nn
from torchvision import models

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a real CNN using pretrained ResNet18.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): If True, use pretrained ImageNet weights.

    Returns:
        nn.Module: ResNet18 model ready for fine-tuning.
    """
    # Load ResNet18 with or without pretrained weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

    # Optionally freeze early layers (set requires_grad=False to freeze)
    # Here we keep all layers trainable
    for param in model.parameters():
        param.requires_grad = True

    # Replace the final fully connected layer to match our number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model