import torch.nn as nn
from torchvision import models


def get_efficientnet_b0_student(num_classes=5, pretrained=True):
    m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Linear(in_features, num_classes)
    return m
