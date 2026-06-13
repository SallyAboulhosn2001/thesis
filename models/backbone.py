import torch.nn as nn
from torchvision import models


def get_efficientnet_b3(num_classes, pretrained=True):
    backbone = models.efficientnet_b3(
        weights="IMAGENET1K_V1" if pretrained else None
    )

    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Linear(in_features, num_classes)

    return backbone
