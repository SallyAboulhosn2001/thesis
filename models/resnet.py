import torch.nn as nn
import torchvision.models as models

def get_resnet50(num_classes=5, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
