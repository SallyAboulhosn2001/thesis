import torch.nn as nn
import torchvision.models as models

def get_densenet121(num_classes=5, pretrained=True):
    model = models.densenet121(pretrained=pretrained)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model
