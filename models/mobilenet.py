import torchvision.models as models
import torch.nn as nn

def get_mobilenet_v2(num_classes=5):
    model = models.mobilenet_v2(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
