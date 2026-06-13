import torchvision.models as models
import torch.nn as nn

def get_vit_b16(num_classes=5):
    model = models.vit_b_16(pretrained=True)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model
