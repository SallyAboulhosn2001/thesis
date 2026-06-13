import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
import time
from models.backbone import get_efficientnet_b3
from models.student import get_efficientnet_b0_student
from models.densenet import get_densenet121
from models.resnet import get_resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def measure_inference(model):
    model.eval()
    dummy = torch.randn(1,3,224,224).to(device)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model(dummy)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start)/100

models = {
    "EffNet-B3": get_efficientnet_b3(5),
    "B0-KD": get_efficientnet_b0_student(5),
    "DenseNet121": get_densenet121(5),
    "ResNet50": get_resnet50(5)
}

for name, model in models.items():
    model = model.to(device)
    params = count_params(model)
    inf_time = measure_inference(model)
    print(f"{name}")
    print(f"Params: {params:,}")
    print(f"Inference time (sec): {inf_time:.6f}")
    print("-"*30)
