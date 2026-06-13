import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader
from datasets_code.dataset_glas import GLASDataset
from models.densenet import get_densenet121


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GLASDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = get_densenet121(num_classes=5)
    model.load_state_dict(torch.load("densenet121_lc25000.pth", map_location=device))
    model.to(device)
    model.eval()

    feats = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            f = model.features(x)
            f = torch.relu(f)
            f = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)

            feats.append(f.cpu())
            labels.append(y)

    X = torch.cat(feats)
    y = torch.cat(labels)

    torch.save({"X": X, "y": y}, "glas_features_densenet121.pt")

    print("GlaS DenseNet features saved.")
    print("Shape:", X.shape)


if __name__ == "__main__":
    main()