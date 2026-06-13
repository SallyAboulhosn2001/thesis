import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader
from datasets_code.dataset_crc_val import CRCVALDataset
from models.densenet import get_densenet121


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CRCVALDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = get_densenet121(num_classes=5)  # same as trained LC25000
    model.load_state_dict(torch.load("densenet121_lc25000.pth", map_location=device))
    model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            features = model.features(x)
            features = torch.flatten(
                torch.nn.functional.adaptive_avg_pool2d(features, 1), 1
            )

            all_features.append(features.cpu())
            all_labels.append(y)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    torch.save(
    {"X": all_features, "y": all_labels},
    "crcval_features_densenet121.pt"
)

    print("CRC-VAL DenseNet features saved.")
    print("Shape:", all_features.shape)


if __name__ == "__main__":
    main()