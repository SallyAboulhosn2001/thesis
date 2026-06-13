import torch


def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            out = model(images)
            features.append(out.cpu())
            labels.append(y)

    return torch.cat(features), torch.cat(labels)
