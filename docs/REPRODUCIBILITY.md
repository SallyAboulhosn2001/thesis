# Reproducibility Notes

## Environment
Standardized reruns used:
- Python 3.10.11
- PyTorch 2.5.1+cu118
- torchvision 0.20.1+cu118
- scikit-learn 1.7.2
- NumPy 2.2.6
- pandas 2.3.3
- Pillow 12.0.0
- matplotlib 3.10.8
- SciPy 1.15.3
- tqdm 4.67.3
- OpenCV 4.13.0.92

The local PyTorch build used CUDA 11.8.

## Determinism
Before standardized runs:
```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

The standardized scripts enforce deterministic PyTorch behavior. For ViT, flash and memory-efficient scaled-dot-product attention were disabled and the math backend enabled. AMP was not used.

## LC25000 standardized protocol
- 224x224 inputs
- ImageNet normalization
- stratified 5-fold CV over Train+Validation only
- packaged Test excluded from CV/model selection
- Adam, learning rate 1e-4
- cross-entropy loss
- max 50 epochs
- patience 7
- best checkpoint by minimum validation loss
- batch size 16
- final model reinitialized from ImageNet weights and trained on all Train+Validation for the median CV best epoch count
- packaged Test evaluated once after training

## Target-domain probe protocol
- 224x224 inputs
- ImageNet normalization
- LC25000 classifier removed
- backbone frozen
- fixed feature extraction
- Logistic Regression
- five stratified 80/20 splits
- seeds 1-5
- `max_iter=3000`
- no feature scaling
- metrics: accuracy, macro precision/recall/F1, macro OVR AUC, MCC, macro FNR

Matched control:
- torchvision DenseNet121 ImageNet-1K V1
- no LC25000 fine-tuning
- same preprocessing and probing protocol

These target experiments are frozen-feature probes, not zero-shot classification.
