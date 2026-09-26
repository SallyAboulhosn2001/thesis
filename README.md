# Thesis Codebase

Source code for the MSc thesis on lung and colon cancer histopathology classification.

## Repository scope
This repository contains source code for LC25000 model training/evaluation, standardized five-fold cross-validation, NCT/CRC frozen-feature transfer probes, knowledge distillation, feature fusion, preprocessing/stain-normalization experiments, evaluation metrics, and explainability.

Large datasets, trained checkpoints, extracted feature tensors, generated figures, logs, and run directories are intentionally excluded from version control.

## Primary standardized scripts
- `evaluation/standardized_lc25000_experiment.py`
- `evaluation/standardized_target_probe.py`

## Principal standardized LC25000 results

| Model | CV Accuracy | CV Macro-F1 | Median epoch | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|
| DenseNet121 | 99.996 ± 0.010% | 99.996 ± 0.010% | 7 | 99.880% | 99.880% |
| EfficientNet-B3 | 100.000 ± 0.000% | 100.000 ± 0.000% | 11 | 99.960% | 99.960% |
| ResNet50 | 99.969 ± 0.030% | 99.969 ± 0.030% | 14 | 99.800% | 99.800% |
| MobileNetV2 | 99.987 ± 0.020% | 99.987 ± 0.020% | 8 | 100.000% | 100.000% |
| ViT-B/16 | 99.876 ± 0.143% | 99.876 ± 0.143% | 14 | 99.240% | 99.239% |

## Standardized transfer results

### NCT-CRC-HE-100K-NONORM
- DenseNet121: 97.304 ± 0.071% accuracy
- EfficientNet-B3: 96.224 ± 0.108%
- ResNet50: 96.308 ± 0.158%
- MobileNetV2: 95.721 ± 0.183%
- ViT-B/16: 96.630 ± 0.180%
- ImageNet-only DenseNet121 control: 97.408 ± 0.050%

### CRC-VAL-HE-7K
- DenseNet121: 98.942 ± 0.159% accuracy
- EfficientNet-B3: 98.022 ± 0.310%
- ResNet50: 97.117 ± 0.405%
- MobileNetV2: 97.409 ± 0.151%
- ViT-B/16: 98.162 ± 0.244%
- ImageNet-only DenseNet121 control: 98.440 ± 0.402%

See `docs/STANDARDIZED_RESULTS_LOCK.md` for the locked protocol/results.

## Standardized rerun environment
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

For deterministic CUDA execution:
```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

## Reproducibility
See:
- `docs/REPRODUCIBILITY.md`
- `docs/METHODOLOGY_CODE_MAP.md`
- `docs/STANDARDIZED_RESULTS_LOCK.md`
