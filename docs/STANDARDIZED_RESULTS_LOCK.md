# Standardized Results Lock — Thesis 2026

This file records the standardized experimental results that should be treated as the canonical primary results for the revised thesis unless a documented implementation error is later found.

## 1. Standardized LC25000 protocol

- Architectures: DenseNet121, EfficientNet-B3, ResNet50, MobileNetV2, ViT-B/16
- Initialization: torchvision ImageNet-1K V1 weights
- Train+Validation manifest: 22,501 LC25000 images
- Cross-validation: 5-fold StratifiedKFold, shuffle=True, random_state=42
- Packaged LC25000 Test split: excluded from fold construction, training, and model selection
- Input size: 224 x 224
- Training augmentation: random horizontal flip p=0.5, random vertical flip p=0.5, random rotation 15 degrees
- Normalization: ImageNet mean/std
- Optimizer: Adam, learning rate 1e-4
- Loss: CrossEntropyLoss
- Batch size: 16
- Maximum epochs: 50
- Early stopping patience: 7
- Checkpoint selection: minimum validation loss
- Final training length: median of the five fold-specific best epochs
- Final model: reinitialized from ImageNet-1K V1 and trained on all Train+Validation samples for the fixed median epoch count
- Packaged Test evaluation: once after final training
- AMP: disabled
- Determinism: seeded execution, deterministic PyTorch/cuDNN settings, CUBLAS_WORKSPACE_CONFIG=:4096:8; ViT flash and memory-efficient SDP disabled, math SDP enabled

### LC25000 results

| Model | CV Accuracy | CV Macro-F1 | Best epochs | Median epoch | Final Test Accuracy | Final Test Macro-F1 |
|---|---:|---:|---|---:|---:|---:|
| DenseNet121 | 99.996% ± 0.010% | 99.996% ± 0.010% | [15, 3, 14, 2, 7] | 7 | 99.880% | 99.880% |
| EfficientNet-B3 | 100.000% ± 0.000% | 100.000% ± 0.000% | [11, 11, 28, 11, 14] | 11 | 99.960% | 99.960% |
| ResNet50 | 99.969% ± 0.030% | 99.969% ± 0.030% | [6, 19, 14, 14, 9] | 14 | 99.800% | 99.800% |
| MobileNetV2 | 99.987% ± 0.020% | 99.987% ± 0.020% | [6, 19, 8, 6, 23] | 8 | 100.000% | 100.000% |
| ViT-B/16 | 99.876% ± 0.143% | 99.876% ± 0.143% | [14, 35, 11, 31, 4] | 14 | 99.240% | 99.239% |

The earlier ViT run that emitted the non-deterministic memory-efficient-attention warning is excluded from the standardized evidence. The clean deterministic rerun above is canonical.

## 2. Standardized target feature-probe protocol

For NCT-CRC-HE-100K-NONORM and CRC-VAL-HE-7K:

- Load the standardized LC25000 final checkpoint.
- Remove the LC25000 5-class head.
- Freeze the deep backbone.
- Extract the architecture's native feature vector.
- Input: 224 x 224, ToTensor, ImageNet normalization.
- Feature dimensions: DenseNet121 1024; EfficientNet-B3 1536; ResNet50 2048; MobileNetV2 1280; ViT-B/16 768.
- Target classifier: LogisticRegression(max_iter=3000, n_jobs=-1).
- Evaluation: five stratified random 80/20 splits with seeds 1, 2, 3, 4, 5.
- Metrics: accuracy, macro precision, macro recall, macro F1, macro one-vs-rest ROC-AUC, MCC, macro FNR.
- Matched control: untouched torchvision ImageNet-1K V1 DenseNet121, classifier removed, same 1024-dimensional pooled representation and identical target-probe protocol.

### NCT-CRC-HE-100K-NONORM

100,000 images; 9 classes: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM.

| Representation | Accuracy | Macro-F1 |
|---|---:|---:|
| LC25000 DenseNet121 | 97.304% ± 0.071% | 97.291% ± 0.080% |
| LC25000 EfficientNet-B3 | 96.224% ± 0.108% | 96.253% ± 0.114% |
| LC25000 ResNet50 | 96.308% ± 0.158% | 96.295% ± 0.159% |
| LC25000 MobileNetV2 | 95.721% ± 0.183% | 95.747% ± 0.186% |
| LC25000 ViT-B/16 | 96.630% ± 0.180% | 96.590% ± 0.185% |
| ImageNet-only DenseNet121 control | 97.408% ± 0.050% | 97.387% ± 0.046% |

Matched DenseNet accuracy difference: LC25000-trained minus ImageNet-only = **-0.104 percentage points**.

Interpretation: DenseNet121 yields strong transferable NCT representations, but the matched control does not support attributing an NCT transfer advantage to LC25000 fine-tuning.

### CRC-VAL-HE-7K

7,180 images; 9 classes: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM.

| Representation | Accuracy | Macro-F1 |
|---|---:|---:|
| LC25000 DenseNet121 | 98.942% ± 0.159% | 98.680% ± 0.246% |
| LC25000 EfficientNet-B3 | 98.022% ± 0.310% | 97.429% ± 0.359% |
| LC25000 ResNet50 | 97.117% ± 0.405% | 96.452% ± 0.464% |
| LC25000 MobileNetV2 | 97.409% ± 0.151% | 96.957% ± 0.233% |
| LC25000 ViT-B/16 | 98.162% ± 0.244% | 97.755% ± 0.371% |
| ImageNet-only DenseNet121 control | 98.440% ± 0.402% | 98.024% ± 0.505% |

Matched DenseNet accuracy difference: LC25000-trained minus ImageNet-only = **+0.502 percentage points**.
Matched DenseNet Macro-F1 difference = **+0.656 percentage points**.

Interpretation: LC25000 fine-tuning shows a modest positive transfer difference on CRC under the matched DenseNet121 protocol. Together with the NCT result, the incremental effect is dataset-dependent rather than universally positive.

## 3. Historical extension results

The following remain valid as historical or extension experiments but must not be merged into the standardized primary tables as if they used the same protocol:

- B0-KD + DenseNet exploratory NCT feature fusion: approximately 98.00% accuracy.
- GlaS DenseNet frozen-feature experiment: approximately 98.79% ± 1.48% accuracy.
- EfficientNet-B3 CRC stain-normalization ablation: no-normalization accuracy approximately 97.34% versus Reinhard-normalized approximately 96.17%.
- Knowledge-distillation experiments using EfficientNet-B3 teacher and EfficientNet-B0 student.

## 4. Canonical local paths

- Standardized LC runner: `evaluation/standardized_lc25000_experiment.py`
- Target probe runner: `evaluation/standardized_target_probe.py`
- LC outputs: `runs/standardized_lc25000/`
- Shared LC manifest: `runs/standardized_lc25000/protocol/lc25000_five_fold_manifest.csv`
- Target-probe outputs: `runs/standardized_target_probes/`

## 5. Re-run policy

Do not rerun completed standardized experiments merely to obtain a different value. Re-run only if a concrete implementation, data-integrity, or protocol error is discovered and documented. If a corrective rerun is necessary, preserve the superseded result and record the reason for replacement.
