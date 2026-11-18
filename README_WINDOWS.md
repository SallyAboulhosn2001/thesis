# Five-backbone hybrid reproducibility package (2026-11-18)

Extract ALL contents directly inside your existing `C:\Users\sally\Desktop\thesis_code` (without a containing top-level directory). Do not replace an existing script of the same name until you have compared it.

The ten archived CSV files have been verified by SHA-256 against the previous five-backbone audit; they are source seed-1 20k validation predictions (heads trained on 80k NCT) and target 7180 CRC predictions (heads refitted on 100k NCT). These inputs ARE NOT raw histology images and do not constitute a trainable end-to-end model.

PowerShell commands from the `thesis_code` root after extraction:

```
conda activate thesis_gpu
python -m pip install numpy pandas scikit-learn
powershell -NoProfile -ExecutionPolicy Bypass -File .\RUN_FIVE_BACKBONE_SCREEN.ps1
```

The runner creates a unique `runs/five_backbone_hybrid_screen/results_YYYYMMDD_HHMMSS` folder and verifies 25 methods / 7180 target images. Do not commit `runs/`, `datasets/`, weights or patient-related data to GitHub. Source selection is retrospective and target CRC was previously examined: no independent clinical generalization claim.

Stage 5 and 6 launchers are provided, but they require their original archived run directories and do NOT use the five-backbone CSVs. Do not execute them unless their recorded input paths are present. No trainable feature-level hybrid has been implemented by this package.
