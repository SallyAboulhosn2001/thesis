# Methodology-to-Code Map

## Standardized LC25000 comparison
Primary:
- `evaluation/standardized_lc25000_experiment.py`

Implements the five principal architectures, shared stratified 5-fold CV over LC25000 Train+Validation, final-model training using the median best epoch, and one-time packaged-Test evaluation.

## Standardized target-domain probes
Primary:
- `evaluation/standardized_target_probe.py`

Implements frozen-feature probing on NCT/CRC with Logistic Regression over five stratified 80/20 splits and the matched ImageNet-only DenseNet121 control.

Historical/supporting:
- `evaluation/extract_nct_features_densenet121.py`
- `evaluation/evaluate_features.py`
- `evaluation/extract_crcval_features_densenet.py`

## Knowledge distillation / no-KD
- `kd_run.py`
- `b0_no_kd_run.py`
- `models/student.py`
- `training/kd_trainer.py`

## Feature fusion
- `evaluation/fuse_b0kd_densenet.py`
- `evaluation/train_light_classifiers_fused.py`
- `evaluation/evaluate_fusion_test_set.py`
- `evaluation/extract_crcval_features_fusion.py`

## Preprocessing / stain normalization
- `preprocessing/stain_normalization.py`
- `preprocessing/normalize.py`
- `evaluation/train_effnetb3_lc25000_ablation.py`
- `evaluation/extract_crcval_features_effnetb3_ablation.py`

## Metrics / evaluation
- `evaluation/full_metrics.py`
- `evaluation/metrics.py`
- `evaluation/confusion_matrix_on_dataset.py`
- `evaluation/generate_master_results.py`

## Explainability
- `xai/gradcam.py`
- `xai/gradcam_simple.py`
- `evaluation/run_gradcam.py`
- `evaluation/gradcam_visualization.py`
- `evaluation/gradcam_efficientnet_b3.py`
- `evaluation/vit_attention_visualize.py`
- `xai/lime.py`

## GlaS / case mining
- `evaluation/xai_case_miner.py`
- `evaluation/xai_case_miner_glas.py`
- `evaluation/extract_glas_features_densenet.py`
