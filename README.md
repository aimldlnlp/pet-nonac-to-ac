# PET Non-AC to AC Correction Pipeline

A Python-only pipeline for estimating attenuation-corrected (AC) PET volumes from 3D non-attenuation-corrected (Non-AC) PET inputs in NIfTI format.

This repository focuses on:
- training a 3D U-Net for Non-AC to AC volume translation
- post-training calibration to preserve quantitative scale
- batch inference on the test set
- global and hotspot-oriented evaluation
- an objective PASS/FAIL verification gate

## Repo layout

```text
.
├── train.py
├── calibrate.py
├── infer.py
├── evaluate.py
├── verify_gate.py
├── run_multiseed_verify.py
├── configs/
├── data/
├── docs/
├── scripts/
└── src/
```

Structure guide:
- `src/` contains the core pipeline implementation
- `configs/` contains experiment presets
- `scripts/` contains supporting utilities
- `docs/` contains repository notes
- `data/` contains the project input dataset

Quick references:
- see `docs/REPO_LAYOUT.md` for a repository structure overview
- see `configs/README.md` for experiment preset summaries
- see `data/README.md` for the dataset format

## Data Structure

The repository expects paired NIfTI volumes with the following structure:

```text
data/
  train/
    nonac/
    ac_gt/
  val/
    nonac/
    ac_gt/
  test/
    nonac/
    ac_gt/
```

Files in `nonac/` and `ac_gt/` must match one-to-one, for example `test_000.nii.gz`.

The dataset is treated as a project input and is consumed directly by the training, inference, and evaluation pipeline.

## Main Workflow

```bash
python train.py --config configs/default.yaml
python calibrate.py --config configs/default.yaml
python infer.py --config configs/default.yaml
python evaluate.py --config configs/default.yaml
```

For gate-based evaluation:

```bash
python verify_gate.py \
  --csv reports/metrics_test.csv \
  --profile balanced_v1 \
  --out_dir reports/verification
```

## Experiment Configurations

Available presets:
- `configs/default.yaml`: general baseline
- `configs/finetune_peak.yaml`: fine-tuning with peak preservation emphasis
- `configs/finetune_peak_v23.yaml`: stronger peak-focused emphasis
- `configs/hotspot_sprint.yaml`: hotspot fidelity focused preset
- `configs/hotspot_balanced.yaml`: trade-off between hotspot fidelity and global quality

Example using the hotspot-balanced preset:

```bash
python train.py --config configs/hotspot_balanced.yaml
python calibrate.py --config configs/hotspot_balanced.yaml --num_workers 0
python infer.py --config configs/hotspot_balanced.yaml
python evaluate.py --config configs/hotspot_balanced.yaml
python verify_gate.py --csv reports/metrics_test.csv --profile balanced_v1 --out_dir reports/verification
```

## Multi-Seed Verification

```bash
python run_multiseed_verify.py \
  --config configs/hotspot_sprint.yaml \
  --seeds 2026 2027 2028 \
  --work_dir runs/multiseed_verify \
  --profile balanced_v1 \
  --train_num_workers 0 \
  --calibrate_num_workers 0
```

Notes:
- The multi-seed runner uses `data.root` from the provided config.
- By default, the runner disables `resume_ckpt` to keep cross-seed evaluation fair.
- Add `--keep_resume` only if you explicitly want to warm-start from an earlier checkpoint.

## Main Outputs

- `checkpoints/best_model.pt`
- `outputs/predictions/*_pred_ac.nii.gz`
- `reports/metrics_test.csv`
- `reports/summary.json`
- `reports/figures/*.png`

The output directories above are local run artifacts. For research review or portfolio presentation, the primary source tree is `src/`, `configs/`, and the pipeline entry points at the repository root.
# pet-nonac-to-ac
