# Repo Layout

This repository is organized into four main areas:

- `src/`: model implementation, dataset loader, losses, metrics, NIfTI I/O, and visualization
- root entry points: main scripts for training, calibration, inference, evaluation, and the verification gate
- `configs/`: reusable experiment presets
- `data/`: NIfTI dataset used as pipeline input

High-level structure:

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

Repository conventions:

- Python files at the repository root are reserved for pipeline entry points.
- Logs, checkpoints, prediction outputs, and evaluation reports belong in output directories that are ignored by Git.
- Experiment variants should live in `configs/`, not be hardcoded into scripts.
- Non-essential utilities belong in `scripts/`.

Directories treated as run artifacts:

- `checkpoints/`
- `outputs/`
- `reports/`
- `runs/`

These directories are useful for local work, but they are not the core source tree you would typically highlight in a portfolio or research review.
