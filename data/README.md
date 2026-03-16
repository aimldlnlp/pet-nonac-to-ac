# Dataset Layout

The dataset is stored as paired NIfTI volumes per split:

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

Naming rules:

- Files in `nonac/` and `ac_gt/` must match one-to-one.
- Example: `test_000.nii.gz` in `nonac/` must have a matching `test_000.nii.gz` in `ac_gt/`.

Notes:

- The `data/` folder is treated as an experiment input, not as a run output.
- This structure is consumed directly by `train.py`, `calibrate.py`, `infer.py`, and `evaluate.py`.
