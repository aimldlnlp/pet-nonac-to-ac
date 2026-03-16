# Config Presets

This folder contains experiment presets for the PET Non-AC to AC pipeline.

Most relevant files:

- `default.yaml`: main baseline
- `hotspot_sprint.yaml`: hotspot-fidelity-focused preset
- `hotspot_balanced.yaml`: trade-off between global quality and hotspot fidelity
- `finetune_peak.yaml`: fine-tuning with peak preservation emphasis
- `finetune_peak_v23.yaml`: stronger peak-focused emphasis
- `smoke.yaml`: lightweight preset for quick validation

Usage principles:

- Use `default.yaml` as the starting point.
- Store experiment variants as new config files rather than editing the pipeline scripts.
- Keep config names descriptive of the experiment objective.
