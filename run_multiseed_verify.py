import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


def _run(cmd, cwd):
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def _write_yaml(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _build_runtime_config(base_cfg, seed, run_root, train_num_workers, keep_resume):
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = int(seed)

    cfg.setdefault("paths", {})
    cfg["paths"]["checkpoint_dir"] = str(run_root / "checkpoints")
    cfg["paths"]["calibration_path"] = str(run_root / "checkpoints" / "calibration.json")
    cfg["paths"]["output_pred_dir"] = str(run_root / "outputs" / "predictions")
    cfg["paths"]["report_dir"] = str(run_root / "reports")

    cfg.setdefault("train", {})
    if train_num_workers is not None:
        cfg["train"]["num_workers"] = int(train_num_workers)
    if not keep_resume:
        cfg["train"].pop("resume_ckpt", None)
        cfg["train"].pop("resume_optimizer", None)

    return cfg


def main():
    ap = argparse.ArgumentParser(description="Run multi-seed pipeline and verification gate.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--work_dir", default="runs/multiseed_verify")
    ap.add_argument("--profile", default="balanced_v1")
    ap.add_argument("--train_num_workers", type=int, default=0)
    ap.add_argument("--calibrate_num_workers", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument(
        "--keep_resume",
        action="store_true",
        help="Keep train.resume_ckpt from input config. Default: disabled for fair multi-seed verification.",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    csv_paths = []

    for seed in args.seeds:
        run_root = work_dir / f"seed_{seed}"
        run_root.mkdir(parents=True, exist_ok=True)

        runtime_cfg = _build_runtime_config(
            base_cfg=base_cfg,
            seed=seed,
            run_root=run_root,
            train_num_workers=args.train_num_workers,
            keep_resume=args.keep_resume,
        )
        cfg_path = run_root / "config_runtime.yaml"
        _write_yaml(cfg_path, runtime_cfg)

        _run([sys.executable, "train.py", "--config", str(cfg_path)], cwd=project_root)
        _run(
            [
                sys.executable,
                "calibrate.py",
                "--config",
                str(cfg_path),
                "--num_workers",
                str(args.calibrate_num_workers),
            ],
            cwd=project_root,
        )
        _run([sys.executable, "infer.py", "--config", str(cfg_path)], cwd=project_root)
        _run([sys.executable, "evaluate.py", "--config", str(cfg_path)], cwd=project_root)

        csv_path = run_root / "reports" / "metrics_test.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected metrics CSV missing: {csv_path}")
        csv_paths.append(str(csv_path))

    verify_out = work_dir / "verification"
    verify_cmd = [
        sys.executable,
        "verify_gate.py",
        "--profile",
        args.profile,
        "--out_dir",
        str(verify_out),
        "--bootstrap",
        str(args.bootstrap),
        "--min_runs",
        str(len(args.seeds)),
        "--csv",
        *csv_paths,
    ]
    _run(verify_cmd, cwd=project_root)

    print(f"Multi-seed verification complete: {verify_out}")


if __name__ == "__main__":
    main()
