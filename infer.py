import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.dataset import PetACDataset
from src.io_nifti import write_nifti_gz
from src.model_unet3d import UNet3D
from src.utils import ensure_dirs, load_config


def apply_calibration_candidate(pred_norm, calib):
    kind = calib.get("kind", "linear")
    a = float(calib.get("a", 1.0))
    b = float(calib.get("b", 0.0))
    out = np.clip(a * pred_norm + b, 0.0, None)

    if kind == "piecewise_highboost":
        t = float(calib.get("threshold", 0.95))
        g = float(calib.get("high_gain", 1.0))
        hi = out > t
        if np.any(hi):
            out = out.copy()
            out[hi] = t + g * (out[hi] - t)

    if kind == "topk_rescale":
        k = int(max(int(calib.get("topk_voxels", 64)), 1))
        g = float(calib.get("topk_gain", 1.0))
        flat = out.reshape(-1)
        if flat.size > 0 and g != 1.0:
            kk = min(k, flat.size)
            idx = np.argpartition(flat, -kk)[-kk:]
            out = out.copy()
            out_flat = out.reshape(-1)
            out_flat[idx] = out_flat[idx] * g

    return np.clip(out, 0.0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--no_calibration", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    norm_cfg = cfg.get("normalization", {})
    input_clip_max = norm_cfg.get("input_clip_max", 1.0)
    target_clip_max = norm_cfg.get("target_clip_max", 1.0)
    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base=cfg["model"]["base_channels"],
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    calib = None
    if not args.no_calibration:
        default_calib = Path(cfg["paths"].get("calibration_path", Path(cfg["paths"]["checkpoint_dir"]) / "calibration.json"))
        calib_path = Path(args.calibration) if args.calibration else default_calib
        if calib_path.exists():
            with open(calib_path, "r", encoding="utf-8") as f:
                calib = json.load(f)
            kind = calib.get("kind", "linear")
            msg = (
                f"Using calibration: {calib_path} "
                f"(kind={kind}, a={calib.get('a', 1.0):.6f}, b={calib.get('b', 0.0):.6f}"
            )
            if kind == "piecewise_highboost":
                msg += (
                    f", threshold={float(calib.get('threshold', 0.95)):.4f}, "
                    f"high_gain={float(calib.get('high_gain', 1.0)):.4f}"
                )
            if kind == "topk_rescale":
                msg += (
                    f", topk_voxels={int(calib.get('topk_voxels', 64))}, "
                    f"topk_gain={float(calib.get('topk_gain', 1.0)):.4f}"
                )
            msg += ")"
            print(msg)
        else:
            print(f"Calibration file not found at {calib_path}. Running without calibration.")

    test_ds = PetACDataset(
        Path(cfg["data"]["root"]) / "test",
        augment=False,
        input_clip_max=input_clip_max,
        target_clip_max=target_clip_max,
    )
    out_dir = Path(cfg["paths"]["output_pred_dir"])
    ensure_dirs(out_dir)

    with torch.no_grad():
        for x, _, case_id, p1_t, scale_t in tqdm(test_ds, desc="infer"):
            x = x.unsqueeze(0).to(device)
            pred_norm = model(x).squeeze().cpu().numpy()
            if calib is not None:
                pred_norm = apply_calibration_candidate(pred_norm, calib)
            p1 = float(p1_t.item())
            scale = float(scale_t.item())
            pred_abs = np.clip(pred_norm * scale + p1, 0.0, None).astype("float32")
            write_nifti_gz(out_dir / f"{case_id}_pred_ac.nii.gz", pred_abs)

    print(f"Inference complete. Saved to {out_dir}")


if __name__ == "__main__":
    main()
