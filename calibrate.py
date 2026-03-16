import argparse
import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PetACDataset
from src.io_nifti import read_nifti_gz
from src.metrics import ssim_3d_slice_mean
from src.model_unet3d import UNet3D
from src.utils import ensure_dirs, load_config


def _build_high_uptake_roi(gt_abs, body_mask, quantile=0.99, min_voxels=256):
    if not np.any(body_mask):
        return body_mask

    q = float(np.clip(quantile, 0.5, 0.9999))
    body_vals = gt_abs[body_mask]
    thr = np.quantile(body_vals, q)
    roi = body_mask & (gt_abs >= thr)
    if np.count_nonzero(roi) >= min_voxels:
        return roi

    body_idx = np.flatnonzero(body_mask.ravel())
    k = min(max(int(min_voxels), 1), body_idx.size)
    vals = gt_abs.ravel()[body_idx]
    topk_local = np.argpartition(vals, -k)[-k:]
    roi_idx = body_idx[topk_local]
    roi = np.zeros_like(body_mask, dtype=bool)
    roi.ravel()[roi_idx] = True
    return roi


def fit_affine_stats(model, loader, device):
    n = 0.0
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0

    with torch.no_grad():
        for x, y, _, _, _ in tqdm(loader, desc="calibrate"):
            x = x.to(device)
            pred_norm = model(x).squeeze().cpu().numpy().astype(np.float64)
            gt_norm = y.squeeze().cpu().numpy().astype(np.float64)

            mask = gt_norm > 0
            if not np.any(mask):
                continue

            xv = pred_norm[mask]
            yv = gt_norm[mask]

            # Light trimming for robust fit.
            lo = np.quantile(yv, 0.01)
            hi = np.quantile(yv, 0.99)
            keep = (yv >= lo) & (yv <= hi)
            xv = xv[keep]
            yv = yv[keep]
            if xv.size == 0:
                continue

            n += float(xv.size)
            sx += float(xv.sum())
            sy += float(yv.sum())
            sxx += float((xv * xv).sum())
            sxy += float((xv * yv).sum())

    if n <= 1.0:
        return 1.0, 0.0, int(n), 1.0, 0.0

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 1.0, 0.0, int(n), 1.0, 0.0

    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    gain_only = sy / max(sx, 1e-12)

    # Constrain to avoid destructive calibration shifts.
    a = float(np.clip(a, 0.85, 1.15))
    b = float(np.clip(b, -0.08, 0.08))
    gain_only = float(np.clip(gain_only, 0.85, 1.15))
    return a, b, int(n), gain_only, 0.0


def apply_calibration_candidate(pred_norm, cand):
    kind = cand.get("kind", "linear")
    a = float(cand.get("a", 1.0))
    b = float(cand.get("b", 0.0))
    out = np.clip(a * pred_norm + b, 0.0, None)

    if kind == "piecewise_highboost":
        t = float(cand.get("threshold", 0.95))
        g = float(cand.get("high_gain", 1.0))
        hi = out > t
        if np.any(hi):
            out = out.copy()
            out[hi] = t + g * (out[hi] - t)

    if kind == "topk_rescale":
        k = int(max(int(cand.get("topk_voxels", 64)), 1))
        g = float(cand.get("topk_gain", 1.0))
        flat = out.reshape(-1)
        if flat.size > 0 and g != 1.0:
            kk = min(k, flat.size)
            idx = np.argpartition(flat, -kk)[-kk:]
            out = out.copy()
            out_flat = out.reshape(-1)
            out_flat[idx] = out_flat[idx] * g

    return np.clip(out, 0.0, None)


def fit_piecewise_highboost_params(
    model,
    loader,
    device,
    a=1.0,
    b=0.0,
    threshold_quantile=0.90,
):
    thresholds = []
    with torch.no_grad():
        for x, _, _, _, _ in loader:
            x = x.to(device)
            pred = model(x).squeeze().cpu().numpy().astype(np.float64)
            pred_lin = np.clip(a * pred + b, 0.0, None)
            pos = pred_lin[pred_lin > 0]
            if pos.size > 0:
                thresholds.append(float(np.quantile(pos, threshold_quantile)))

    if not thresholds:
        return 0.95, 1.0, 0

    threshold = float(np.clip(np.median(thresholds), 0.75, 0.95))

    num = 0.0
    den = 0.0
    n = 0
    with torch.no_grad():
        for x, y, _, _, _ in loader:
            x = x.to(device)
            pred = model(x).squeeze().cpu().numpy().astype(np.float64)
            gt = y.squeeze().cpu().numpy().astype(np.float64)
            pred_lin = np.clip(a * pred + b, 0.0, None)

            roi = pred_lin > threshold
            if not np.any(roi):
                continue

            dx = pred_lin[roi] - threshold
            dy = gt[roi] - threshold
            keep = dx > 1e-6
            dx = dx[keep]
            dy = dy[keep]
            if dx.size == 0:
                continue

            num += float(np.dot(dx, dy))
            den += float(np.dot(dx, dx))
            n += int(dx.size)

    if den <= 1e-12:
        return threshold, 1.0, n

    high_gain = num / den
    # Allow both compression (<1) and boosting (>1) in high-uptake tail.
    high_gain = float(np.clip(high_gain, 0.80, 1.20))
    return threshold, high_gain, n


def evaluate_candidate_metrics(
    model,
    loader,
    device,
    cand,
    val_split_root,
    roi_quantile=0.99,
    roi_min_voxels=256,
):
    total_err = 0.0
    total_n = 0
    psnr_vals = []
    ssim_vals = []
    suvmean_vals = []
    suvmax_vals = []
    suvpeak_vals = []
    raw_gt_cache = {}
    with torch.no_grad():
        for x, y, case_id, p1_t, scale_t in loader:
            x = x.to(device)
            pred_norm = model(x).squeeze().cpu().numpy().astype(np.float64)
            gt_norm = y.squeeze().cpu().numpy().astype(np.float64)

            pred_cal = apply_calibration_candidate(pred_norm, cand)
            # Keep metric semantics aligned with evaluate.py (normalized domain clipped to [0, 1]).
            pred_img = np.clip(pred_cal, 0.0, 1.0)
            gt_img = np.clip(gt_norm, 0.0, 1.0)
            mask = gt_img > 0
            if not np.any(mask):
                continue

            err = np.abs(pred_img[mask] - gt_img[mask])
            total_err += float(err.sum())
            total_n += int(err.size)

            dr = max(float(gt_img.max() - gt_img.min()), 1e-6)
            psnr_vals.append(float(peak_signal_noise_ratio(gt_img, pred_img, data_range=dr)))
            ssim_vals.append(float(ssim_3d_slice_mean(pred_img, gt_img)))

            # Evaluate SUV errors in absolute domain, matching evaluate.py semantics
            # by using raw GT volume (not clipped/normalized target tensor).
            if isinstance(case_id, (list, tuple)):
                case_key = case_id[0]
            else:
                case_key = case_id
            case_key = str(case_key)

            if case_key not in raw_gt_cache:
                gt_path = val_split_root / "ac_gt" / f"{case_key}.nii.gz"
                raw_gt_cache[case_key] = read_nifti_gz(gt_path).astype(np.float64)
            gt_abs = raw_gt_cache[case_key]

            p1 = float(p1_t.view(-1)[0].item())
            scale = float(scale_t.view(-1)[0].item())
            pred_abs = np.clip(pred_cal * scale + p1, 0.0, None)
            body = gt_abs > 0
            roi = _build_high_uptake_roi(
                gt_abs, body, quantile=roi_quantile, min_voxels=roi_min_voxels
            )

            if np.any(body) and np.any(roi):
                pv_body = pred_abs[body]
                gv_body = gt_abs[body]
                pv_roi = pred_abs[roi]
                gv_roi = gt_abs[roi]
                suvmean = abs(float(pv_body.mean()) - float(gv_body.mean())) / (float(gv_body.mean()) + 1e-6) * 100.0
                suvmax = abs(float(pv_roi.max()) - float(gv_roi.max())) / (float(gv_roi.max()) + 1e-6) * 100.0
                k = min(64, int(gv_roi.size))
                pv_peak = float(np.mean(np.partition(pv_roi, -k)[-k:])) if k > 0 else 0.0
                gv_peak = float(np.mean(np.partition(gv_roi, -k)[-k:])) if k > 0 else 0.0
                suvpeak = abs(pv_peak - gv_peak) / (gv_peak + 1e-6) * 100.0
                suvmean_vals.append(suvmean)
                suvmax_vals.append(suvmax)
                suvpeak_vals.append(suvpeak)

    mae = total_err / max(total_n, 1)
    psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0
    ssim = float(np.mean(ssim_vals)) if ssim_vals else 0.0
    suvmean = float(np.mean(suvmean_vals)) if suvmean_vals else 0.0
    suvmax = float(np.mean(suvmax_vals)) if suvmax_vals else 0.0
    suvpeak = float(np.mean(suvpeak_vals)) if suvpeak_vals else 0.0
    return mae, psnr, ssim, suvmean, suvmax, suvpeak


def fit_topk_gain(
    model,
    loader,
    device,
    val_split_root,
    a=1.0,
    b=0.0,
    roi_quantile=0.99,
    roi_min_voxels=256,
    peak_voxels=64,
):
    ratios = []
    raw_gt_cache = {}
    with torch.no_grad():
        for x, _, case_id, p1_t, scale_t in loader:
            x = x.to(device)
            pred_norm = model(x).squeeze().cpu().numpy().astype(np.float64)
            pred_norm = np.clip(a * pred_norm + b, 0.0, None)

            if isinstance(case_id, (list, tuple)):
                case_key = str(case_id[0])
            else:
                case_key = str(case_id)

            if case_key not in raw_gt_cache:
                gt_path = val_split_root / "ac_gt" / f"{case_key}.nii.gz"
                raw_gt_cache[case_key] = read_nifti_gz(gt_path).astype(np.float64)
            gt_abs = raw_gt_cache[case_key]

            p1 = float(p1_t.view(-1)[0].item())
            scale = float(scale_t.view(-1)[0].item())
            pred_abs = np.clip(pred_norm * scale + p1, 0.0, None)

            body = gt_abs > 0
            roi = _build_high_uptake_roi(
                gt_abs, body, quantile=roi_quantile, min_voxels=roi_min_voxels
            )
            if not np.any(roi):
                continue

            pv = pred_abs[roi]
            gv = gt_abs[roi]
            k = min(max(int(peak_voxels), 1), int(gv.size))
            if k <= 0:
                continue

            pred_peak = float(np.mean(np.partition(pv, -k)[-k:]))
            gt_peak = float(np.mean(np.partition(gv, -k)[-k:]))
            ratios.append(float(gt_peak / (pred_peak + 1e-6)))

    if not ratios:
        return 1.0
    return float(np.clip(np.median(ratios), 1.0, 3.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg.get("eval", {})
    norm_cfg = cfg.get("normalization", {})
    input_clip_max = norm_cfg.get("input_clip_max", 1.0)
    target_clip_max = norm_cfg.get("target_clip_max", 1.0)
    roi_quantile = float(eval_cfg.get("suv_roi_quantile", 0.99))
    roi_min_voxels = int(eval_cfg.get("suv_roi_min_voxels", 256))
    peak_voxels = int(eval_cfg.get("suv_peak_voxels", 64))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pt"

    model = UNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base=cfg["model"]["base_channels"],
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    val_ds = PetACDataset(
        Path(cfg["data"]["root"]) / "val",
        augment=False,
        input_clip_max=input_clip_max,
        target_clip_max=target_clip_max,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    a_aff, b_aff, samples, a_gain, b_gain = fit_affine_stats(model, val_loader, device)

    pw_t_id, pw_g_id, _ = fit_piecewise_highboost_params(
        model, val_loader, device, a=1.0, b=0.0, threshold_quantile=0.90
    )
    pw_t_gain, pw_g_gain, _ = fit_piecewise_highboost_params(
        model, val_loader, device, a=a_gain, b=b_gain, threshold_quantile=0.90
    )
    # Candidate gains to explore both compress and boost behavior on hotspot tail.
    pw_g_id_compress_mild = float(min(pw_g_id, 0.95))
    pw_g_id_compress_strong = float(min(pw_g_id, 0.88))
    pw_g_id_expand_mild = float(max(pw_g_id, 1.05))

    pw_g_gain_compress_mild = float(min(pw_g_gain, 0.95))
    pw_g_gain_compress_strong = float(min(pw_g_gain, 0.88))
    pw_g_gain_expand_mild = float(max(pw_g_gain, 1.05))
    pw_g_gain_expand_strong = float(max(pw_g_gain, 1.12))

    val_split_root = Path(cfg["data"]["root"]) / "val"
    topk_gain_id = fit_topk_gain(
        model,
        val_loader,
        device,
        val_split_root=val_split_root,
        a=1.0,
        b=0.0,
        roi_quantile=roi_quantile,
        roi_min_voxels=roi_min_voxels,
        peak_voxels=peak_voxels,
    )
    topk_gain_gain = fit_topk_gain(
        model,
        val_loader,
        device,
        val_split_root=val_split_root,
        a=a_gain,
        b=b_gain,
        roi_quantile=roi_quantile,
        roi_min_voxels=roi_min_voxels,
        peak_voxels=peak_voxels,
    )

    candidates = [
        {"name": "identity", "kind": "linear", "a": 1.0, "b": 0.0},
        {"name": "gain_only", "kind": "linear", "a": a_gain, "b": b_gain},
        {"name": "affine", "kind": "linear", "a": a_aff, "b": b_aff},
        {
            "name": "piecewise_highboost_identity",
            "kind": "piecewise_highboost",
            "a": 1.0,
            "b": 0.0,
            "threshold": pw_t_id,
            "high_gain": pw_g_id,
        },
        {
            "name": "piecewise_highboost_identity_compress_mild",
            "kind": "piecewise_highboost",
            "a": 1.0,
            "b": 0.0,
            "threshold": pw_t_id,
            "high_gain": pw_g_id_compress_mild,
        },
        {
            "name": "piecewise_highboost_identity_compress_strong",
            "kind": "piecewise_highboost",
            "a": 1.0,
            "b": 0.0,
            "threshold": pw_t_id,
            "high_gain": pw_g_id_compress_strong,
        },
        {
            "name": "piecewise_highboost_identity_expand_mild",
            "kind": "piecewise_highboost",
            "a": 1.0,
            "b": 0.0,
            "threshold": pw_t_id,
            "high_gain": pw_g_id_expand_mild,
        },
        {
            "name": "piecewise_highboost_gain",
            "kind": "piecewise_highboost",
            "a": a_gain,
            "b": b_gain,
            "threshold": pw_t_gain,
            "high_gain": pw_g_gain,
        },
        {
            "name": "piecewise_highboost_gain_compress_mild",
            "kind": "piecewise_highboost",
            "a": a_gain,
            "b": b_gain,
            "threshold": pw_t_gain,
            "high_gain": pw_g_gain_compress_mild,
        },
        {
            "name": "piecewise_highboost_gain_compress_strong",
            "kind": "piecewise_highboost",
            "a": a_gain,
            "b": b_gain,
            "threshold": pw_t_gain,
            "high_gain": pw_g_gain_compress_strong,
        },
        {
            "name": "piecewise_highboost_gain_expand_mild",
            "kind": "piecewise_highboost",
            "a": a_gain,
            "b": b_gain,
            "threshold": pw_t_gain,
            "high_gain": pw_g_gain_expand_mild,
        },
        {
            "name": "piecewise_highboost_gain_expand_strong",
            "kind": "piecewise_highboost",
            "a": a_gain,
            "b": b_gain,
            "threshold": pw_t_gain,
            "high_gain": pw_g_gain_expand_strong,
        },
        {
            "name": "topk_rescale_identity",
            "kind": "topk_rescale",
            "a": 1.0,
            "b": 0.0,
            "topk_voxels": peak_voxels,
            "topk_gain": topk_gain_id,
        },
        {
            "name": "topk_rescale_gain",
            "kind": "topk_rescale",
            "a": a_gain,
            "b": b_gain,
            "topk_voxels": peak_voxels,
            "topk_gain": topk_gain_gain,
        },
        {
            "name": "topk_rescale_gain_mild",
            "kind": "topk_rescale",
            "a": a_gain,
            "b": b_gain,
            "topk_voxels": peak_voxels,
            "topk_gain": float(np.clip(topk_gain_gain * 0.85, 1.0, 3.0)),
        },
        {
            "name": "topk_rescale_gain_strong",
            "kind": "topk_rescale",
            "a": a_gain,
            "b": b_gain,
            "topk_voxels": peak_voxels,
            "topk_gain": float(np.clip(topk_gain_gain * 1.15, 1.0, 3.0)),
        },
    ]

    scored = []
    for cand in candidates:
        mae, psnr, ssim, suvmean, suvmax, suvpeak = evaluate_candidate_metrics(
            model,
            val_loader,
            device,
            cand=cand,
            val_split_root=val_split_root,
            roi_quantile=roi_quantile,
            roi_min_voxels=roi_min_voxels,
        )
        scored.append(
            {
                **cand,
                "val_mae": float(mae),
                "val_psnr": float(psnr),
                "val_ssim": float(ssim),
                "val_suvmean_err_pct": float(suvmean),
                "val_suvmax_err_pct": float(suvmax),
                "val_suvpeak_err_pct": float(suvpeak),
            }
        )

    score_map = {row["name"]: row for row in scored}
    id_row = score_map["identity"]
    id_mae = float(id_row["val_mae"])
    id_psnr = float(id_row["val_psnr"])
    id_ssim = float(id_row["val_ssim"])
    id_suvmean = float(id_row["val_suvmean_err_pct"])
    id_suvmax = float(id_row["val_suvmax_err_pct"])
    id_suvpeak = float(id_row["val_suvpeak_err_pct"])

    # Pick the best candidate among those that pass safety constraints.
    safe = []
    for row in scored:
        if row["name"] == "identity":
            safe.append(row)
            continue

        mae = float(row["val_mae"])
        psnr = float(row["val_psnr"])
        ssim = float(row["val_ssim"])
        suvmean = float(row["val_suvmean_err_pct"])
        suvmax = float(row["val_suvmax_err_pct"])
        suvpeak = float(row["val_suvpeak_err_pct"])

        psnr_safe = psnr >= (id_psnr - 0.10)
        ssim_safe = ssim >= (id_ssim - 0.005)
        suvmean_safe = suvmean <= (id_suvmean * 1.05)
        improvement = (
            (suvpeak <= id_suvpeak * 0.985)
            or (suvmax <= id_suvmax * 0.985)
            or (mae <= id_mae * 0.985)
        )

        if psnr_safe and ssim_safe and suvmean_safe and improvement:
            safe.append(row)

    best = min(
        safe,
        key=lambda r: (
            (float(r["val_suvpeak_err_pct"]) + float(r["val_suvmax_err_pct"])) / 2.0,
            float(r["val_suvpeak_err_pct"]),
            float(r["val_suvmax_err_pct"]),
            float(r["val_suvmean_err_pct"]),
            -float(r["val_psnr"]),
            -float(r["val_ssim"]),
            float(r["val_mae"]),
        ),
    )

    default_out = Path(cfg["paths"].get("calibration_path", Path(cfg["paths"]["checkpoint_dir"]) / "calibration.json"))
    out_path = Path(args.out) if args.out else default_out
    ensure_dirs(out_path.parent)

    payload = {
        "method": f"validation_calibration_{best['name']}",
        "kind": best.get("kind", "linear"),
        "a": float(best.get("a", 1.0)),
        "b": float(best.get("b", 0.0)),
        "threshold": float(best.get("threshold", 0.0)) if "threshold" in best else None,
        "high_gain": float(best.get("high_gain", 1.0)) if "high_gain" in best else None,
        "topk_voxels": int(best.get("topk_voxels", 0)) if "topk_voxels" in best else None,
        "topk_gain": float(best.get("topk_gain", 1.0)) if "topk_gain" in best else None,
        "num_samples": samples,
        "val_mae": float(best["val_mae"]),
        "val_psnr": float(best["val_psnr"]),
        "val_ssim": float(best["val_ssim"]),
        "val_suvmean_err_pct": float(best["val_suvmean_err_pct"]),
        "val_suvmax_err_pct": float(best["val_suvmax_err_pct"]),
        "val_suvpeak_err_pct": float(best["val_suvpeak_err_pct"]),
        "candidates": scored,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved calibration to {out_path}")
    print(payload)


if __name__ == "__main__":
    main()
