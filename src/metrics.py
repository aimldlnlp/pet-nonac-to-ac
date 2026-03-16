import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _safe_data_range(gt):
    dr = float(gt.max() - gt.min())
    return max(dr, 1e-6)


def ssim_3d_slice_mean(pred, gt):
    # Stable SSIM for 3D volume by averaging axial slices.
    vals = []
    dr = _safe_data_range(gt)
    for i in range(pred.shape[2]):
        vals.append(structural_similarity(pred[:, :, i], gt[:, :, i], data_range=dr))
    return float(np.mean(vals))


def nrmse(pred, gt):
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    denom = np.sqrt(np.mean(gt**2)) + 1e-6
    return float(rmse / denom)


def _suvpeak(vals, peak_voxels=64):
    if vals.size == 0:
        return 0.0
    k = int(max(min(peak_voxels, vals.size), 1))
    # Top-k average as SUVpeak surrogate.
    topk = np.partition(vals, -k)[-k:]
    return float(np.mean(topk))


def suv_errors(pred, gt, mask_mean=None, mask_max=None, peak_voxels=64):
    if mask_mean is not None and np.any(mask_mean):
        pred_mean_vals = pred[mask_mean]
        gt_mean_vals = gt[mask_mean]
    else:
        pred_mean_vals = pred
        gt_mean_vals = gt

    if mask_max is not None and np.any(mask_max):
        pred_max_vals = pred[mask_max]
        gt_max_vals = gt[mask_max]
    else:
        pred_max_vals = pred
        gt_max_vals = gt

    suvmean_err = abs(pred_mean_vals.mean() - gt_mean_vals.mean()) / (gt_mean_vals.mean() + 1e-6) * 100.0
    suvmax_err = abs(pred_max_vals.max() - gt_max_vals.max()) / (gt_max_vals.max() + 1e-6) * 100.0
    pred_peak = _suvpeak(pred_max_vals, peak_voxels=peak_voxels)
    gt_peak = _suvpeak(gt_max_vals, peak_voxels=peak_voxels)
    suvpeak_err = abs(pred_peak - gt_peak) / (gt_peak + 1e-6) * 100.0
    return float(suvmean_err), float(suvmax_err), float(suvpeak_err)


def compute_case_metrics(
    nonac,
    pred,
    gt,
    suv_pred=None,
    suv_gt=None,
    suv_mask_mean=None,
    suv_mask_max=None,
    suv_mask=None,
    suv_peak_voxels=64,
):
    dr = _safe_data_range(gt)
    psnr_pred = peak_signal_noise_ratio(gt, pred, data_range=dr)
    psnr_nonac = peak_signal_noise_ratio(gt, nonac, data_range=dr)
    ssim_pred = ssim_3d_slice_mean(pred, gt)
    nrmse_pred = nrmse(pred, gt)
    if suv_pred is None:
        suv_pred = pred
    if suv_gt is None:
        suv_gt = gt
    # Backward-compatible single-mask behavior if caller still passes suv_mask.
    if suv_mask is not None and suv_mask_mean is None and suv_mask_max is None:
        suv_mask_mean = suv_mask
        suv_mask_max = suv_mask
    suvmean_err, suvmax_err, suvpeak_err = suv_errors(
        suv_pred,
        suv_gt,
        mask_mean=suv_mask_mean,
        mask_max=suv_mask_max,
        peak_voxels=suv_peak_voxels,
    )

    return {
        "psnr": float(psnr_pred),
        "ssim": float(ssim_pred),
        "nrmse": float(nrmse_pred),
        "suvmean_err_pct": float(suvmean_err),
        "suvmax_err_pct": float(suvmax_err),
        "suvpeak_err_pct": float(suvpeak_err),
        "psnr_gain_db": float(psnr_pred - psnr_nonac),
    }


def save_metrics_report(rows, out_csv, out_json):
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    summary = {
        "num_cases": int(len(df)),
        "mean": {k: float(df[k].mean()) for k in df.columns if k != "case_id"},
        "std": {k: float(df[k].std(ddof=0)) for k in df.columns if k != "case_id"},
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return df, summary
