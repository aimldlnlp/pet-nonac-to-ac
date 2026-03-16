import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.io_nifti import read_nifti_gz
from src.metrics import compute_case_metrics, save_metrics_report
from src.utils import ensure_dirs, load_config
from src.viz import (
    save_best_worst_gallery,
    save_case_panel,
    save_error_hist,
    save_hotspot_zoom_panel,
    save_line_profile,
    save_metric_boxplot,
    save_mip_panel,
    save_multiview_panel,
    save_paper_summary_scatter,
    save_roi_bar_chart,
)


def _compute_input_scaler(inp):
    p1 = float(np.percentile(inp, 1))
    p99 = float(np.percentile(inp, 99))
    scale = max(p99 - p1, 1e-6)
    return p1, scale


def _normalize_with_scaler(vol, p1, scale):
    return np.clip((vol - p1) / scale, 0.0, 1.0).astype(np.float32)


def _build_high_uptake_roi(gt_raw, body_mask, quantile=0.99, min_voxels=256):
    if not np.any(body_mask):
        return body_mask

    q = float(np.clip(quantile, 0.5, 0.9999))
    body_vals = gt_raw[body_mask]
    thr = np.quantile(body_vals, q)
    roi = body_mask & (gt_raw >= thr)

    if np.count_nonzero(roi) >= min_voxels:
        return roi

    body_idx = np.flatnonzero(body_mask.ravel())
    k = min(max(int(min_voxels), 1), body_idx.size)
    vals = gt_raw.ravel()[body_idx]
    topk_local = np.argpartition(vals, -k)[-k:]
    roi_idx = body_idx[topk_local]
    roi = np.zeros_like(body_mask, dtype=bool)
    roi.ravel()[roi_idx] = True
    return roi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["data"]["root"]) / "test"
    pred_root = Path(cfg["paths"]["output_pred_dir"])
    report_root = Path(cfg["paths"]["report_dir"])
    fig_root = report_root / "figures"
    ensure_dirs(report_root, fig_root)
    eval_cfg = cfg.get("eval", {})
    roi_quantile = float(eval_cfg.get("suv_roi_quantile", 0.99))
    roi_min_voxels = int(eval_cfg.get("suv_roi_min_voxels", 256))
    roi_peak_voxels = int(eval_cfg.get("suv_peak_voxels", 64))

    rows = []

    gt_paths = sorted((root / "ac_gt").glob("*.nii.gz"))
    for gt_path in tqdm(gt_paths, desc="evaluate"):
        case_id = gt_path.name.replace(".nii.gz", "")
        nonac_path = root / "nonac" / f"{case_id}.nii.gz"
        pred_path = pred_root / f"{case_id}_pred_ac.nii.gz"

        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction missing: {pred_path}")

        gt_raw = read_nifti_gz(gt_path)
        nonac_raw = read_nifti_gz(nonac_path)
        pred_raw = read_nifti_gz(pred_path)

        p1, scale = _compute_input_scaler(nonac_raw)
        gt = _normalize_with_scaler(gt_raw, p1, scale)
        nonac = _normalize_with_scaler(nonac_raw, p1, scale)
        pred = _normalize_with_scaler(pred_raw, p1, scale)

        body_mask = gt_raw > 0
        high_uptake_roi = _build_high_uptake_roi(
            gt_raw, body_mask, quantile=roi_quantile, min_voxels=roi_min_voxels
        )

        m = compute_case_metrics(
            nonac=nonac,
            pred=pred,
            gt=gt,
            suv_pred=pred_raw,
            suv_gt=gt_raw,
            suv_mask_mean=body_mask,
            suv_mask_max=high_uptake_roi,
            suv_peak_voxels=roi_peak_voxels,
        )
        m["suv_roi_voxels"] = int(np.count_nonzero(high_uptake_roi))
        m["case_id"] = case_id
        rows.append(m)

        save_case_panel(case_id, nonac, pred, gt, fig_root / f"{case_id}_panel.png")
        save_multiview_panel(case_id, nonac, pred, gt, fig_root / f"{case_id}_multiview.png")
        save_mip_panel(case_id, nonac, pred, gt, fig_root / f"{case_id}_mip.png")
        save_hotspot_zoom_panel(
            case_id,
            nonac,
            pred,
            gt,
            gt_raw,
            high_uptake_roi,
            fig_root / f"{case_id}_hotspot_zoom.png",
        )
        save_line_profile(case_id, nonac, pred, gt, fig_root / f"{case_id}_line_profile.png")
        save_error_hist(case_id, pred, gt, fig_root / f"{case_id}_error_hist.png")
        save_roi_bar_chart(case_id, pred, gt, fig_root / f"{case_id}_roi_bar.png")

    rows = [{"case_id": r["case_id"], **{k: v for k, v in r.items() if k != "case_id"}} for r in rows]
    df, summary = save_metrics_report(rows, report_root / "metrics_test.csv", report_root / "summary.json")

    save_metric_boxplot(df, fig_root / "metrics_boxplot.png")
    save_paper_summary_scatter(df, fig_root / "paper_summary_scatter.png")
    save_best_worst_gallery(df, fig_root, fig_root / "best_worst_gallery.png")

    print("Evaluation complete.")
    print(summary)


if __name__ == "__main__":
    main()
