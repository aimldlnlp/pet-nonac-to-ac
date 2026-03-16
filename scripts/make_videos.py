import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy import ndimage

from src.io_nifti import read_nifti_gz
from src.utils import ensure_dirs, load_config


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


def _find_hotspot_center(gt_raw, roi_mask):
    if np.any(roi_mask):
        idx = np.argwhere(roi_mask)
        vals = gt_raw[roi_mask]
        peak = idx[int(np.argmax(vals))]
        return int(peak[0]), int(peak[1]), int(peak[2])
    return gt_raw.shape[0] // 2, gt_raw.shape[1] // 2, gt_raw.shape[2] // 2


def _slice_list(length, step):
    idx = list(range(0, length, max(int(step), 1)))
    if idx[-1] != length - 1:
        idx.append(length - 1)
    return idx


def _save_mp4(fig, anim_obj, out_path, fps):
    writer = animation.FFMpegWriter(
        fps=int(fps),
        codec="libx264",
        bitrate=-1,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
    )
    anim_obj.save(out_path, writer=writer, dpi=140)
    plt.close(fig)


def _panel_axes():
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.8))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, axes


def make_axial_panel_video(nonac, pred, gt, out_path, fps=20, step=1):
    err = np.abs(pred - gt)
    vmin = 0.0
    vmax = float(max(np.percentile(gt, 99.5), 1e-6))
    emax = float(max(np.percentile(err, 99.5), 1e-6))

    z_list = _slice_list(gt.shape[2], step)
    fig, axes = _panel_axes()
    titles = ["Input Non-AC", "Predicted AC", "Reference AC", "Absolute Error"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)

    ims = [
        axes[0].imshow(nonac[:, :, z_list[0]].T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax),
        axes[1].imshow(pred[:, :, z_list[0]].T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax),
        axes[2].imshow(gt[:, :, z_list[0]].T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax),
        axes[3].imshow(err[:, :, z_list[0]].T, origin="lower", cmap="magma", vmin=0.0, vmax=emax),
    ]
    fig.colorbar(ims[0], ax=axes[:3], fraction=0.02, pad=0.02).set_label("Normalized Intensity")
    fig.colorbar(ims[3], ax=[axes[3]], fraction=0.08, pad=0.02).set_label("Abs. Error")

    title = fig.suptitle("Axial Cine", y=0.98, fontsize=12)

    def _update(i):
        z = z_list[i]
        ims[0].set_data(nonac[:, :, z].T)
        ims[1].set_data(pred[:, :, z].T)
        ims[2].set_data(gt[:, :, z].T)
        ims[3].set_data(err[:, :, z].T)
        title.set_text(f"Axial Cine (z={z})")
        return ims

    anim_obj = animation.FuncAnimation(fig, _update, frames=len(z_list), interval=1000 / fps, blit=False)
    _save_mp4(fig, anim_obj, out_path, fps=fps)


def make_triplanar_video(nonac, pred, gt, out_path, fps=20, step=1):
    err = np.abs(pred - gt)
    vmin = 0.0
    vmax = float(max(np.percentile(gt, 99.5), 1e-6))
    emax = float(max(np.percentile(err, 99.5), 1e-6))

    max_len = max(gt.shape)
    n_frames = max(2, int(np.ceil(max_len / max(int(step), 1))))
    x_idx = np.linspace(0, gt.shape[0] - 1, n_frames).astype(int)
    y_idx = np.linspace(0, gt.shape[1] - 1, n_frames).astype(int)
    z_idx = np.linspace(0, gt.shape[2] - 1, n_frames).astype(int)

    fig, axes = plt.subplots(3, 4, figsize=(12.8, 8.6))
    titles = ["Input", "Pred", "Ref", "Error"]
    rows = ["Axial", "Coronal", "Sagittal"]
    for r in range(3):
        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10)
            if c == 0:
                axes[r, c].set_ylabel(rows[r], fontsize=10)

    xi, yi, zi = x_idx[0], y_idx[0], z_idx[0]
    axial = [nonac[:, :, zi].T, pred[:, :, zi].T, gt[:, :, zi].T, err[:, :, zi].T]
    coronal = [nonac[:, yi, :].T, pred[:, yi, :].T, gt[:, yi, :].T, err[:, yi, :].T]
    sagittal = [nonac[xi, :, :].T, pred[xi, :, :].T, gt[xi, :, :].T, err[xi, :, :].T]
    grid = [axial, coronal, sagittal]

    ims = []
    for r in range(3):
        row_ims = []
        for c in range(4):
            cmap = "cividis" if c < 3 else "magma"
            lo, hi = (vmin, vmax) if c < 3 else (0.0, emax)
            row_ims.append(axes[r, c].imshow(grid[r][c], origin="lower", cmap=cmap, vmin=lo, vmax=hi))
        ims.append(row_ims)

    fig.colorbar(ims[0][0], ax=axes[:, :3].ravel().tolist(), fraction=0.02, pad=0.02).set_label(
        "Normalized Intensity"
    )
    fig.colorbar(ims[0][3], ax=axes[:, 3].ravel().tolist(), fraction=0.03, pad=0.02).set_label("Abs. Error")
    title = fig.suptitle("Tri-Planar Cine", y=0.99, fontsize=12)
    fig.subplots_adjust(top=0.93, wspace=0.05, hspace=0.12)

    def _update(i):
        xi, yi, zi = x_idx[i], y_idx[i], z_idx[i]
        frame_grid = [
            [nonac[:, :, zi].T, pred[:, :, zi].T, gt[:, :, zi].T, err[:, :, zi].T],
            [nonac[:, yi, :].T, pred[:, yi, :].T, gt[:, yi, :].T, err[:, yi, :].T],
            [nonac[xi, :, :].T, pred[xi, :, :].T, gt[xi, :, :].T, err[xi, :, :].T],
        ]
        for r in range(3):
            for c in range(4):
                ims[r][c].set_data(frame_grid[r][c])
        title.set_text(f"Tri-Planar Cine (x={xi}, y={yi}, z={zi})")
        return [im for row in ims for im in row]

    anim_obj = animation.FuncAnimation(fig, _update, frames=n_frames, interval=1000 / fps, blit=False)
    _save_mp4(fig, anim_obj, out_path, fps=fps)


def make_hotspot_zoom_video(nonac, pred, gt, gt_raw, roi_mask, out_path, fps=20, step=1, zoom_size=48):
    cx, cy, _ = _find_hotspot_center(gt_raw, roi_mask)
    half = max(int(zoom_size // 2), 8)
    x0 = max(cx - half, 0)
    x1 = min(cx + half, gt.shape[0])
    y0 = max(cy - half, 0)
    y1 = min(cy + half, gt.shape[1])

    err = np.abs(pred - gt)
    vmin = 0.0
    vmax = float(max(np.percentile(gt, 99.5), 1e-6))
    emax = float(max(np.percentile(err, 99.5), 1e-6))
    z_list = _slice_list(gt.shape[2], step)

    fig, axes = _panel_axes()
    titles = ["Zoom Input", "Zoom Pred", "Zoom Ref", "Zoom |Error|"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10)

    z = z_list[0]
    ims = [
        axes[0].imshow(nonac[x0:x1, y0:y1, z].T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax),
        axes[1].imshow(pred[x0:x1, y0:y1, z].T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax),
        axes[2].imshow(gt[x0:x1, y0:y1, z].T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax),
        axes[3].imshow(err[x0:x1, y0:y1, z].T, origin="lower", cmap="magma", vmin=0.0, vmax=emax),
    ]
    fig.colorbar(ims[0], ax=axes[:3], fraction=0.02, pad=0.02).set_label("Normalized Intensity")
    fig.colorbar(ims[3], ax=[axes[3]], fraction=0.08, pad=0.02).set_label("Abs. Error")
    title = fig.suptitle("Hotspot Zoom Cine", y=0.98, fontsize=12)

    def _update(i):
        z = z_list[i]
        ims[0].set_data(nonac[x0:x1, y0:y1, z].T)
        ims[1].set_data(pred[x0:x1, y0:y1, z].T)
        ims[2].set_data(gt[x0:x1, y0:y1, z].T)
        ims[3].set_data(err[x0:x1, y0:y1, z].T)
        title.set_text(f"Hotspot Zoom Cine (z={z})")
        return ims

    anim_obj = animation.FuncAnimation(fig, _update, frames=len(z_list), interval=1000 / fps, blit=False)
    _save_mp4(fig, anim_obj, out_path, fps=fps)


def make_mip_rotation_video(nonac, pred, gt, out_path, fps=20, n_angles=72):
    err_vol = np.abs(pred - gt)
    vmin = 0.0
    vmax = float(max(np.percentile(gt, 99.5), 1e-6))
    emax = float(max(np.percentile(err_vol, 99.5), 1e-6))
    angles = np.linspace(0, 180, int(max(n_angles, 8)), endpoint=False)

    fig, axes = _panel_axes()
    titles = ["MIP Input", "MIP Pred", "MIP Ref", "MIP |Error|"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10)

    def _mip_at_angle(vol, angle_deg):
        rot = ndimage.rotate(vol, angle_deg, axes=(0, 1), reshape=False, order=1, mode="nearest")
        return np.max(rot, axis=2).T

    a0 = float(angles[0])
    ims = [
        axes[0].imshow(_mip_at_angle(nonac, a0), origin="lower", cmap="cividis", vmin=vmin, vmax=vmax),
        axes[1].imshow(_mip_at_angle(pred, a0), origin="lower", cmap="cividis", vmin=vmin, vmax=vmax),
        axes[2].imshow(_mip_at_angle(gt, a0), origin="lower", cmap="cividis", vmin=vmin, vmax=vmax),
        axes[3].imshow(_mip_at_angle(err_vol, a0), origin="lower", cmap="magma", vmin=0.0, vmax=emax),
    ]
    fig.colorbar(ims[0], ax=axes[:3], fraction=0.02, pad=0.02).set_label("Normalized Intensity")
    fig.colorbar(ims[3], ax=[axes[3]], fraction=0.08, pad=0.02).set_label("Abs. Error")
    title = fig.suptitle("MIP Rotation", y=0.98, fontsize=12)

    def _update(i):
        angle = float(angles[i])
        ims[0].set_data(_mip_at_angle(nonac, angle))
        ims[1].set_data(_mip_at_angle(pred, angle))
        ims[2].set_data(_mip_at_angle(gt, angle))
        ims[3].set_data(_mip_at_angle(err_vol, angle))
        title.set_text(f"MIP Rotation (angle={angle:.1f}°)")
        return ims

    anim_obj = animation.FuncAnimation(fig, _update, frames=len(angles), interval=1000 / fps, blit=False)
    _save_mp4(fig, anim_obj, out_path, fps=fps)


def _parse_args():
    ap = argparse.ArgumentParser(description="Generate MP4 visualization videos for one PET-AC test case.")
    ap.add_argument("--config", default="configs/hotspot_balanced.yaml")
    ap.add_argument("--case_id", default="test_005")
    ap.add_argument("--pred_path", default=None, help="Optional absolute path to prediction NIfTI (.nii.gz).")
    ap.add_argument("--out_dir", default="reports/videos")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--step", type=int, default=1, help="Slice step for cine videos.")
    ap.add_argument("--zoom_size", type=int, default=48)
    ap.add_argument("--mip_angles", type=int, default=72)
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["axial_panel", "tri_planar", "hotspot_zoom", "mip_rotation"],
        choices=["axial_panel", "tri_planar", "hotspot_zoom", "mip_rotation"],
    )
    return ap.parse_args()


def main():
    args = _parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg first.")

    cfg = load_config(args.config)
    root = Path(cfg["data"]["root"]) / "test"
    pred_root = Path(cfg["paths"]["output_pred_dir"])
    case_id = args.case_id

    gt_path = root / "ac_gt" / f"{case_id}.nii.gz"
    nonac_path = root / "nonac" / f"{case_id}.nii.gz"
    pred_path = Path(args.pred_path) if args.pred_path else pred_root / f"{case_id}_pred_ac.nii.gz"

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    if not nonac_path.exists():
        raise FileNotFoundError(f"Non-AC input not found: {nonac_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction not found: {pred_path}")

    gt_raw = read_nifti_gz(gt_path)
    nonac_raw = read_nifti_gz(nonac_path)
    pred_raw = read_nifti_gz(pred_path)

    p1, scale = _compute_input_scaler(nonac_raw)
    gt = _normalize_with_scaler(gt_raw, p1, scale)
    nonac = _normalize_with_scaler(nonac_raw, p1, scale)
    pred = _normalize_with_scaler(pred_raw, p1, scale)

    eval_cfg = cfg.get("eval", {})
    roi_quantile = float(eval_cfg.get("suv_roi_quantile", 0.99))
    roi_min_voxels = int(eval_cfg.get("suv_roi_min_voxels", 256))
    body_mask = gt_raw > 0
    roi_mask = _build_high_uptake_roi(gt_raw, body_mask, quantile=roi_quantile, min_voxels=roi_min_voxels)

    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    outputs = []
    if "axial_panel" in args.modes:
        out = out_dir / f"{case_id}_axial_panel.mp4"
        make_axial_panel_video(nonac, pred, gt, out, fps=args.fps, step=args.step)
        outputs.append(out)
    if "tri_planar" in args.modes:
        out = out_dir / f"{case_id}_tri_planar.mp4"
        make_triplanar_video(nonac, pred, gt, out, fps=args.fps, step=args.step)
        outputs.append(out)
    if "hotspot_zoom" in args.modes:
        out = out_dir / f"{case_id}_hotspot_zoom.mp4"
        make_hotspot_zoom_video(
            nonac,
            pred,
            gt,
            gt_raw,
            roi_mask,
            out,
            fps=args.fps,
            step=args.step,
            zoom_size=args.zoom_size,
        )
        outputs.append(out)
    if "mip_rotation" in args.modes:
        out = out_dir / f"{case_id}_mip_rotation.mp4"
        make_mip_rotation_video(nonac, pred, gt, out, fps=args.fps, n_angles=args.mip_angles)
        outputs.append(out)

    print("Generated videos:")
    for p in outputs:
        print(f"- {p}")


if __name__ == "__main__":
    main()
