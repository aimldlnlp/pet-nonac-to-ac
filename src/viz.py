from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _apply_paper_style():
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
        }
    )


def _save_figure(fig, out_path):
    out_path = Path(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")


def save_case_panel(case_id, nonac, pred, gt, out_path):
    _apply_paper_style()
    _ensure_dir(Path(out_path).parent)
    z = gt.shape[2] // 2

    vmin = 0.0
    vmax = max(np.percentile(gt, 99.5), 1e-6)
    err = np.abs(pred - gt)
    emax = max(np.percentile(err, 99.5), 1e-6)

    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.6), constrained_layout=True)
    panels = [nonac[:, :, z], pred[:, :, z], gt[:, :, z], err[:, :, z]]
    titles = ["Input Non-AC", "Predicted AC", "Reference AC", "Absolute Error"]

    ims = []
    for i, (ax, img, title) in enumerate(zip(axes, panels, titles)):
        if i < 3:
            im = ax.imshow(img.T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(img.T, origin="lower", cmap="magma", vmin=0.0, vmax=emax)
        ims.append(im)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_img = fig.colorbar(ims[0], ax=axes[:3], fraction=0.02, pad=0.02)
    cbar_img.set_label("Normalized Intensity")
    cbar_err = fig.colorbar(ims[3], ax=[axes[3]], fraction=0.08, pad=0.02)
    cbar_err.set_label("Abs. Error")

    fig.suptitle("Qualitative Comparison", y=1.03)
    _save_figure(fig, out_path)
    plt.close(fig)


def save_multiview_panel(case_id, nonac, pred, gt, out_path):
    _apply_paper_style()
    _ensure_dir(Path(out_path).parent)
    xi, yi, zi = gt.shape[0] // 2, gt.shape[1] // 2, gt.shape[2] // 2

    views = [
        (nonac[:, :, zi], pred[:, :, zi], gt[:, :, zi], np.abs(pred[:, :, zi] - gt[:, :, zi]), "Axial"),
        (nonac[:, yi, :], pred[:, yi, :], gt[:, yi, :], np.abs(pred[:, yi, :] - gt[:, yi, :]), "Coronal"),
        (nonac[xi, :, :], pred[xi, :, :], gt[xi, :, :], np.abs(pred[xi, :, :] - gt[xi, :, :]), "Sagittal"),
    ]

    vmin = 0.0
    vmax = max(np.percentile(gt, 99.5), 1e-6)
    emax = max(np.percentile(np.abs(pred - gt), 99.5), 1e-6)

    fig, axes = plt.subplots(3, 4, figsize=(8.5, 6.2), constrained_layout=True)
    for r, (a, b, c, d, name) in enumerate(views):
        imgs = [a, b, c, d]
        titles = [f"{name} Input", f"{name} Pred", f"{name} Ref", f"{name} Error"]
        for c_idx, (img, title) in enumerate(zip(imgs, titles)):
            ax = axes[r, c_idx]
            if c_idx < 3:
                im = ax.imshow(img.T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(img.T, origin="lower", cmap="magma", vmin=0.0, vmax=emax)
            if r == 0:
                ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Three-Plane Visualization", y=1.01)
    _save_figure(fig, out_path)
    plt.close(fig)


def save_line_profile(case_id, nonac, pred, gt, out_path):
    _apply_paper_style()
    _ensure_dir(Path(out_path).parent)
    y = gt.shape[1] // 2
    z = gt.shape[2] // 2
    x = np.arange(gt.shape[0])

    fig, ax = plt.subplots(figsize=(5.4, 2.4), constrained_layout=True)
    ax.plot(x, gt[:, y, z], label="Reference AC", color="#1b4332", linewidth=1.8)
    ax.plot(x, pred[:, y, z], label="Predicted AC", color="#2d6a4f", linewidth=1.5)
    ax.plot(x, nonac[:, y, z], label="Input Non-AC", color="#b08968", linewidth=1.2, alpha=0.9)
    ax.set_title("Line Profile")
    ax.set_xlabel("Voxel Index")
    ax.set_ylabel("Normalized Intensity")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    _save_figure(fig, out_path)
    plt.close(fig)


def save_error_hist(case_id, pred, gt, out_path):
    _apply_paper_style()
    _ensure_dir(Path(out_path).parent)
    err = (pred - gt).ravel()
    fig, ax = plt.subplots(figsize=(4.6, 2.8), constrained_layout=True)
    ax.hist(err, bins=70, color="#bc4749", alpha=0.86, edgecolor="white", linewidth=0.2)
    ax.set_title("Voxel-wise Error Distribution")
    ax.set_xlabel("Prediction - Reference")
    ax.set_ylabel("Voxel Count")
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, out_path)
    plt.close(fig)


def save_metric_boxplot(df, out_path):
    _apply_paper_style()
    _ensure_dir(Path(out_path).parent)

    cols = ["psnr", "ssim", "suvmean_err_pct", "psnr_gain_db"]
    labels = ["PSNR", "SSIM", "SUVmean Err(%)", "PSNR Gain(dB)"]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), constrained_layout=True)

    data_left = [df["psnr"].values, df["ssim"].values]
    bp_left = axes[0].boxplot(data_left, tick_labels=labels[:2], patch_artist=True)
    for patch, c in zip(bp_left["boxes"], ["#a8dadc", "#457b9d"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    axes[0].set_title("Core Image Quality Metrics")
    axes[0].grid(axis="y", alpha=0.25)

    data_right = [df["suvmean_err_pct"].values, df["psnr_gain_db"].values]
    bp_right = axes[1].boxplot(data_right, tick_labels=labels[2:], patch_artist=True)
    for patch, c in zip(bp_right["boxes"], ["#f4a261", "#2a9d8f"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.82)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_title("Correction Performance")
    axes[1].grid(axis="y", alpha=0.25)

    _save_figure(fig, out_path)
    plt.close(fig)


def save_paper_summary_scatter(df, out_path):
    _apply_paper_style()
    _ensure_dir(Path(out_path).parent)

    fig, ax = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)
    x = df["psnr_gain_db"].values
    y = df["ssim"].values
    c = df["suvmean_err_pct"].values

    sc = ax.scatter(x, y, c=c, cmap="viridis_r", s=42, edgecolors="black", linewidths=0.3)

    for _, r in df.iterrows():
        ax.annotate(r["case_id"], (r["psnr_gain_db"], r["ssim"]), fontsize=6, alpha=0.8)

    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xlabel("PSNR Gain (dB)")
    ax.set_ylabel("SSIM")
    ax.set_title("Case-wise Performance Map")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("SUVmean Error (%)")

    _save_figure(fig, out_path)
    plt.close(fig)


def save_roi_bar_chart(case_id, pred, gt, out_path):
    _apply_paper_style()
    _ensure_dir(Path(out_path).parent)

    mask = gt > 0
    if np.any(mask):
        q1, q2 = np.quantile(gt[mask], [0.45, 0.8])
    else:
        q1, q2 = 0.2, 0.6

    rois = {
        "Low uptake": (gt > 0) & (gt <= q1),
        "Medium uptake": (gt > q1) & (gt <= q2),
        "High uptake": gt > q2,
    }

    gt_vals, pred_vals, labels = [], [], []
    for name, roi in rois.items():
        if np.count_nonzero(roi) == 0:
            continue
        labels.append(name)
        gt_vals.append(float(gt[roi].mean()))
        pred_vals.append(float(pred[roi].mean()))

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(5.6, 2.8), constrained_layout=True)
    ax.bar(x - w / 2, gt_vals, width=w, label="Reference", color="#1d3557")
    ax.bar(x + w / 2, pred_vals, width=w, label="Prediction", color="#457b9d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Intensity")
    ax.set_title("ROI Uptake Comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    _save_figure(fig, out_path)
    plt.close(fig)


def save_best_worst_gallery(df, panels_dir, out_path):
    _apply_paper_style()
    _ensure_dir(Path(out_path).parent)
    scored = df.sort_values(["ssim", "psnr_gain_db"], ascending=[False, False])
    best_ids = list(scored.head(3)["case_id"])
    worst_ids = list(scored.tail(3)["case_id"])

    fig, ax = plt.subplots(figsize=(7.0, 2.1), constrained_layout=True)
    ax.axis("off")
    ax.text(0.0, 0.88, "Best Cases:", fontsize=9, weight="bold")
    ax.text(0.0, 0.63, ", ".join(best_ids), fontsize=9)
    ax.text(0.0, 0.36, "Worst Cases:", fontsize=9, weight="bold")
    ax.text(0.0, 0.11, ", ".join(worst_ids), fontsize=9)
    ax.set_title("Case Ranking by SSIM + PSNR Gain")

    _save_figure(fig, out_path)
    plt.close(fig)


def save_mip_panel(case_id, nonac, pred, gt, out_path):
    _apply_paper_style()
    _ensure_dir(Path(out_path).parent)

    nonac_mip = np.max(nonac, axis=2)
    pred_mip = np.max(pred, axis=2)
    gt_mip = np.max(gt, axis=2)
    err_mip = np.max(np.abs(pred - gt), axis=2)

    vmin = 0.0
    vmax = max(np.percentile(gt, 99.5), 1e-6)
    emax = max(np.percentile(np.abs(pred - gt), 99.5), 1e-6)

    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.6), constrained_layout=True)
    panels = [nonac_mip, pred_mip, gt_mip, err_mip]
    titles = ["MIP Input Non-AC", "MIP Predicted AC", "MIP Reference AC", "MIP |Error|"]

    ims = []
    for i, (ax, img, title) in enumerate(zip(axes, panels, titles)):
        if i < 3:
            im = ax.imshow(img.T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(img.T, origin="lower", cmap="magma", vmin=0.0, vmax=emax)
        ims.append(im)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_img = fig.colorbar(ims[0], ax=axes[:3], fraction=0.02, pad=0.02)
    cbar_img.set_label("Normalized Intensity")
    cbar_err = fig.colorbar(ims[3], ax=[axes[3]], fraction=0.08, pad=0.02)
    cbar_err.set_label("Abs. Error")
    fig.suptitle("MIP Comparison", y=1.03)
    _save_figure(fig, out_path)
    plt.close(fig)


def save_hotspot_zoom_panel(case_id, nonac, pred, gt, gt_raw, roi_mask, out_path, patch_size=32):
    _apply_paper_style()
    _ensure_dir(Path(out_path).parent)

    if np.any(roi_mask):
        idx = np.argwhere(roi_mask)
        vals = gt_raw[roi_mask]
        peak = idx[int(np.argmax(vals))]
        cx, cy, cz = int(peak[0]), int(peak[1]), int(peak[2])
    else:
        cx, cy, cz = gt.shape[0] // 2, gt.shape[1] // 2, gt.shape[2] // 2

    half = int(max(patch_size // 2, 8))
    x0 = max(cx - half, 0)
    x1 = min(cx + half, gt.shape[0])
    y0 = max(cy - half, 0)
    y1 = min(cy + half, gt.shape[1])

    nonac_zoom = nonac[x0:x1, y0:y1, cz]
    pred_zoom = pred[x0:x1, y0:y1, cz]
    gt_zoom = gt[x0:x1, y0:y1, cz]
    err_zoom = np.abs(pred_zoom - gt_zoom)

    vmin = 0.0
    vmax = max(np.percentile(gt, 99.5), 1e-6)
    emax = max(np.percentile(np.abs(pred - gt), 99.5), 1e-6)

    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.6), constrained_layout=True)
    panels = [nonac_zoom, pred_zoom, gt_zoom, err_zoom]
    titles = ["Zoom Input", "Zoom Pred", "Zoom Ref", "Zoom |Error|"]
    ims = []
    for i, (ax, img, title) in enumerate(zip(axes, panels, titles)):
        if i < 3:
            im = ax.imshow(img.T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(img.T, origin="lower", cmap="magma", vmin=0.0, vmax=emax)
        ims.append(im)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_img = fig.colorbar(ims[0], ax=axes[:3], fraction=0.02, pad=0.02)
    cbar_img.set_label("Normalized Intensity")
    cbar_err = fig.colorbar(ims[3], ax=[axes[3]], fraction=0.08, pad=0.02)
    cbar_err.set_label("Abs. Error")
    fig.suptitle(f"Hotspot Zoom (z={cz})", y=1.03)
    _save_figure(fig, out_path)
    plt.close(fig)
