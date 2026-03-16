import torch
import torch.nn.functional as F


def gradient_loss(pred, target):
    def grad(x):
        dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return dx, dy, dz

    pdx, pdy, pdz = grad(pred)
    tdx, tdy, tdz = grad(target)
    return (
        F.l1_loss(pdx, tdx)
        + F.l1_loss(pdy, tdy)
        + F.l1_loss(pdz, tdz)
    ) / 3.0


def total_loss(pred, target, w_l1=0.8, w_grad=0.2):
    l1 = F.l1_loss(pred, target)
    gl = gradient_loss(pred, target)
    return w_l1 * l1 + w_grad * gl


def weighted_high_intensity_l1(pred, target, threshold=0.8, boost=3.0):
    base = torch.abs(pred - target)
    high_mask = (target >= threshold).float()
    weights = 1.0 + boost * high_mask
    return torch.mean(base * weights)


def high_uptake_mask(target, quantile=0.99, min_voxels=256):
    b, c, *_ = target.shape
    flat = target.reshape(b, c, -1)
    mask = torch.zeros_like(flat)

    q = float(max(min(quantile, 0.9999), 0.5))
    min_voxels = int(max(min_voxels, 1))

    for bi in range(b):
        for ci in range(c):
            vals = flat[bi, ci]
            positive = vals > 0
            if not torch.any(positive):
                continue

            pos_vals = vals[positive]
            thr = torch.quantile(pos_vals, q)
            roi = positive & (vals >= thr)

            if int(roi.sum().item()) < min_voxels:
                pos_idx = torch.nonzero(positive, as_tuple=False).squeeze(1)
                k = min(min_voxels, int(pos_idx.numel()))
                top_local = torch.topk(vals[pos_idx], k=k, largest=True, sorted=False).indices
                idx = pos_idx[top_local]
                mask[bi, ci, idx] = 1.0
            else:
                mask[bi, ci, roi] = 1.0

    return mask.reshape_as(target)


def relative_high_intensity_loss(pred, target, threshold=0.8, eps=1e-3):
    mask = (target >= threshold).float()
    rel = torch.abs(pred - target) / (torch.abs(target) + eps)
    denom = mask.sum()
    if denom <= 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return (rel * mask).sum() / denom


def log_high_intensity_loss(pred, target, mask, log_alpha=20.0, eps=1e-6):
    pred_log = torch.log1p(log_alpha * torch.clamp(pred, min=0.0))
    target_log = torch.log1p(log_alpha * torch.clamp(target, min=0.0))
    denom = mask.sum()
    if denom <= 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return (torch.abs(pred_log - target_log) * mask).sum() / (denom + eps)


def topk_peak_loss(pred, target, topk_frac=0.01, min_voxels=256):
    b, c, *_ = target.shape
    pred_f = pred.reshape(b, c, -1)
    target_f = target.reshape(b, c, -1)
    n = target_f.shape[-1]
    k = min(max(int(n * topk_frac), min_voxels), n)
    if k <= 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    idx = torch.topk(target_f, k=k, dim=-1, largest=True, sorted=False).indices
    pred_topk = torch.gather(pred_f, -1, idx)
    target_topk = torch.gather(target_f, -1, idx)
    return torch.mean(torch.abs(pred_topk - target_topk))


def suvpeak_surrogate_loss(pred, target, roi_mask, peak_voxels=64):
    b, c, *_ = target.shape
    pred_f = pred.reshape(b, c, -1)
    target_f = target.reshape(b, c, -1)
    mask_f = roi_mask.reshape(b, c, -1) > 0
    peak_voxels = int(max(peak_voxels, 1))

    losses = []
    for bi in range(b):
        for ci in range(c):
            m = mask_f[bi, ci]
            if not torch.any(m):
                continue
            pv = pred_f[bi, ci][m]
            tv = target_f[bi, ci][m]
            k = min(peak_voxels, int(tv.numel()))
            idx = torch.topk(tv, k=k, largest=True, sorted=False).indices
            p_peak = torch.mean(torch.gather(pv, 0, idx))
            t_peak = torch.mean(torch.gather(tv, 0, idx))
            losses.append(torch.abs(p_peak - t_peak))

    if not losses:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return torch.mean(torch.stack(losses))


def hotspot_underestimation_loss(pred, target, roi_mask, rel_eps=1e-3):
    diff = target - pred
    under = F.relu(diff)
    denom = roi_mask.sum()
    if denom <= 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    abs_term = (under * roi_mask).sum() / (denom + 1e-6)
    rel_term = ((under / (target + rel_eps)) * roi_mask).sum() / (denom + 1e-6)
    return 0.5 * abs_term + 0.5 * rel_term


def sprint1_loss(
    pred,
    target,
    w_l1=0.55,
    w_grad=0.15,
    w_hi=0.15,
    w_topk=0.10,
    w_rel=0.10,
    w_log=0.10,
    w_peak=0.10,
    w_under=0.00,
    hi_threshold=0.8,
    hi_boost=3.0,
    roi_quantile=0.99,
    roi_min_voxels=256,
    peak_voxels=64,
    topk_frac=0.01,
    topk_min_voxels=256,
    rel_eps=1e-3,
    log_alpha=20.0,
    base_clip_max=1.0,
):
    if base_clip_max is not None:
        pred_base = torch.clamp(pred, min=0.0, max=base_clip_max)
        target_base = torch.clamp(target, min=0.0, max=base_clip_max)
    else:
        pred_base = pred
        target_base = target

    l1 = F.l1_loss(pred_base, target_base)
    gl = gradient_loss(pred_base, target_base)
    hil = weighted_high_intensity_l1(pred_base, target_base, threshold=hi_threshold, boost=hi_boost)
    tkl = topk_peak_loss(pred, target, topk_frac=topk_frac, min_voxels=topk_min_voxels)
    rhl = relative_high_intensity_loss(pred_base, target_base, threshold=hi_threshold, eps=rel_eps)
    roi = high_uptake_mask(target, quantile=roi_quantile, min_voxels=roi_min_voxels)
    lhl = log_high_intensity_loss(pred, target, roi, log_alpha=log_alpha)
    psl = suvpeak_surrogate_loss(pred, target, roi, peak_voxels=peak_voxels)
    hul = hotspot_underestimation_loss(pred, target, roi, rel_eps=rel_eps)

    return (
        w_l1 * l1
        + w_grad * gl
        + w_hi * hil
        + w_topk * tkl
        + w_rel * rhl
        + w_log * lhl
        + w_peak * psl
        + w_under * hul
    )
