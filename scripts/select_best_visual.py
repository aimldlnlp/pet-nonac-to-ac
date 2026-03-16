#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _safe_get(summary, key):
    return float(summary.get("mean", {}).get(key, 0.0))


def _minmax(values):
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def load_runs(runs_root: Path, prefix: str):
    rows = []
    for d in sorted(runs_root.glob(f"{prefix}*")):
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            continue
        rows.append(
            {
                "run_dir": d,
                "summary_path": summary_path,
                "psnr": _safe_get(summary, "psnr"),
                "ssim": _safe_get(summary, "ssim"),
                "nrmse": _safe_get(summary, "nrmse"),
            }
        )
    return rows


def rank_runs(rows, w_psnr=0.4, w_ssim=0.4, w_nrmse=0.2):
    psnr_norm = _minmax([r["psnr"] for r in rows])
    ssim_norm = _minmax([r["ssim"] for r in rows])
    nrmse_inv_norm = _minmax([-r["nrmse"] for r in rows])  # smaller nrmse is better

    ranked = []
    for i, r in enumerate(rows):
        score = (
            w_psnr * psnr_norm[i]
            + w_ssim * ssim_norm[i]
            + w_nrmse * nrmse_inv_norm[i]
        )
        ranked.append({**r, "score": score})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def main():
    ap = argparse.ArgumentParser(
        description="Select best visual run using PSNR/SSIM/NRMSE from summary.json."
    )
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--prefix", default="best_visual_")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--w_psnr", type=float, default=0.4)
    ap.add_argument("--w_ssim", type=float, default=0.4)
    ap.add_argument("--w_nrmse", type=float, default=0.2)
    ap.add_argument(
        "--mark_file",
        default="runs/best_visual_selected.txt",
        help="Write best run path to this file.",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    rows = load_runs(runs_root, args.prefix)
    if not rows:
        print("No runs found. Expected folders like runs/best_visual_*/summary.json")
        return

    ranked = rank_runs(
        rows,
        w_psnr=args.w_psnr,
        w_ssim=args.w_ssim,
        w_nrmse=args.w_nrmse,
    )

    print("Rank | Score  | PSNR    | SSIM    | NRMSE   | Run")
    print("-----+--------+---------+---------+---------+-------------------------------")
    for i, r in enumerate(ranked[: max(args.top_k, 1)], start=1):
        print(
            f"{i:>4} | {r['score']:.4f} | {r['psnr']:.4f} | "
            f"{r['ssim']:.4f} | {r['nrmse']:.4f} | {r['run_dir']}"
        )

    best = ranked[0]["run_dir"]
    mark_file = Path(args.mark_file)
    mark_file.parent.mkdir(parents=True, exist_ok=True)
    mark_file.write_text(str(best) + "\n", encoding="utf-8")
    print(f"\nBest run: {best}")
    print(f"Marked in: {mark_file}")


if __name__ == "__main__":
    main()
