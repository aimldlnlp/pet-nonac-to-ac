import argparse
import json
import math
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


PROFILES = {
    "global_v1": {
        "description": "Global image quality and quantification gate.",
        "mean_rules": [
            {"id": "mean_psnr", "metric": "psnr", "op": "ge", "threshold": 31.0, "ci_required": True},
            {"id": "mean_ssim", "metric": "ssim", "op": "ge", "threshold": 0.91, "ci_required": True},
            {"id": "mean_nrmse", "metric": "nrmse", "op": "le", "threshold": 0.07, "ci_required": True},
            {
                "id": "mean_suvmean_err",
                "metric": "suvmean_err_pct",
                "op": "le",
                "threshold": 3.0,
                "ci_required": True,
            },
            {
                "id": "mean_psnr_gain",
                "metric": "psnr_gain_db",
                "op": "ge",
                "threshold": 4.5,
                "ci_required": True,
            },
        ],
        "pass_rate_rules": [
            {
                "id": "rate_ssim_090",
                "metric": "ssim",
                "op": "ge",
                "threshold": 0.90,
                "min_rate": 0.80,
            },
            {
                "id": "rate_psnr_gain_3db",
                "metric": "psnr_gain_db",
                "op": "ge",
                "threshold": 3.0,
                "min_rate": 0.90,
            },
            {
                "id": "rate_suvmean_8pct",
                "metric": "suvmean_err_pct",
                "op": "le",
                "threshold": 8.0,
                "min_rate": 0.95,
            },
        ],
    },
    "balanced_v1": {
        "description": "Global quality + hotspot fidelity gate (strict).",
        "mean_rules": [
            {"id": "mean_psnr", "metric": "psnr", "op": "ge", "threshold": 31.0, "ci_required": True},
            {"id": "mean_ssim", "metric": "ssim", "op": "ge", "threshold": 0.91, "ci_required": True},
            {"id": "mean_nrmse", "metric": "nrmse", "op": "le", "threshold": 0.07, "ci_required": True},
            {
                "id": "mean_suvmean_err",
                "metric": "suvmean_err_pct",
                "op": "le",
                "threshold": 3.0,
                "ci_required": True,
            },
            {
                "id": "mean_psnr_gain",
                "metric": "psnr_gain_db",
                "op": "ge",
                "threshold": 4.5,
                "ci_required": True,
            },
            {
                "id": "mean_suvmax_err",
                "metric": "suvmax_err_pct",
                "op": "le",
                "threshold": 35.0,
                "ci_required": True,
            },
            {
                "id": "mean_suvpeak_err",
                "metric": "suvpeak_err_pct",
                "op": "le",
                "threshold": 35.0,
                "ci_required": True,
            },
        ],
        "pass_rate_rules": [
            {
                "id": "rate_ssim_090",
                "metric": "ssim",
                "op": "ge",
                "threshold": 0.90,
                "min_rate": 0.80,
            },
            {
                "id": "rate_psnr_gain_3db",
                "metric": "psnr_gain_db",
                "op": "ge",
                "threshold": 3.0,
                "min_rate": 0.90,
            },
            {
                "id": "rate_suvmean_8pct",
                "metric": "suvmean_err_pct",
                "op": "le",
                "threshold": 8.0,
                "min_rate": 0.95,
            },
            {
                "id": "rate_suvmax_35pct",
                "metric": "suvmax_err_pct",
                "op": "le",
                "threshold": 35.0,
                "min_rate": 0.80,
            },
            {
                "id": "rate_suvpeak_35pct",
                "metric": "suvpeak_err_pct",
                "op": "le",
                "threshold": 35.0,
                "min_rate": 0.80,
            },
        ],
    },
}


def _compare(value, op, threshold):
    if op == "ge":
        return value >= threshold
    if op == "le":
        return value <= threshold
    raise ValueError(f"Unsupported op: {op}")


def _bootstrap_ci_mean(values, n_boot, alpha, rng):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1.0 - alpha / 2))
    return lo, hi


def _discover_csvs(csv_args, glob_args):
    paths = []
    for p in csv_args:
        paths.append(Path(p))
    for pat in glob_args:
        for p in sorted(glob(pat)):
            paths.append(Path(p))

    if not paths:
        default = Path("reports/metrics_test.csv")
        if default.exists():
            paths = [default]

    dedup = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        dedup.append(rp)

    if not dedup:
        raise FileNotFoundError("No input CSV found. Use --csv or --glob.")
    for p in dedup:
        if not p.exists():
            raise FileNotFoundError(f"Input CSV not found: {p}")
    return dedup


def _derive_run_id(path, idx, used):
    path = Path(path)
    if path.parent.name == "reports" and path.parent.parent.name:
        base = path.parent.parent.name
    else:
        base = path.stem

    base = base.replace(" ", "_")
    run_id = base
    k = 2
    while run_id in used:
        run_id = f"{base}_{k}"
        k += 1
    used.add(run_id)
    return run_id


def _load_all_csvs(csv_paths):
    used_ids = set()
    frames = []
    source_map = {}

    for i, csv_path in enumerate(csv_paths):
        run_id = _derive_run_id(csv_path, i, used_ids)
        df = pd.read_csv(csv_path)
        if "case_id" not in df.columns:
            df["case_id"] = [f"case_{j:03d}" for j in range(len(df))]

        for c in df.columns:
            if c != "case_id":
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df["run_id"] = run_id
        df["source_csv"] = str(csv_path)
        frames.append(df)
        source_map[run_id] = str(csv_path)

    all_df = pd.concat(frames, ignore_index=True)
    return all_df, source_map


def _numeric_metrics(df):
    blocked = {"case_id", "run_id", "source_csv"}
    metrics = []
    for c in df.columns:
        if c in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            metrics.append(c)
    return metrics


def _build_per_run_summary(df, metrics, source_map):
    rows = []
    for run_id, g in df.groupby("run_id", sort=True):
        row = {"run_id": run_id, "source_csv": source_map.get(run_id, ""), "num_cases": int(len(g))}
        for m in metrics:
            vals = g[m].dropna().to_numpy(np.float64)
            row[f"{m}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{m}_std"] = float(np.std(vals, ddof=0)) if vals.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _build_metric_stats(df, per_run_summary, metrics, n_boot, alpha, seed):
    rng = np.random.default_rng(seed)
    stats = {}

    for m in metrics:
        vals = df[m].dropna().to_numpy(np.float64)
        case_mean = float(np.mean(vals)) if vals.size else float("nan")
        case_std = float(np.std(vals, ddof=0)) if vals.size else float("nan")
        ci_low, ci_high = _bootstrap_ci_mean(vals, n_boot=n_boot, alpha=alpha, rng=rng)

        run_mean_col = f"{m}_mean"
        if run_mean_col in per_run_summary.columns and len(per_run_summary) > 0:
            run_vals = per_run_summary[run_mean_col].dropna().to_numpy(np.float64)
        else:
            run_vals = np.array([], dtype=np.float64)
        run_mean = float(np.mean(run_vals)) if run_vals.size else float("nan")
        run_std = float(np.std(run_vals, ddof=0)) if run_vals.size else float("nan")
        run_ci_low, run_ci_high = _bootstrap_ci_mean(run_vals, n_boot=n_boot, alpha=alpha, rng=rng)

        stats[m] = {
            "n_cases": int(vals.size),
            "case_mean": case_mean,
            "case_std": case_std,
            "case_ci95": {"low": ci_low, "high": ci_high},
            "n_runs": int(run_vals.size),
            "run_mean_of_means": run_mean,
            "run_std_of_means": run_std,
            "run_ci95_of_means": {"low": run_ci_low, "high": run_ci_high},
        }

    return stats


def _eval_rule_mean(rule, metric_stats, allow_missing):
    metric = rule["metric"]
    if metric not in metric_stats:
        return {
            "id": rule["id"],
            "type": "mean",
            "metric": metric,
            "status": "skip" if allow_missing else "fail",
            "reason": "metric_missing",
        }

    st = metric_stats[metric]
    point = st["case_mean"]
    point_pass = bool(_compare(point, rule["op"], rule["threshold"]))
    ci_required = bool(rule.get("ci_required", False))
    ci_check = True
    ci_value = None
    if ci_required:
        if rule["op"] == "ge":
            ci_value = st["case_ci95"]["low"]
        else:
            ci_value = st["case_ci95"]["high"]
        ci_check = bool(_compare(ci_value, rule["op"], rule["threshold"]))

    passed = point_pass and ci_check
    return {
        "id": rule["id"],
        "type": "mean",
        "metric": metric,
        "op": rule["op"],
        "threshold": float(rule["threshold"]),
        "value": float(point),
        "ci_required": ci_required,
        "ci_value": float(ci_value) if ci_value is not None else None,
        "status": "pass" if passed else "fail",
        "reason": "ok" if passed else ("ci_fail" if ci_required and point_pass else "value_fail"),
    }


def _eval_rule_pass_rate(rule, df, allow_missing):
    metric = rule["metric"]
    if metric not in df.columns:
        return {
            "id": rule["id"],
            "type": "pass_rate",
            "metric": metric,
            "status": "skip" if allow_missing else "fail",
            "reason": "metric_missing",
        }

    vals = df[metric].dropna().to_numpy(np.float64)
    if vals.size == 0:
        return {
            "id": rule["id"],
            "type": "pass_rate",
            "metric": metric,
            "status": "skip" if allow_missing else "fail",
            "reason": "no_values",
        }

    hit = vals >= rule["threshold"] if rule["op"] == "ge" else vals <= rule["threshold"]
    rate = float(np.mean(hit))
    passed = rate >= float(rule["min_rate"])
    return {
        "id": rule["id"],
        "type": "pass_rate",
        "metric": metric,
        "op": rule["op"],
        "threshold": float(rule["threshold"]),
        "min_rate": float(rule["min_rate"]),
        "rate": rate,
        "passed_cases": int(hit.sum()),
        "total_cases": int(vals.size),
        "status": "pass" if passed else "fail",
        "reason": "ok" if passed else "rate_fail",
    }


def _evaluate_profile(profile_cfg, metric_stats, df, allow_missing):
    results = []
    for rule in profile_cfg.get("mean_rules", []):
        results.append(_eval_rule_mean(rule, metric_stats, allow_missing=allow_missing))
    for rule in profile_cfg.get("pass_rate_rules", []):
        results.append(_eval_rule_pass_rate(rule, df, allow_missing=allow_missing))

    checked = [r for r in results if r["status"] != "skip"]
    has_fail = any(r["status"] == "fail" for r in checked)
    overall_pass = bool(checked) and (not has_fail)
    return results, overall_pass


def _metric_stats_frame(metric_stats):
    rows = []
    for metric, st in metric_stats.items():
        rows.append(
            {
                "metric": metric,
                "n_cases": st["n_cases"],
                "case_mean": st["case_mean"],
                "case_std": st["case_std"],
                "case_ci95_low": st["case_ci95"]["low"],
                "case_ci95_high": st["case_ci95"]["high"],
                "n_runs": st["n_runs"],
                "run_mean_of_means": st["run_mean_of_means"],
                "run_std_of_means": st["run_std_of_means"],
                "run_ci95_low_of_means": st["run_ci95_of_means"]["low"],
                "run_ci95_high_of_means": st["run_ci95_of_means"]["high"],
            }
        )
    return pd.DataFrame(rows)


def _fmt(v, digits=4):
    if v is None:
        return "-"
    if isinstance(v, (float, int)) and (math.isnan(v) or math.isinf(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _write_markdown(path, payload, profile_name, alpha):
    lines = []
    lines.append("# Verification Gate Report")
    lines.append("")
    lines.append(f"- Profile: `{profile_name}`")
    lines.append(f"- Description: {payload['profile_description']}")
    lines.append(f"- Num runs: {payload['num_runs']}")
    lines.append(f"- Num cases total: {payload['num_cases_total']}")
    lines.append(f"- Bootstrap iterations: {payload['bootstrap_iterations']}")
    lines.append(f"- CI level: {int((1.0 - alpha) * 100)}%")
    lines.append(f"- Final decision: `{'PASS' if payload['final_pass'] else 'FAIL'}`")
    lines.append("")
    lines.append("## Rule Results")
    lines.append("")
    lines.append("| id | type | metric | status | detail |")
    lines.append("|---|---|---|---|---|")

    for r in payload["rules"]:
        if r["type"] == "mean":
            detail = (
                f"value={_fmt(r.get('value'))} "
                f"{r.get('op', '')} {r.get('threshold', '')}"
            )
            if r.get("ci_required", False):
                detail += f", ci_ref={_fmt(r.get('ci_value'))}"
        elif r["type"] == "pass_rate":
            detail = (
                f"rate={_fmt(r.get('rate'))}, "
                f"target>={_fmt(r.get('min_rate'))}, "
                f"cond: {r.get('op', '')} {r.get('threshold', '')}"
            )
        else:
            detail = r.get("reason", "")
        lines.append(
            f"| {r.get('id', '-')} | {r.get('type', '-')} | {r.get('metric', '-')} | "
            f"{r.get('status', '-').upper()} | {detail} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Objective verification gate for PET Non-AC -> AC metrics.")
    ap.add_argument("--csv", nargs="*", default=[], help="One or more metrics CSV paths.")
    ap.add_argument("--glob", nargs="*", default=[], help="Glob patterns for metrics CSV paths.")
    ap.add_argument("--profile", default="balanced_v1", choices=sorted(PROFILES.keys()))
    ap.add_argument("--out_dir", default="reports/verification")
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min_runs", type=int, default=1)
    ap.add_argument(
        "--allow_missing_metrics",
        action="store_true",
        help="Skip missing-metric rules instead of failing them.",
    )
    args = ap.parse_args()

    csv_paths = _discover_csvs(args.csv, args.glob)
    df, source_map = _load_all_csvs(csv_paths)
    metrics = _numeric_metrics(df)
    per_run_summary = _build_per_run_summary(df, metrics, source_map)
    metric_stats = _build_metric_stats(
        df=df,
        per_run_summary=per_run_summary,
        metrics=metrics,
        n_boot=args.bootstrap,
        alpha=args.alpha,
        seed=args.seed,
    )

    profile_cfg = PROFILES[args.profile]
    rules, rules_pass = _evaluate_profile(
        profile_cfg=profile_cfg,
        metric_stats=metric_stats,
        df=df,
        allow_missing=args.allow_missing_metrics,
    )

    run_count_ok = int(df["run_id"].nunique()) >= int(args.min_runs)
    final_pass = bool(rules_pass and run_count_ok)
    if not run_count_ok:
        rules.append(
            {
                "id": "min_runs_guard",
                "type": "meta",
                "metric": "run_count",
                "status": "fail",
                "reason": f"need_at_least_{args.min_runs}_runs",
                "value": int(df["run_id"].nunique()),
                "threshold": int(args.min_runs),
            }
        )

    payload = {
        "profile": args.profile,
        "profile_description": profile_cfg["description"],
        "inputs": [str(p) for p in csv_paths],
        "num_runs": int(df["run_id"].nunique()),
        "num_cases_total": int(len(df)),
        "bootstrap_iterations": int(args.bootstrap),
        "alpha": float(args.alpha),
        "rules": rules,
        "metric_stats": metric_stats,
        "final_pass": final_pass,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "gate_report.json"
    md_path = out_dir / "gate_report.md"
    per_run_csv_path = out_dir / "per_run_summary.csv"
    metric_csv_path = out_dir / "aggregate_metrics.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _write_markdown(md_path, payload, profile_name=args.profile, alpha=args.alpha)
    per_run_summary.to_csv(per_run_csv_path, index=False)
    _metric_stats_frame(metric_stats).to_csv(metric_csv_path, index=False)

    print(f"Verification report JSON: {json_path}")
    print(f"Verification report MD:   {md_path}")
    print(f"Per-run summary CSV:      {per_run_csv_path}")
    print(f"Aggregate metrics CSV:    {metric_csv_path}")
    print(f"Final decision: {'PASS' if final_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
