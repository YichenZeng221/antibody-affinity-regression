"""Aggregate ANDD stratified unweighted vs tail-aware-w2 multi-seed results.

中文说明：
这个脚本不训练、不运行 inference，只读取各 run 在 test set 上生成的
`checkpoint_comparison.csv`。汇总时按相同 validation checkpoint policy 比较，
主结论默认关注 `best_val_tail_mae`，因为这是本次 tail 行为实验的预注册选择口径。
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/seqproft_xdg_cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/seqproft_matplotlib_cache")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/andd_antibody_v2_stratified/multiseed"
SUMMARY_CSV = OUTPUT_DIR / "multiseed_summary.csv"
SUMMARY_MD = OUTPUT_DIR / "multiseed_summary.md"
FIGURE_PATH = ROOT / "outputs/final_reports/figures/multiseed_w2_vs_baseline.png"
EXPECTED_SEEDS = [42, 123, 2026]
POLICIES = {
    "best_val_mae": "best val MAE",
    "best_val_spearman": "best val Spearman",
    "best_val_spread": "best val spread",
    "best_val_tail_mae": "best val tail MAE",
}
METRICS = [
    "MAE",
    "RMSE",
    "Spearman",
    "pred_std_true_std",
    "error_vs_true_Pearson",
    "below_train_p10_MAE",
    "above_train_p90_MAE",
    "tail_MAE",
]
RUN_SPECS = [
    {
        "group": "unweighted",
        "seed": 42,
        "prefix": "Unweighted s42",
        "path": OUTPUT_DIR / "unweighted_seed42/checkpoint_comparison.csv",
    },
    {
        "group": "unweighted",
        "seed": 123,
        "prefix": "Unweighted s123",
        "path": OUTPUT_DIR / "unweighted_seed123/checkpoint_comparison.csv",
    },
    {
        "group": "unweighted",
        "seed": 2026,
        "prefix": "Unweighted s2026",
        "path": OUTPUT_DIR / "unweighted_seed2026/checkpoint_comparison.csv",
    },
    {
        "group": "tailaware_w2",
        "seed": 42,
        "prefix": "Tail-aware w2",
        "path": ROOT
        / "outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_tailaware_w2/"
        / "tailaware_w2_checkpoint_comparison.csv",
    },
    {
        "group": "tailaware_w2",
        "seed": 123,
        "prefix": "Tail-aware w2 s123",
        "path": OUTPUT_DIR / "tailaware_w2_seed123/checkpoint_comparison.csv",
    },
    {
        "group": "tailaware_w2",
        "seed": 2026,
        "prefix": "Tail-aware w2 s2026",
        "path": OUTPUT_DIR / "tailaware_w2_seed2026/checkpoint_comparison.csv",
    },
]


def read_seed_rows() -> tuple[pd.DataFrame, list[str]]:
    """Read only each run's own four selected-checkpoint rows."""

    rows = []
    missing_files = []
    for spec in RUN_SPECS:
        path = Path(spec["path"])
        if not path.exists():
            missing_files.append(str(path.relative_to(ROOT)))
            continue
        frame = pd.read_csv(path)
        for policy, suffix in POLICIES.items():
            expected_label = f"{spec['prefix']}: {suffix}"
            selected = frame.loc[frame["model"] == expected_label]
            if selected.empty:
                missing_files.append(f"{path.relative_to(ROOT)} :: {expected_label}")
                continue
            source = selected.iloc[0]
            row = {
                "summary_type": "seed",
                "group": spec["group"],
                "seed": spec["seed"],
                "policy": policy,
                "n_seeds": 1,
            }
            row.update({metric: float(source[metric]) for metric in METRICS})
            rows.append(row)
    return pd.DataFrame(rows), missing_files


def aggregate_rows(seed_frame: pd.DataFrame) -> pd.DataFrame:
    """Append mean and std rows for each model group/checkpoint policy."""

    output_rows = [seed_frame] if not seed_frame.empty else []
    summaries = []
    for (group, policy), subset in seed_frame.groupby(["group", "policy"], observed=True):
        for summary_type in ["mean", "std"]:
            row = {
                "summary_type": summary_type,
                "group": group,
                "seed": "",
                "policy": policy,
                "n_seeds": int(subset["seed"].nunique()),
            }
            values = subset[METRICS].mean() if summary_type == "mean" else subset[METRICS].std(ddof=1)
            row.update({metric: values[metric] for metric in METRICS})
            summaries.append(row)
    if summaries:
        output_rows.append(pd.DataFrame(summaries))
    if not output_rows:
        return pd.DataFrame(columns=["summary_type", "group", "seed", "policy", "n_seeds", *METRICS])
    return pd.concat(output_rows, ignore_index=True)


def mean_std_text(mean_value: float, std_value: float | None, n: int) -> str:
    """Format metric summary while showing when only one seed is available."""

    if n < 2 or std_value is None or pd.isna(std_value):
        return f"{mean_value:.4f} (n={n})"
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def primary_summary(seed_frame: pd.DataFrame) -> pd.DataFrame:
    """Extract mean/std for the predefined best-val-tail-MAE comparison."""

    primary = seed_frame.loc[seed_frame["policy"] == "best_val_tail_mae"]
    rows = []
    for group, subset in primary.groupby("group", observed=True):
        row = {"group": group, "n_seeds": int(subset["seed"].nunique())}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_std"] = float(subset[metric].std(ddof=1)) if len(subset) > 1 else None
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(primary: pd.DataFrame) -> None:
    """Visualize the same checkpoint policy across groups with seed error bars."""

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if primary.empty:
        return
    labels = primary["group"].replace({"unweighted": "Unweighted", "tailaware_w2": "Tail-aware w2"})
    colors = ["#507DBC" if group == "unweighted" else "#D66C44" for group in primary["group"]]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    plot_metrics = [
        ("MAE", "MAE (lower is better)"),
        ("Spearman", "Spearman (higher is better)"),
        ("pred_std_true_std", "Pred std / true std (closer to 1)"),
        ("tail_MAE", "P10/P90 tail MAE (lower is better)"),
    ]
    for axis, (metric, title) in zip(axes.flat, plot_metrics):
        means = primary[f"{metric}_mean"]
        errors = primary[f"{metric}_std"].fillna(0.0)
        axis.bar(labels, means, yerr=errors, capsize=5, color=colors, alpha=0.88)
        if metric == "pred_std_true_std":
            axis.axhline(1.0, linestyle="--", linewidth=1, color="#444444")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    n_text = ", ".join(f"{row.group}: n={int(row.n_seeds)}" for row in primary.itertuples())
    figure.suptitle(
        "ANDD stratified multi-seed: best validation tail-MAE checkpoints\n" + n_text,
        fontsize=13,
        fontweight="bold",
    )
    figure.savefig(FIGURE_PATH, dpi=300)
    plt.close(figure)


def write_report(seed_frame: pd.DataFrame, missing_files: list[str]) -> None:
    """Write transparent status/report; incomplete runs are not silently averaged."""

    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    primary = primary_summary(seed_frame)
    lines = [
        "# ANDD Stratified Multi-Seed Validation: Unweighted vs Tail-Aware W2",
        "",
        "## Comparison Design",
        "",
        "- Dataset and split are fixed: `expanded_affinity_antibody_v2_stratified`.",
        "- Architecture is fixed: all-CDR cross-attention.",
        "- Training settings are fixed: `lr=3e-5`, `epochs=20`, `batch_size=1`.",
        "- Seeds: `42`, `123`, `2026`.",
        "- Unweighted control uses identical code with `tail_weight=regular_weight=1.0`.",
        "- Tail-aware w2 uses train-P10/P90 tail weight `2.0` and regular weight `1.0`.",
        "- Primary comparison policy: **best validation tail MAE checkpoint**.",
        "- Historical unweighted seed-42 `epochs=10` result is not used in the formal multi-seed aggregate.",
        "",
        "## Completion Status",
        "",
    ]
    for group in ["unweighted", "tailaware_w2"]:
        present = sorted(seed_frame.loc[seed_frame["group"] == group, "seed"].dropna().unique().tolist())
        lines.append(f"- `{group}` completed/evaluated seeds found: `{present}` / expected `{EXPECTED_SEEDS}`.")
    if missing_files:
        lines.extend(["", "Missing outputs (run training and evaluation before final interpretation):", ""])
        lines.extend([f"- `{item}`" for item in missing_files])
    if not primary.empty:
        lines.extend(
            [
                "",
                "## Primary Policy Summary: Best Validation Tail MAE",
                "",
                "| group | n seeds | MAE | RMSE | Spearman | pred std / true std | error vs true Pearson | below P10 MAE | above P90 MAE | tail MAE |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in primary.itertuples():
            n = int(row.n_seeds)
            lines.append(
                f"| `{row.group}` | {n} | {mean_std_text(row.MAE_mean, row.MAE_std, n)} | "
                f"{mean_std_text(row.RMSE_mean, row.RMSE_std, n)} | "
                f"{mean_std_text(row.Spearman_mean, row.Spearman_std, n)} | "
                f"{mean_std_text(row.pred_std_true_std_mean, row.pred_std_true_std_std, n)} | "
                f"{mean_std_text(row.error_vs_true_Pearson_mean, row.error_vs_true_Pearson_std, n)} | "
                f"{mean_std_text(row.below_train_p10_MAE_mean, row.below_train_p10_MAE_std, n)} | "
                f"{mean_std_text(row.above_train_p90_MAE_mean, row.above_train_p90_MAE_std, n)} | "
                f"{mean_std_text(row.tail_MAE_mean, row.tail_MAE_std, n)} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "Only after all three seeds are present should we claim stability. A convincing w2 result should preserve lower tail MAE, healthier prediction spread, and a Pearson error trend closer to zero without consistently worsening overall MAE or Spearman.",
            "",
            "## Outputs",
            "",
            f"- CSV: `{SUMMARY_CSV.relative_to(ROOT)}`",
            f"- Figure: `{FIGURE_PATH.relative_to(ROOT)}`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Aggregate any completed evaluated runs and state missing ones explicitly."""

    seed_frame, missing_files = read_seed_rows()
    combined = aggregate_rows(seed_frame)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(SUMMARY_CSV, index=False)
    save_figure(primary_summary(seed_frame))
    write_report(seed_frame, missing_files)
    print(f"Saved multi-seed summary CSV to {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"Saved multi-seed report to {SUMMARY_MD.relative_to(ROOT)}")
    if not primary_summary(seed_frame).empty:
        print(f"Saved figure to {FIGURE_PATH.relative_to(ROOT)}")
    if missing_files:
        print(f"Missing evaluated runs: {len(missing_files)} entries; report remains provisional.")


if __name__ == "__main__":
    main()
