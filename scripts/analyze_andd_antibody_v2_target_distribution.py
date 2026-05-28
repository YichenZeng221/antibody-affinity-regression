""" ANDD antibody v2  target distribution,

:
affinity regression  label  `neg_log10_affinity_candidate`
,;
 train/val/test , split 

 CSV,, dataset
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

#  PNG,
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DATA_DIR = ROOT / "data/processed_affinity/expanded_affinity_antibody_v2_cdr_annotated"
OUTPUT_DIR = ROOT / "outputs/andd_antibody_v2/target_distribution"
TARGET_COLUMN = "neg_log10_affinity_candidate"
SPLITS = ["train", "val", "test"]
QUANTILES = [0.0, 0.10, 0.25, 1 / 3, 0.50, 2 / 3, 0.75, 0.90, 1.0]
BIN_LABELS = ["low_target", "mid_target", "high_target"]
COLORS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}


def load_splits() -> dict[str, pd.DataFrame]:
    """ split, regression target """

    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = DATA_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing input CSV: {path}")
        frame = pd.read_csv(path)
        if TARGET_COLUMN not in frame.columns:
            raise KeyError(f"{path.name} is missing target column: {TARGET_COLUMN}")
        frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        if frame[TARGET_COLUMN].isna().any():
            missing_count = int(frame[TARGET_COLUMN].isna().sum())
            raise ValueError(f"{path.name} contains {missing_count} non-numeric/missing target values.")
        frames[split] = frame
    return frames


def summarize_targets(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, dict]]:
    """ split  count / mean / std /  quantiles"""

    rows: list[dict] = []
    json_summary: dict[str, dict] = {}
    for split, frame in frames.items():
        target = frame[TARGET_COLUMN]
        values = {
            "count": int(target.count()),
            "mean": float(target.mean()),
            "std": float(target.std()),
            "min": float(target.min()),
            "max": float(target.max()),
        }
        quantile_values = target.quantile(QUANTILES)
        for quantile, value in quantile_values.items():
            label = f"q{quantile * 100:.1f}".replace(".", "_")
            values[label] = float(value)
        rows.append({"split": split, **values})
        json_summary[split] = values
    return pd.DataFrame(rows), json_summary


def assign_train_tertile_bin(target: pd.Series, low_edge: float, high_edge: float) -> pd.Series:
    """ train  split  bin, test"""

    return pd.cut(
        target,
        bins=[-np.inf, low_edge, high_edge, np.inf],
        labels=BIN_LABELS,
        include_lowest=True,
        right=True,
    )


def summarize_train_bins(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, dict], float, float]:
    """ split  low/mid/high """

    train_target = frames["train"][TARGET_COLUMN]
    low_edge = float(train_target.quantile(1 / 3))
    high_edge = float(train_target.quantile(2 / 3))
    if low_edge >= high_edge:
        raise ValueError("Train tertile thresholds are identical; cannot create three target bins.")

    rows: list[dict] = []
    json_summary: dict[str, dict] = {}
    for split, frame in frames.items():
        bins = assign_train_tertile_bin(frame[TARGET_COLUMN], low_edge, high_edge)
        counts = bins.value_counts(sort=False).reindex(BIN_LABELS, fill_value=0)
        json_summary[split] = {}
        for bin_name in BIN_LABELS:
            count = int(counts[bin_name])
            proportion = float(count / len(frame))
            rows.append(
                {
                    "split": split,
                    "target_bin": bin_name,
                    "count": count,
                    "proportion": proportion,
                }
            )
            json_summary[split][bin_name] = {"count": count, "proportion": proportion}
    return pd.DataFrame(rows), json_summary, low_edge, high_edge


def summarize_extreme_tails(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, float, float]:
    """ train  10% , tertile  extreme tails"""

    lower_edge = float(frames["train"][TARGET_COLUMN].quantile(0.10))
    upper_edge = float(frames["train"][TARGET_COLUMN].quantile(0.90))
    rows: list[dict] = []
    for split, frame in frames.items():
        target = frame[TARGET_COLUMN]
        low_count = int((target <= lower_edge).sum())
        high_count = int((target >= upper_edge).sum())
        rows.extend(
            [
                {
                    "split": split,
                    "tail": "bottom_10pct_by_train",
                    "threshold": lower_edge,
                    "count": low_count,
                    "proportion": low_count / len(frame),
                },
                {
                    "split": split,
                    "tail": "top_10pct_by_train",
                    "threshold": upper_edge,
                    "count": high_count,
                    "proportion": high_count / len(frame),
                },
            ]
        )
    return pd.DataFrame(rows), lower_edge, upper_edge


def plot_target_histogram(frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    """ bin edges  train/val/test target """

    all_targets = pd.concat([frames[split][TARGET_COLUMN] for split in SPLITS], ignore_index=True)
    global_min = float(all_targets.min())
    global_max = float(all_targets.max())
    shared_bins = np.linspace(global_min, global_max, 21)

    plt.figure(figsize=(8, 5))
    for split in SPLITS:
        plt.hist(
            frames[split][TARGET_COLUMN],
            bins=shared_bins,
            density=True,
            alpha=0.42,
            color=COLORS[split],
            label=f"{split} (n={len(frames[split])})",
        )
    #  split  bin edges,
    plt.xlabel("neg_log10_affinity_candidate")
    plt.ylabel("Density")
    plt.title("ANDD antibody v2 target distribution (shared histogram bins)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    """ Markdown package """

    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.{float_digits}f}")
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def write_report(
    target_summary: pd.DataFrame,
    bin_summary: pd.DataFrame,
    tail_summary: pd.DataFrame,
    low_edge: float,
    high_edge: float,
    lower_tail_edge: float,
    upper_tail_edge: float,
) -> None:
    """ Markdown """

    bin_display = bin_summary.copy()
    bin_display["proportion"] = bin_display["proportion"] * 100
    bin_display = bin_display.rename(columns={"proportion": "proportion_pct"})
    tail_display = tail_summary.copy()
    tail_display["proportion"] = tail_display["proportion"] * 100
    tail_display = tail_display.rename(columns={"proportion": "proportion_pct"})

    train_bins = bin_summary[bin_summary["split"] == "train"].set_index("target_bin")
    low_pct = float(train_bins.loc["low_target", "proportion"] * 100)
    high_pct = float(train_bins.loc["high_target", "proportion"] * 100)

    lines = [
        "# ANDD Antibody v2 Target Distribution Audit",
        "",
        "## Scope",
        "",
        "-  CDR-annotated train/val/test CSV;,",
        f"- Target column: `{TARGET_COLUMN}`",
        "- : label  regression-to-the-mean",
        "",
        "## 1. Target Summary",
        "",
        markdown_table(target_summary),
        "",
        "## 2. Quantile / Range Interpretation",
        "",
        f"- Train target  `{target_summary.loc[target_summary['split'] == 'train', 'min'].iloc[0]:.4f}`  "
        f"`{target_summary.loc[target_summary['split'] == 'train', 'max'].iloc[0]:.4f}`;"
        f"test  `{target_summary.loc[target_summary['split'] == 'test', 'min'].iloc[0]:.4f}`  "
        f"`{target_summary.loc[target_summary['split'] == 'test', 'max'].iloc[0]:.4f}`",
        "- Train  val/test  high-target tail,",
        "",
        "## 3. Low / Mid / High Bins Defined From Train Tertiles",
        "",
        " train  split ;test :",
        "",
        f"- `low_target`: target <= `{low_edge:.4f}`",
        f"- `mid_target`: `{low_edge:.4f}` < target <= `{high_edge:.4f}`",
        f"- `high_target`: target > `{high_edge:.4f}`",
        "",
        markdown_table(bin_display),
        "",
        ": train tertiles ,train "
        " val/test  target distribution shift,",
        "",
        "## 4. Extreme Tail Context",
        "",
        f", train  P10 (`{lower_tail_edge:.4f}`)  P90 (`{upper_tail_edge:.4f}`) :",
        "",
        markdown_table(tail_display),
        "",
        "## 5. Does Target Imbalance Explain Regression-To-The-Mean?",
        "",
        f"-  tertile ,train  low/high  `{low_pct:.1f}%` / `{high_pct:.1f}%`, mid",
        "- val/test  low/mid/high  train ;, prediction range  label imbalance",
        "-  target ; affinity ,low/high ",
        "",
        "## 6. Modeling Implication",
        "",
        "-  low/high  train , sampling  weighting, Huber loss ",
        "-  train tertiles  low/high `HuberLoss` ,,",
        "- : post-hoc calibration weighting/checkpoint policy,"
        " binding interaction ;Huber  label ",
        "",
        "## Files",
        "",
        "- `target_summary.csv`: count / mean / std / range / quantiles",
        "- `target_bin_counts.csv`: train-tertile low/mid/high counts and proportions",
        "- `extreme_tail_counts.csv`: train P10/P90 tail context",
        "- `target_distribution_histogram.png`: shared-bin target histogram",
    ]
    (OUTPUT_DIR / "target_distribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_splits()
    target_summary, json_summary = summarize_targets(frames)
    bin_summary, bin_json, low_edge, high_edge = summarize_train_bins(frames)
    tail_summary, lower_tail_edge, upper_tail_edge = summarize_extreme_tails(frames)

    target_summary.to_csv(OUTPUT_DIR / "target_summary.csv", index=False)
    bin_summary.to_csv(OUTPUT_DIR / "target_bin_counts.csv", index=False)
    tail_summary.to_csv(OUTPUT_DIR / "extreme_tail_counts.csv", index=False)
    plot_target_histogram(frames, OUTPUT_DIR / "target_distribution_histogram.png")
    write_report(
        target_summary,
        bin_summary,
        tail_summary,
        low_edge,
        high_edge,
        lower_tail_edge,
        upper_tail_edge,
    )

    summary_json = {
        "target_column": TARGET_COLUMN,
        "train_tertile_edges": {"low_upper": low_edge, "mid_upper": high_edge},
        "train_tail_edges": {"p10": lower_tail_edge, "p90": upper_tail_edge},
        "target_summary": json_summary,
        "target_bins": bin_json,
    }
    (OUTPUT_DIR / "target_distribution_summary.json").write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )

    print("ANDD antibody v2 target distribution audit completed.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train tertile edges: low <= {low_edge:.4f}, high > {high_edge:.4f}")
    print(":train tertiles  train low/mid/high ; tail counts ")


if __name__ == "__main__":
    main()
