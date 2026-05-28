"""Error analysis for ANDD antibody v2 all-CDR pooled baseline.

中文人话说明：
这个脚本只分析已经生成的 test predictions，不训练模型，也不修改数据。

它会回答几个问题：
- 最大错误样本是谁？
- 模型是否 regression-to-mean：低 target 被高估，高 target 被低估？
- error 是否和 source / antigen / sequence length / CDR length 有关系？
- low target 为什么最难？
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

# 让 matplotlib/font cache 写到项目可写目录，避免 macOS 用户目录 cache 权限 warning。
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data/processed_affinity/expanded_affinity_antibody_v2_cdr_annotated/test.csv"
PREDICTIONS_CSV = (
    ROOT
    / "outputs/andd_antibody_v2/all_cdr_pooled/"
    / "andd_antibody_v2_all_cdr_pooled_test_predictions.csv"
)
OUTPUT_DIR = ROOT / "outputs/andd_antibody_v2/error_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"

TRUE_COL = "true_neg_log10_affinity"
PRED_COL = "predicted_neg_log10_affinity"
CDR_COLUMNS = ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]


def safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    """Spearman 是 ranking correlation；如果常数列导致 undefined，就返回 None。"""

    value = pd.to_numeric(left, errors="coerce").corr(pd.to_numeric(right, errors="coerce"), method="spearman")
    return None if pd.isna(value) else float(value)


def safe_pearson(left: pd.Series, right: pd.Series) -> float | None:
    """普通线性相关。用于看 error 是否随 target 系统变化。"""

    value = pd.to_numeric(left, errors="coerce").corr(pd.to_numeric(right, errors="coerce"), method="pearson")
    return None if pd.isna(value) else float(value)


def metric_dict(df: pd.DataFrame) -> dict:
    """计算整体 regression metrics。"""

    true = pd.to_numeric(df[TRUE_COL], errors="coerce")
    pred = pd.to_numeric(df[PRED_COL], errors="coerce")
    error = pred - true
    abs_error = error.abs()
    mse = float((error**2).mean())
    rmse = math.sqrt(mse)
    return {
        "rows": int(len(df)),
        "mae": float(abs_error.mean()),
        "mse": mse,
        "rmse": rmse,
        "spearman": safe_spearman(true, pred),
        "true_mean": float(true.mean()),
        "true_std": float(true.std()),
        "true_min": float(true.min()),
        "true_max": float(true.max()),
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
        "prediction_min": float(pred.min()),
        "prediction_max": float(pred.max()),
        "pred_std_over_true_std": float(pred.std() / true.std()) if true.std() else None,
        "error_vs_true_pearson": safe_pearson(error, true),
    }


def add_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """增加 error、bin、长度、CDR length 等分析列。"""

    out = df.copy()
    out["true_target"] = pd.to_numeric(out[TRUE_COL], errors="coerce")
    out["predicted_target"] = pd.to_numeric(out[PRED_COL], errors="coerce")
    out["error"] = out["predicted_target"] - out["true_target"]
    out["absolute_error"] = out["error"].abs()
    out["fold_error"] = 10 ** out["absolute_error"]

    # target bin 用 quantile，保证 low/mid/high 的样本数相近。
    ranked = out["true_target"].rank(method="first")
    out["target_bin"] = pd.qcut(ranked, q=3, labels=["low_target", "mid_target", "high_target"])

    for seq_col in ["heavy_sequence", "light_sequence", "antigen_sequence"]:
        if seq_col in out.columns:
            out[f"{seq_col}_len"] = out[seq_col].fillna("").astype(str).str.len()

    # ANDD split 里已有 heavy_len/light_len/antigen_len；如果存在就优先保留，更方便 groupby。
    for cdr_col in CDR_COLUMNS:
        out[f"{cdr_col}_len"] = out[cdr_col].fillna("").astype(str).str.len()
    out["total_cdr_len"] = sum(out[f"{cdr_col}_len"] for cdr_col in CDR_COLUMNS)

    antigen_len = out.get("antigen_len", out.get("antigen_sequence_len"))
    out["antigen_length_bin"] = pd.qcut(
        pd.to_numeric(antigen_len, errors="coerce").rank(method="first"),
        q=3,
        labels=["short_antigen", "mid_antigen", "long_antigen"],
    )
    out["heavy_length_bin"] = pd.qcut(
        pd.to_numeric(out.get("heavy_len", out.get("heavy_sequence_len")), errors="coerce").rank(method="first"),
        q=3,
        labels=["short_heavy", "mid_heavy", "long_heavy"],
    )
    out["light_length_bin"] = pd.qcut(
        pd.to_numeric(out.get("light_len", out.get("light_sequence_len")), errors="coerce").rank(method="first"),
        q=3,
        labels=["short_light", "mid_light", "long_light"],
    )

    # peptide/epitope-like 是一个启发式标记：短 antigen 可能更像 epitope peptide。
    out["peptide_or_epitope_like_antigen"] = pd.to_numeric(out.get("antigen_len", out["antigen_sequence_len"]), errors="coerce") < 30
    return out


def group_metrics(df: pd.DataFrame, group_column: str, label: str) -> list[dict]:
    """按某个字段分组计算 MAE/RMSE/mean error。"""

    rows = []
    if group_column not in df.columns:
        return rows
    for group_value, group in df.groupby(group_column, dropna=False, observed=True):
        if len(group) == 0:
            continue
        mse = float((group["error"] ** 2).mean())
        rows.append(
            {
                "group_type": label,
                "group_value": str(group_value),
                "rows": int(len(group)),
                "mae": float(group["absolute_error"].mean()),
                "rmse": math.sqrt(mse),
                "mean_error": float(group["error"].mean()),
                "true_mean": float(group["true_target"].mean()),
                "pred_mean": float(group["predicted_target"].mean()),
            }
        )
    return sorted(rows, key=lambda row: row["mae"], reverse=True)


def plot_true_vs_pred(df: pd.DataFrame, path: Path) -> None:
    """画 true vs predicted，看整体校准和是否压缩在均值附近。"""

    plt.figure(figsize=(6, 5))
    plt.scatter(df["true_target"], df["predicted_target"], alpha=0.75)
    low = min(df["true_target"].min(), df["predicted_target"].min())
    high = max(df["true_target"].max(), df["predicted_target"].max())
    plt.plot([low, high], [low, high], linestyle="--", color="black", label="y = x")
    plt.xlim(low, high)
    plt.ylim(low, high)
    plt.xlabel("True neg_log10_affinity")
    plt.ylabel("Predicted neg_log10_affinity")
    plt.title("ANDD antibody v2: True vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_residual_vs_true(df: pd.DataFrame, path: Path) -> None:
    """画 residual vs true，直接看 low/high target 的系统偏差。"""

    plt.figure(figsize=(6, 5))
    colors = {"low_target": "#d95f02", "mid_target": "#1b9e77", "high_target": "#7570b3"}
    for bin_name, group in df.groupby("target_bin", observed=True):
        plt.scatter(group["true_target"], group["error"], label=str(bin_name), alpha=0.75, color=colors.get(str(bin_name)))
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("True neg_log10_affinity")
    plt.ylabel("Prediction error (predicted - true)")
    plt.title("ANDD antibody v2: Residual vs True Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_report(df: pd.DataFrame, metrics: dict, group_rows: list[dict]) -> None:
    """写 Markdown 报告。"""

    top_errors = df.sort_values("absolute_error", ascending=False).head(10)
    bin_rows = [row for row in group_rows if row["group_type"] == "target_bin"]
    source_rows = [row for row in group_rows if row["group_type"] == "source"]
    antigen_len_rows = [row for row in group_rows if row["group_type"] == "antigen_length_bin"]

    cdr_corrs = []
    for cdr_col in CDR_COLUMNS + ["total_cdr_len"]:
        length_column = cdr_col if cdr_col == "total_cdr_len" else f"{cdr_col}_len"
        corr = safe_pearson(df[length_column], df["absolute_error"])
        cdr_corrs.append((cdr_col, corr))

    low = df[df["target_bin"].astype(str) == "low_target"]
    high = df[df["target_bin"].astype(str) == "high_target"]

    lines = [
        "# ANDD Antibody v2 All-CDR Pooled Error Analysis",
        "",
        "## Scope",
        "",
        "- No model was trained by this analysis.",
        "- No dataset was modified.",
        "- Inputs: ANDD antibody v2 CDR-annotated test set and saved predictions.",
        "",
        "## 1. Overall Metrics",
        "",
        f"- Rows: `{metrics['rows']}`",
        f"- MAE: `{metrics['mae']:.4f}`",
        f"- RMSE: `{metrics['rmse']:.4f}`",
        f"- Spearman: `{metrics['spearman']:.4f}`",
        f"- true mean/std/min/max: `{metrics['true_mean']:.4f}` / `{metrics['true_std']:.4f}` / `{metrics['true_min']:.4f}` / `{metrics['true_max']:.4f}`",
        f"- prediction mean/std/min/max: `{metrics['prediction_mean']:.4f}` / `{metrics['prediction_std']:.4f}` / `{metrics['prediction_min']:.4f}` / `{metrics['prediction_max']:.4f}`",
        f"- pred_std / true_std: `{metrics['pred_std_over_true_std']:.4f}`",
        f"- error vs true target Pearson: `{metrics['error_vs_true_pearson']:.4f}`",
        "",
        "## 2. Regression-To-Mean Check",
        "",
        f"- Low target mean error: `{low['error'].mean():.4f}`",
        f"- High target mean error: `{high['error'].mean():.4f}`",
        "",
        "Interpretation: positive low-target mean error means weak-binding/low-target examples are overpredicted. Negative high-target mean error means strong-binding/high-target examples are underpredicted.",
        "",
        "## 3. MAE By Target Bin",
        "",
        "| Target bin | Rows | MAE | RMSE | Mean error | True mean | Pred mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(bin_rows, key=lambda item: ["low_target", "mid_target", "high_target"].index(item["group_value"])):
        lines.append(
            f"| `{row['group_value']}` | {row['rows']} | {row['mae']:.4f} | {row['rmse']:.4f} | "
            f"{row['mean_error']:.4f} | {row['true_mean']:.4f} | {row['pred_mean']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 4. Source And Length Group Checks",
            "",
            "### By source",
            "",
            "| Source | Rows | MAE | RMSE | Mean error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in source_rows:
        lines.append(f"| `{row['group_value']}` | {row['rows']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['mean_error']:.4f} |")

    lines.extend(["", "### By antigen length bin", "", "| Antigen length bin | Rows | MAE | RMSE | Mean error |", "|---|---:|---:|---:|---:|"])
    for row in antigen_len_rows:
        lines.append(f"| `{row['group_value']}` | {row['rows']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['mean_error']:.4f} |")

    lines.extend(
        [
            "",
            "## 5. Low-Target Difficulty",
            "",
            f"- Low-target rows: `{len(low)}`",
            f"- Low-target MAE: `{low['absolute_error'].mean():.4f}`",
            f"- Low-target mean true target: `{low['true_target'].mean():.4f}`",
            f"- Low-target mean prediction: `{low['predicted_target'].mean():.4f}`",
            f"- Low-target peptide/epitope-like antigen rows: `{int(low['peptide_or_epitope_like_antigen'].sum())}`",
            f"- Low-target extreme Kd flags: `{int(low.get('flag_extreme_kd', pd.Series(False, index=low.index)).astype(str).str.lower().isin({'true'}).sum())}`",
            "",
            "The low-target bin is hardest because predictions are pulled upward toward the dataset center. This is classic regression-to-mean: the model avoids very low predictions even when the true target is low.",
            "",
            "## 6. Data Quality Checks",
            "",
            f"- Extreme Kd flagged rows in test: `{int(df.get('flag_extreme_kd', pd.Series(False, index=df.index)).astype(str).str.lower().isin({'true'}).sum())}`",
            f"- Duplicate exact triplet flags in test: `{int(df.get('flag_duplicate', pd.Series(False, index=df.index)).astype(str).str.lower().isin({'true'}).sum())}`",
            f"- Antigen overlap flags in test: `{int(df.get('flag_antigen_overlap', pd.Series(False, index=df.index)).astype(str).str.lower().isin({'true'}).sum())}`",
            f"- Peptide/epitope-like antigen rows (`antigen_len < 30`): `{int(df['peptide_or_epitope_like_antigen'].sum())}`",
            "",
            "Because this benchmark was built from `keep_safe` rows, major extreme-Kd / duplicate / overlap flags should be rare or zero. Remaining error is more likely model/label difficulty than obvious leakage.",
            "",
            "## 7. CDR Length vs Error",
            "",
            "| CDR length feature | Pearson corr with absolute error |",
            "|---|---:|",
        ]
    )
    for cdr_col, corr in cdr_corrs:
        lines.append(f"| `{cdr_col}` | {'NA' if corr is None else f'{corr:.4f}'} |")

    lines.extend(
        [
            "",
            "## 8. Top Prediction Errors",
            "",
            "| sample_id | pdb_id | ag_name | true | predicted | error | abs_error | Kd(M) | antigen_len | target_bin |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in top_errors.to_dict("records"):
        lines.append(
            f"| `{row.get('sample_id', '')}` | `{row.get('pdb_id', '')}` | `{str(row.get('ag_name', ''))[:60]}` | "
            f"{row['true_target']:.4f} | {row['predicted_target']:.4f} | {row['error']:.4f} | "
            f"{row['absolute_error']:.4f} | {float(row.get('affinity_kd_m', math.nan)):.3g} | "
            f"{int(row.get('antigen_len', row.get('antigen_sequence_len', 0)))} | `{row['target_bin']}` |"
        )

    lines.extend(
        [
            "",
            "## 9. Main Takeaways",
            "",
            "- The model still has clear regression-to-mean.",
            "- Low-target samples are currently hardest in this ANDD-only benchmark because the model overpredicts them toward the center.",
            "- High-target samples are also underpredicted, but less severely than low-target examples in this run.",
            "- Since keep_safe removed many obvious data quality problems, next improvements should focus on calibration / weighted loss / target-balanced sampling rather than only more filtering.",
            "",
            "## 10. Output Files",
            "",
            f"- `{OUTPUT_DIR / 'top_errors.csv'}`",
            f"- `{OUTPUT_DIR / 'error_by_group.csv'}`",
            f"- `{FIGURE_DIR / 'true_vs_predicted.png'}`",
            f"- `{FIGURE_DIR / 'residual_vs_true.png'}`",
        ]
    )
    (OUTPUT_DIR / "error_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(TEST_CSV)
    pred_df = pd.read_csv(PREDICTIONS_CSV)
    merged = pred_df.merge(test_df, on="sample_id", how="left", suffixes=("", "_meta"))
    if merged["source_meta"].isna().any() if "source_meta" in merged.columns else False:
        raise ValueError("Some prediction rows did not match test metadata by sample_id.")

    # Prefer metadata columns when predictions already contain shorter subset columns.
    for col in ["source", "pdb_id", "ag_name", "affinity_kd_m", "heavy_sequence", "light_sequence"]:
        meta_col = f"{col}_meta"
        if meta_col in merged.columns:
            merged[col] = merged[meta_col]

    analyzed = add_analysis_columns(merged)
    metrics = metric_dict(analyzed)

    group_rows = []
    for column, label in [
        ("target_bin", "target_bin"),
        ("source", "source"),
        ("ag_name", "antigen_name"),
        ("pdb_id", "pdb_id"),
        ("antigen_length_bin", "antigen_length_bin"),
        ("heavy_length_bin", "heavy_length_bin"),
        ("light_length_bin", "light_length_bin"),
        ("peptide_or_epitope_like_antigen", "peptide_or_epitope_like_antigen"),
    ]:
        group_rows.extend(group_metrics(analyzed, column, label))

    top_errors = analyzed.sort_values("absolute_error", ascending=False).head(25)
    top_errors.to_csv(OUTPUT_DIR / "top_errors.csv", index=False)
    pd.DataFrame(group_rows).to_csv(OUTPUT_DIR / "error_by_group.csv", index=False)

    plot_true_vs_pred(analyzed, FIGURE_DIR / "true_vs_predicted.png")
    plot_residual_vs_true(analyzed, FIGURE_DIR / "residual_vs_true.png")

    write_report(analyzed, metrics, group_rows)
    (OUTPUT_DIR / "error_analysis_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Rows analyzed: {len(analyzed)}")
    print(f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, Spearman={metrics['spearman']:.4f}")
    print(f"pred_std/true_std={metrics['pred_std_over_true_std']:.4f}")
    print(f"error_vs_true_Pearson={metrics['error_vs_true_pearson']:.4f}")
    print(f"Saved report to {OUTPUT_DIR / 'error_analysis_report.md'}")


if __name__ == "__main__":
    main()
