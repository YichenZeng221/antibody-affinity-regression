"""Analyze affinity regression dataset and prediction behavior.

中文人话说明：
这个脚本不训练模型，也不改数据。
它帮助我们 debug：

1. 数据分布是不是奇怪？
2. train/test 是否有 sequence overlap？
3. 模型预测是不是塌缩到接近平均值？
"""

from pathlib import Path
import math
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_config


def describe_numeric(series: pd.Series) -> str:
    """Return min/max/mean/std for a numeric series."""

    values = pd.to_numeric(series, errors="coerce")
    return (
        f"min={values.min():.4f}, "
        f"max={values.max():.4f}, "
        f"mean={values.mean():.4f}, "
        f"std={values.std():.4f}"
    )


def describe_lengths(dataframe: pd.DataFrame, column_name: str) -> str:
    """Return min/max/mean sequence length stats."""

    lengths = dataframe[column_name].astype(str).str.len()
    return (
        f"min={int(lengths.min())}, "
        f"max={int(lengths.max())}, "
        f"mean={lengths.mean():.2f}"
    )


def duplicate_count(dataframe: pd.DataFrame, column_name: str) -> int:
    """Count duplicated values in one column."""

    return int(dataframe[column_name].astype(str).duplicated().sum())


def overlap_count(first: pd.DataFrame, second: pd.DataFrame, column_name: str) -> int:
    """Count exact-value overlap between two dataframes for one column."""

    return len(set(first[column_name].astype(str)) & set(second[column_name].astype(str)))


def make_pair_key(dataframe: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Create one text key from several columns.

    中文人话说明：
    有些泄漏不是单独 heavy_sequence 重复，而是 heavy+light 这个组合重复。
    把多列拼成一个 key 后，就可以像检查普通字符串一样检查 overlap。
    """

    return dataframe[columns].astype(str).agg("||".join, axis=1)


def overlap_count_for_columns(first: pd.DataFrame, second: pd.DataFrame, columns: list[str]) -> int:
    """Count overlap for combined keys, for example heavy+light pair."""

    first_keys = set(make_pair_key(first, columns))
    second_keys = set(make_pair_key(second, columns))
    return len(first_keys & second_keys)


def compute_prediction_metrics(dataframe: pd.DataFrame) -> dict:
    """Compute MAE/RMSE/fold_error for a prediction dataframe slice.

    中文人话说明：
    这个函数常用于“分组看错误”：
    比如 SPR vs ITC、protein vs peptide。
    如果某一组错误特别大，就说明模型可能对那类数据学不好。
    """

    if len(dataframe) == 0:
        return {
            "count": 0,
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "fold_error_mean": float("nan"),
            "fold_error_median": float("nan"),
        }

    errors = pd.to_numeric(dataframe["error"], errors="coerce")
    absolute_errors = errors.abs()
    mse = (errors ** 2).mean()
    return {
        "count": len(dataframe),
        "MAE": absolute_errors.mean(),
        "RMSE": math.sqrt(mse),
        "fold_error_mean": pd.to_numeric(dataframe["fold_error"], errors="coerce").mean(),
        "fold_error_median": pd.to_numeric(dataframe["fold_error"], errors="coerce").median(),
    }


def print_prediction_metric_table(dataframe: pd.DataFrame, group_column: str) -> None:
    """Print prediction metrics grouped by one metadata column."""

    rows = []
    for value, group in dataframe.groupby(group_column, dropna=False):
        metrics = compute_prediction_metrics(group)
        rows.append(
            {
                group_column: value,
                "count": metrics["count"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "fold_error_mean": metrics["fold_error_mean"],
                "fold_error_median": metrics["fold_error_median"],
            }
        )

    table = pd.DataFrame(rows).sort_values("count", ascending=False)
    if len(table) == 0:
        print("(no rows)")
        return

    print(
        table.assign(
            MAE=table["MAE"].map(lambda value: f"{value:.4f}"),
            RMSE=table["RMSE"].map(lambda value: f"{value:.4f}"),
            fold_error_mean=table["fold_error_mean"].map(lambda value: f"{value:.1f}x"),
            fold_error_median=table["fold_error_median"].map(lambda value: f"{value:.1f}x"),
        ).to_string(index=False)
    )


def print_split_summary(split_name: str, dataframe: pd.DataFrame, target_column: str) -> None:
    """Print summary for one split."""

    print("=" * 80)
    print(f"Split: {split_name}")
    print(f"Rows: {len(dataframe)}")
    print(f"Target {target_column}: {describe_numeric(dataframe[target_column])}")
    print()
    print("antigen_type counts:")
    print(dataframe["antigen_type"].value_counts(dropna=False).head(15).to_string())
    print()
    print("affinity_method counts:")
    print(dataframe["affinity_method"].value_counts(dropna=False).head(15).to_string())
    print()
    temperature_text = dataframe["temperature"].astype(str).str.strip().str.upper()
    temperature_missing = temperature_text.isin(["", "NA", "NAN", "NONE"]).sum()
    print(f"Temperature missing count: {int(temperature_missing)}")
    print(f"heavy_sequence length: {describe_lengths(dataframe, 'heavy_sequence')}")
    print(f"light_sequence length: {describe_lengths(dataframe, 'light_sequence')}")
    print(f"antigen_sequence length: {describe_lengths(dataframe, 'antigen_sequence')}")
    print(f"Duplicated heavy_sequence count: {duplicate_count(dataframe, 'heavy_sequence')}")
    print(f"Duplicated light_sequence count: {duplicate_count(dataframe, 'light_sequence')}")
    print(f"Duplicated antigen_sequence count: {duplicate_count(dataframe, 'antigen_sequence')}")


def print_split_overlap_check(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print exact overlap between train and test.

    中文人话说明：
    PDB-level split 只能保证同一个 pdb 不跨 split。
    但是不同 PDB 里仍可能出现同一条 antibody 或 antigen sequence。
    如果 train/test 有重复 sequence，模型评估就会变得不那么严格。
    """

    print("=" * 80)
    print("1. Split overlap check: train vs test")
    for column_name in ["pdb", "heavy_sequence", "light_sequence", "antigen_sequence"]:
        print(f"{column_name} overlap: {overlap_count(train, test, column_name)}")

    heavy_light_overlap = overlap_count_for_columns(
        train,
        test,
        ["heavy_sequence", "light_sequence"],
    )
    triplet_overlap = overlap_count_for_columns(
        train,
        test,
        ["heavy_sequence", "light_sequence", "antigen_sequence"],
    )
    print(f"heavy+light pair overlap: {heavy_light_overlap}")
    print(f"heavy+light+antigen triplet overlap: {triplet_overlap}")
    print()


def print_duplicate_sample_check(all_data: pd.DataFrame, target_column: str) -> None:
    """Print repeated PDB and repeated sequence-combination checks.

    重复 heavy+light+antigen triplet 表示模型看到的三条输入序列完全一样。
    如果重复很多，表面样本数会比“真正独立的信息量”更大。
    如果重复 triplet 的 target 还不一致，那就是更严重的 label conflict。
    """

    print("=" * 80)
    print("2. Duplicate / repeated sample check")
    print("Top 10 most repeated pdb values:")
    print(all_data["pdb"].value_counts().head(10).to_string())
    print()

    triplet_columns = ["heavy_sequence", "light_sequence", "antigen_sequence"]
    all_data = all_data.copy()
    all_data["triplet_key"] = make_pair_key(all_data, triplet_columns)

    duplicate_rows = all_data[all_data["triplet_key"].duplicated(keep=False)]
    duplicate_triplet_count = duplicate_rows["triplet_key"].nunique()
    print(f"Repeated heavy+light+antigen triplet groups: {duplicate_triplet_count}")
    print(f"Rows involved in repeated triplets: {len(duplicate_rows)}")

    if len(duplicate_rows) == 0:
        print("No repeated complete triplets found.")
        print()
        return

    rows = []
    for triplet_key, group in duplicate_rows.groupby("triplet_key"):
        targets = group[target_column].astype(float)
        rows.append(
            {
                "count": len(group),
                "target_min": targets.min(),
                "target_max": targets.max(),
                "target_range": targets.max() - targets.min(),
                "pdb_values": ", ".join(group["pdb"].astype(str).unique()[:5]),
                "first_sample_id": group["sample_id"].astype(str).iloc[0],
            }
        )

    table = pd.DataFrame(rows).sort_values("target_range", ascending=False)
    consistent_count = int((table["target_range"] < 1e-8).sum())
    inconsistent_count = len(table) - consistent_count
    print(f"Repeated triplets with same target: {consistent_count}")
    print(f"Repeated triplets with different target: {inconsistent_count}")
    print()
    print("Top 10 repeated triplets with largest target difference:")
    print(
        table.head(10)
        .assign(
            target_min=table["target_min"].map(lambda value: f"{value:.4f}"),
            target_max=table["target_max"].map(lambda value: f"{value:.4f}"),
            target_range=table["target_range"].map(lambda value: f"{value:.4f}"),
        )
        .to_string(index=False)
    )
    print()


def print_affinity_method_check(all_data: pd.DataFrame, predictions: pd.DataFrame | None, target_column: str) -> None:
    """Print target and prediction stats by affinity_method.

    不同实验方法，比如 SPR / ITC，测出来的 affinity 可能有系统差异。
    如果混在一起训练，模型可能会学到很多 assay noise，而不是 sequence signal。
    """

    print("=" * 80)
    print("3. Affinity method check")
    print("affinity_method counts:")
    print(all_data["affinity_method"].value_counts(dropna=False).to_string())
    print()
    print("Target by affinity_method:")
    target_table = (
        all_data.groupby("affinity_method", dropna=False)[target_column]
        .agg(["count", "mean", "std", "min", "max"])
        .sort_values("count", ascending=False)
    )
    print(
        target_table.assign(
            mean=target_table["mean"].map(lambda value: f"{value:.4f}"),
            std=target_table["std"].map(lambda value: "NaN" if pd.isna(value) else f"{value:.4f}"),
            min=target_table["min"].map(lambda value: f"{value:.4f}"),
            max=target_table["max"].map(lambda value: f"{value:.4f}"),
        ).to_string()
    )

    if predictions is not None and "affinity_method" in predictions.columns:
        print()
        print("Prediction metrics by affinity_method:")
        print_prediction_metric_table(predictions, "affinity_method")
    print()


def print_antigen_type_check(all_data: pd.DataFrame, predictions: pd.DataFrame | None, target_column: str) -> None:
    """Print target and prediction stats by antigen_type.

    protein、peptide、protein | protein 的难度可能不同。
    分开看可以发现：是不是某一类 antigen 特别拖累整体指标。
    """

    print("=" * 80)
    print("4. Antigen type check")
    print("antigen_type counts:")
    print(all_data["antigen_type"].value_counts(dropna=False).to_string())
    print()
    print("Target by antigen_type:")
    target_table = (
        all_data.groupby("antigen_type", dropna=False)[target_column]
        .agg(["count", "mean", "std"])
        .sort_values("count", ascending=False)
    )
    print(
        target_table.assign(
            mean=target_table["mean"].map(lambda value: f"{value:.4f}"),
            std=target_table["std"].map(lambda value: "NaN" if pd.isna(value) else f"{value:.4f}"),
        ).to_string()
    )

    if predictions is not None and "antigen_type" in predictions.columns:
        print()
        print("Prediction metrics by antigen_type:")
        print_prediction_metric_table(predictions, "antigen_type")
    print()


def print_extreme_target_check(
    split_data: dict[str, pd.DataFrame],
    predictions: pd.DataFrame | None,
    target_column: str,
) -> None:
    """Print very low / very high target checks.

    如果低 affinity / 高 affinity 的极端样本很少，
    模型容易学成“猜平均值”，因为猜平均值在大多数中间样本上不会太离谱。
    """

    print("=" * 80)
    print("5. Extreme target check")
    for split_name, dataframe in split_data.items():
        targets = dataframe[target_column].astype(float)
        print(
            f"{split_name}: target < 6: {int((targets < 6).sum())}, "
            f"target > 10: {int((targets > 10).sum())}"
        )

    if predictions is not None:
        print()
        print("Extreme test samples with predictions:")
        extreme_predictions = predictions[
            (predictions["true_neg_log10_affinity"].astype(float) < 6)
            | (predictions["true_neg_log10_affinity"].astype(float) > 10)
        ].copy()
        if len(extreme_predictions) == 0:
            print("(no extreme samples in prediction file)")
        else:
            columns = [
                "sample_id",
                "pdb",
                "true_neg_log10_affinity",
                "predicted_neg_log10_affinity",
                "error",
                "fold_error",
            ]
            print(
                extreme_predictions[columns]
                .assign(
                    true_neg_log10_affinity=extreme_predictions["true_neg_log10_affinity"].map(lambda value: f"{value:.4f}"),
                    predicted_neg_log10_affinity=extreme_predictions["predicted_neg_log10_affinity"].map(lambda value: f"{value:.4f}"),
                    error=extreme_predictions["error"].map(lambda value: f"{value:.4f}"),
                    fold_error=extreme_predictions["fold_error"].map(lambda value: f"{value:.1f}x"),
                )
                .to_string(index=False)
            )
    print()


def print_truncation_check(
    split_data: dict[str, pd.DataFrame],
    predictions: pd.DataFrame | None,
    max_length: int,
) -> None:
    """Print how often sequences are longer than tokenizer max_length.

    truncation 表示序列超过 max_length 的部分会被切掉。
    如果 antigen 很长，切掉的信息可能刚好是 binding 相关区域。
    """

    print("=" * 80)
    print("6. Truncation check")
    print(f"Configured max_length: {max_length}")
    for split_name, dataframe in split_data.items():
        print(f"{split_name}:")
        for column_name in ["heavy_sequence", "light_sequence", "antigen_sequence"]:
            lengths = dataframe[column_name].astype(str).str.len()
            print(f"  {column_name} length > max_length: {int((lengths > max_length).sum())}")

    if predictions is not None:
        print()
        print("Prediction metrics by antigen truncation group:")
        predictions = predictions.copy()
        predictions["antigen_length"] = predictions["antigen_sequence"].astype(str).str.len()
        predictions["antigen_length_group"] = predictions["antigen_length"].map(
            lambda length: f"<= {max_length}" if length <= max_length else f"> {max_length}"
        )
        print_prediction_metric_table(predictions, "antigen_length_group")
    print()


def analyze_predictions(predictions_path: Path, train_target_mean: float) -> None:
    """Analyze saved model predictions if the CSV exists."""

    print("=" * 80)
    print("Prediction distribution")

    if not predictions_path.exists():
        print(f"No predictions file found at {predictions_path}")
        return

    predictions = pd.read_csv(predictions_path)

    print(f"Prediction rows: {len(predictions)}")
    print(f"true_neg_log10_affinity: {describe_numeric(predictions['true_neg_log10_affinity'])}")
    print(f"predicted_neg_log10_affinity: {describe_numeric(predictions['predicted_neg_log10_affinity'])}")
    print(f"error: {describe_numeric(predictions['error'])}")

    fold_error = pd.to_numeric(predictions["fold_error"], errors="coerce")
    print(
        "fold_error: "
        f"min={fold_error.min():.2f}x, "
        f"max={fold_error.max():.2f}x, "
        f"median={fold_error.median():.2f}x, "
        f"mean={fold_error.mean():.2f}x"
    )

    predicted_std = predictions["predicted_neg_log10_affinity"].std()
    true_std = predictions["true_neg_log10_affinity"].std()
    predicted_mean = predictions["predicted_neg_log10_affinity"].mean()
    distance_from_train_mean = abs(predicted_mean - train_target_mean)

    print()
    print("Collapse check:")
    print(f"Train target mean: {train_target_mean:.4f}")
    print(f"Prediction mean: {predicted_mean:.4f}")
    print(f"True target std: {true_std:.4f}")
    print(f"Prediction std: {predicted_std:.4f}")
    print(f"Distance between prediction mean and train mean: {distance_from_train_mean:.4f}")

    if predicted_std < true_std * 0.25 and distance_from_train_mean < 0.5:
        print("Likely collapsed near the train mean: YES")
    else:
        print("Likely collapsed near the train mean: not clearly")


def load_predictions_with_metadata(predictions_path: Path, test: pd.DataFrame) -> pd.DataFrame | None:
    """Load predictions and attach test-set metadata.

    中文人话说明：
    predictions CSV 主要存预测值。
    affinity_method / antigen_type 这些分组信息在 test.csv 里。
    所以这里用 sample_id 把它们合并起来，方便做分组误差分析。
    """

    if not predictions_path.exists():
        return None

    predictions = pd.read_csv(predictions_path)
    metadata_columns = [
        "sample_id",
        "affinity_method",
        "antigen_type",
        "antigen_name",
        "affinity",
        "delta_g",
        "temperature",
        "pmid",
    ]
    available_columns = [column for column in metadata_columns if column in test.columns]

    if "sample_id" in predictions.columns and "sample_id" in test.columns:
        predictions = predictions.merge(
            test[available_columns],
            on="sample_id",
            how="left",
            suffixes=("", "_from_test"),
        )

    return predictions


def main() -> None:
    """Run dataset and prediction analysis."""

    config = load_config("config_affinity.yaml")
    target_column = config["target_column"]
    max_length = int(config.get("max_length", 512))

    train = pd.read_csv(config["train_csv"])
    val = pd.read_csv(config["val_csv"])
    test = pd.read_csv(config["test_csv"])
    split_data = {"train": train, "val": val, "test": test}
    all_data = pd.concat(
        [
            train.assign(split="train"),
            val.assign(split="val"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )

    for split_name, dataframe in split_data.items():
        print_split_summary(split_name, dataframe, target_column)
        print()

    predictions_path = Path(config["predictions_path"])
    predictions = load_predictions_with_metadata(predictions_path, test)

    print_split_overlap_check(train, test)
    print_duplicate_sample_check(all_data, target_column)
    print_affinity_method_check(all_data, predictions, target_column)
    print_antigen_type_check(all_data, predictions, target_column)
    print_extreme_target_check(split_data, predictions, target_column)
    print_truncation_check(split_data, predictions, max_length)

    train_target_mean = float(train[target_column].astype(float).mean())
    analyze_predictions(predictions_path, train_target_mean)


if __name__ == "__main__":
    main()
