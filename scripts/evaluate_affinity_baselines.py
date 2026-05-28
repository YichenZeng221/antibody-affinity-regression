"""Evaluate simple affinity regression baselines.

中文人话说明：
这个脚本不训练模型。
它只回答一个 debugging 问题：

    ESM2+LoRA 是否真的超过了“永远预测训练集平均值/中位数”的简单 baseline？

如果模型没有明显超过 mean/median baseline，说明它可能只是学会预测接近平均值。
"""

import argparse
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_config


def parse_args() -> argparse.Namespace:
    """Read command line arguments.

    默认使用 config_affinity.yaml。
    传 --config 可以比较 clean_v2 数据的 mean/median/random baseline。
    """

    parser = argparse.ArgumentParser(description="Evaluate affinity regression baselines.")
    parser.add_argument("--config", default="config_affinity.yaml")
    return parser.parse_args()


def compute_metrics(true_values: list[float], predicted_values: list[float]) -> dict:
    """Compute regression metrics for one set of predictions.

    中文人话说明：
    - MAE / RMSE 越小越好。
    - Spearman 看的是排序能力：预测值能不能把强/弱 binding 大概排对。
    - 如果所有 prediction 都一样，Spearman 没有意义，所以直接设为 NaN。
    """

    # MAE / MSE / RMSE 都是用 error 算出来的。
    # error = predicted - true；绝对值越小，预测越准。
    errors = [pred - true for true, pred in zip(true_values, predicted_values)]
    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]

    mae = sum(absolute_errors) / len(absolute_errors)
    mse = sum(squared_errors) / len(squared_errors)
    rmse = math.sqrt(mse)

    prediction_mean = float(pd.Series(predicted_values).mean())
    prediction_std = float(pd.Series(predicted_values).std())

    # Spearman measures ranking ability.
    # Constant prediction means every sample gets the same score/rank.
    # In that case Spearman is undefined, so we show NaN on purpose.
    if pd.isna(prediction_std) or prediction_std < 1e-12:
        spearman = float("nan")
    else:
        spearman = pd.Series(true_values).corr(pd.Series(predicted_values), method="spearman")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Spearman": spearman,
        # 因为 target 是 -log10(affinity)，所以 log10 误差可以转成倍数误差。
        # 例如 MAE=1 约等于平均差 10 倍，MAE=2 约等于差 100 倍。
        "MAE_fold_error": 10 ** mae,
        "RMSE_fold_error": 10 ** rmse,
        "prediction_mean": prediction_mean,
        "prediction_std": prediction_std,
    }


def format_float(value: float) -> str:
    """Format floats for a clean table."""

    if pd.isna(value):
        return "NaN"
    return f"{value:.4f}"


def make_result_row(model_name: str, true_values: list[float], predicted_values: list[float]) -> dict:
    """Create one table row for one model/baseline."""

    metrics = compute_metrics(true_values, predicted_values)
    return {"model": model_name, **metrics}


def add_current_model_if_available(rows: list[dict], predictions_path: Path) -> None:
    """Add ESM2+LoRA metrics if the saved predictions CSV exists.

    中文人话说明：
    baseline 和当前模型放在同一张表里，才能看出模型有没有真的学到东西。
    如果 ESM2+LoRA 还打不过 mean baseline，说明现在更该 debug 数据/任务。
    """

    if not predictions_path.exists():
        print(f"Prediction file not found, skipping ESM2+LoRA row: {predictions_path}")
        return

    predictions = pd.read_csv(predictions_path)
    required_columns = {
        "true_neg_log10_affinity",
        "predicted_neg_log10_affinity",
    }
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        print(f"Prediction file is missing columns, skipping ESM2+LoRA row: {missing_columns}")
        return

    true_values = predictions["true_neg_log10_affinity"].astype(float).tolist()
    predicted_values = predictions["predicted_neg_log10_affinity"].astype(float).tolist()
    rows.append(make_result_row("ESM2_LoRA_current", true_values, predicted_values))


def main() -> None:
    """Run baseline comparison without training any model."""

    args = parse_args()
    config = load_config(args.config)
    target_column = config.get("target_column", "neg_log10_affinity")
    predictions_path = PROJECT_ROOT / config["predictions_path"]

    train = pd.read_csv(config["train_csv"])
    test = pd.read_csv(config["test_csv"])

    train_targets = train[target_column].astype(float)
    test_targets = test[target_column].astype(float).tolist()

    # mean baseline 很重要：
    # 它代表“完全不看 sequence，只永远猜训练集平均 target”。
    # 真模型至少应该明显超过它，才说明 sequence 信息有帮助。
    mean_value = float(train_targets.mean())
    median_value = float(train_targets.median())

    rows = []
    rows.append(make_result_row("mean_baseline", test_targets, [mean_value] * len(test_targets)))
    rows.append(make_result_row("median_baseline", test_targets, [median_value] * len(test_targets)))

    # Random baseline:
    # 从训练集 target 里“有放回抽样”出和 test set 一样多的预测值。
    # 它不看 sequence，只回答：随便从训练答案分布里猜，会有多差？
    rng = np.random.default_rng(seed=42)
    random_predictions = rng.choice(train_targets.to_numpy(), size=len(test_targets), replace=True)
    rows.append(make_result_row("random_train_distribution", test_targets, random_predictions.tolist()))

    add_current_model_if_available(rows, predictions_path)

    table = pd.DataFrame(rows)

    print("Affinity regression baseline comparison")
    print(f"Train target mean: {mean_value:.4f}")
    print(f"Train target median: {median_value:.4f}")
    print(f"Test size: {len(test_targets)}")
    print()
    print(
        table.assign(
            MAE=table["MAE"].map(lambda value: f"{value:.4f}"),
            MSE=table["MSE"].map(lambda value: f"{value:.4f}"),
            RMSE=table["RMSE"].map(lambda value: f"{value:.4f}"),
            MAE_fold_error=table["MAE_fold_error"].map(lambda value: f"{value:.1f}x"),
            RMSE_fold_error=table["RMSE_fold_error"].map(lambda value: f"{value:.1f}x"),
            Spearman=table["Spearman"].map(format_float),
            prediction_mean=table["prediction_mean"].map(lambda value: f"{value:.4f}"),
            prediction_std=table["prediction_std"].map(lambda value: f"{value:.4f}"),
        )[
            [
                "model",
                "MAE",
                "MSE",
                "RMSE",
                "MAE_fold_error",
                "RMSE_fold_error",
                "Spearman",
                "prediction_mean",
                "prediction_std",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        "Note: Spearman is NaN for constant predictors because they have no ranking ability."
    )

    if "ESM2_LoRA_current" in set(table["model"]):
        mean_row = table.loc[table["model"] == "mean_baseline"].iloc[0]
        model_row = table.loc[table["model"] == "ESM2_LoRA_current"].iloc[0]

        print()
        if model_row["MAE"] >= mean_row["MAE"] or model_row["RMSE"] >= mean_row["RMSE"]:
            print("Current model does not clearly beat mean baseline.")
        else:
            print("Current model beats mean baseline on MAE/RMSE.")

        if model_row["prediction_std"] < 0.1:
            print("Current model predictions are collapsed near one value.")


if __name__ == "__main__":
    main()
