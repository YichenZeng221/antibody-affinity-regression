"""Evaluate the trained affinity regression model on the test set.
中文人话说明：
test set 是最终考试集，不应该参与训练或调参。
这个脚本只加载已经训练好的 checkpoint，在 test.csv 上做预测，
然后计算整体指标，并把每条样本的预测保存成 CSV。
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.affinity_dataset import AffinityRegressionDataset
from src.affinity_evaluate import compute_regression_metrics, evaluate_affinity_model
from src.affinity_model import SeqProFTAffinityRegressor
from src.utils import get_device, load_config


OPTIONAL_METADATA_COLUMNS = ["pdb", "antibody_id", "antigen_id", "source"]


def parse_args() -> argparse.Namespace:
    """Read command line arguments.

    默认使用 config_affinity.yaml。
    传 --config 可以评估 clean_v2 all_methods / spr_only 的 checkpoint。
    """

    parser = argparse.ArgumentParser(description="Evaluate affinity regression test set.")
    parser.add_argument("--config", default="config_affinity.yaml")
    return parser.parse_args()


def build_prediction_row(row: dict, true_value: float, predicted_value: float) -> dict:
    """Build one predictions.csv row without assuming every dataset has pdb.

    中文人话说明：
    sequence_only / clean_v2 数据有 pdb 列。
    TDC v1 数据没有 pdb，但有 antibody_id / antigen_id / source。
    所以这里用 row.get(...) 安全读取可选 metadata，避免 KeyError。
    """

    error = predicted_value - true_value
    absolute_error = abs(error)

    # 因为 target 是 -log10(affinity)，log10 scale 上的误差可以转换成
    # approximate fold error。比如 absolute_error=1 约等于差 10 倍。
    fold_error = 10 ** absolute_error

    prediction_row = {
        "sample_id": row.get("sample_id", ""),
        "true_neg_log10_affinity": true_value,
        "predicted_neg_log10_affinity": predicted_value,
        "error": error,
        "absolute_error": absolute_error,
        "fold_error": fold_error,
        "heavy_sequence": row.get("heavy_sequence", ""),
        "light_sequence": row.get("light_sequence", ""),
        "antigen_sequence": row.get("antigen_sequence", ""),
    }

    for column_name in OPTIONAL_METADATA_COLUMNS:
        if column_name in row:
            prediction_row[column_name] = row.get(column_name, "")

    return prediction_row


def top_error_columns(predictions: pd.DataFrame) -> list[str]:
    """Choose columns for top-error printing based on what exists."""

    columns = ["sample_id"]
    for column_name in ["pdb", "antibody_id", "antigen_id", "source"]:
        if column_name in predictions.columns:
            columns.append(column_name)

    columns.extend(
        [
            "true_neg_log10_affinity",
            "predicted_neg_log10_affinity",
            "error",
            "absolute_error",
            "fold_error",
        ]
    )
    return columns


def main() -> None:
    """Load checkpoint, evaluate test set, and save per-sample predictions."""

    args = parse_args()

    # 读取训练时同一份配置，确保模型结构、max_length、数据路径一致。
    config = load_config(args.config)
    device = get_device()
    print(f"Using device: {device}")

    # tokenizer 必须和训练时的 model_name 一致。
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    test_dataset = AffinityRegressionDataset(
        csv_path=config["test_csv"],
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        target_column=config["target_column"],
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    # 先创建同样结构的模型，再把 checkpoint 里的参数加载进去。
    model = SeqProFTAffinityRegressor(config).to(device)
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # evaluate_affinity_model 内部会使用 model.eval() 和 torch.no_grad()，
    # 所以不会更新模型参数。
    metrics, true_values, predicted_values = evaluate_affinity_model(
        model, test_dataloader, device
    )

    print()
    print("Affinity test set evaluation")
    print(f"Test MAE: {metrics['mae']:.4f} log10 units")
    print(f"Approx average fold error from MAE: {metrics['approx_mae_fold_error']:.1f}x")
    print(f"Test MSE: {metrics['mse']:.4f}")
    print(f"Test RMSE: {metrics['rmse']:.4f} log10 units")
    print(f"Approx fold error from RMSE: {metrics['approx_rmse_fold_error']:.1f}x")
    print(f"Test Spearman: {metrics['spearman']:.4f}")

    # raw_test 保留了 sample_id/pdb/sequence 等原始信息，
    # 这样 predictions CSV 不只是数字，也能回到具体样本看错误。
    raw_test = pd.read_csv(config["test_csv"])
    prediction_rows = []

    for row, true_value, predicted_value in zip(
        raw_test.to_dict("records"),
        true_values,
        predicted_values,
    ):
        prediction_rows.append(build_prediction_row(row, true_value, predicted_value))

    predictions = pd.DataFrame(prediction_rows)
    predictions_path = Path(config["predictions_path"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)

    print()
    print("Top 10 largest prediction errors")
    if predictions.empty:
        print("No predictions to show.")
    else:
        columns = top_error_columns(predictions)
        print(
            predictions.sort_values("absolute_error", ascending=False)
            .head(10)[columns]
            .to_string(index=False)
        )

    print()
    print(f"Saved affinity test predictions to {predictions_path}")


if __name__ == "__main__":
    main()
