"""Evaluation metrics for affinity regression.

中文人话说明：
训练时我们用 loss 来更新模型；
评估时我们更关心“预测到底差多少”。

这里集中放 regression metrics：
- MAE：平均绝对误差，越小越好，比较直观。
- MSE：平均平方误差，对大错误惩罚更重。
- RMSE：MSE 开根号，单位又回到 target 的单位。
- Spearman：看排序能力，预测能不能把强/弱 binding 大概排对。
"""

import math

import pandas as pd
import torch


def compute_regression_metrics(true_values: list[float], predicted_values: list[float]) -> dict:
    """Compute MAE, MSE, RMSE, Spearman, and approximate fold error.

    中文人话说明：
    我们的 target 是 -log10(affinity)。
    所以 1.0 的 log10 误差，大约对应 affinity 差 10 倍。
    """

    if len(true_values) == 0:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "spearman": 0.0,
            "approx_mae_fold_error": 1.0,
            "approx_rmse_fold_error": 1.0,
        }

    # error = prediction - true。
    # 正数表示预测偏高，负数表示预测偏低。
    errors = [pred - true for true, pred in zip(true_values, predicted_values)]
    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]

    # MAE 是平均 abs(error)，适合直观看“平均差多少 log10 unit”。
    mae = sum(absolute_errors) / len(absolute_errors)

    # MSE/RMSE 会更重视少数特别大的错误。
    mse = sum(squared_errors) / len(squared_errors)
    rmse = math.sqrt(mse)

    # Spearman 不看数值差多少，只看排序是否接近。
    # 如果模型预测几乎是常数，Spearman 会很低或 NaN。
    spearman = pd.Series(true_values).corr(pd.Series(predicted_values), method="spearman")
    if pd.isna(spearman):
        spearman = 0.0

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "spearman": float(spearman),
        "approx_mae_fold_error": 10 ** mae,
        "approx_rmse_fold_error": 10 ** rmse,
    }


def evaluate_affinity_model(model, dataloader, device) -> tuple[dict, list[float], list[float]]:
    """Run inference on a dataloader and return metrics plus raw values.

    中文人话说明：
    evaluate 不应该更新模型参数，所以会：
    - model.eval()：关闭 dropout 等训练行为。
    - torch.no_grad()：不记录梯度，省内存，也更快。
    """

    # eval mode 和 train mode 不一样：eval mode 下 dropout 会关闭。
    model.eval()
    true_values = []
    predicted_values = []

    # no_grad 表示这里只做前向计算，不做反向传播。
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)

            outputs = model(
                heavy_input_ids=batch["heavy_input_ids"].to(device),
                heavy_attention_mask=batch["heavy_attention_mask"].to(device),
                light_input_ids=batch["light_input_ids"].to(device),
                light_attention_mask=batch["light_attention_mask"].to(device),
                antigen_input_ids=batch["antigen_input_ids"].to(device),
                antigen_attention_mask=batch["antigen_attention_mask"].to(device),
            )

            # 保存到 CPU list，方便后面用 pandas/math 计算指标或写 CSV。
            true_values.extend(labels.cpu().tolist())
            predicted_values.extend(outputs["predictions"].cpu().tolist())

    metrics = compute_regression_metrics(true_values, predicted_values)
    return metrics, true_values, predicted_values
