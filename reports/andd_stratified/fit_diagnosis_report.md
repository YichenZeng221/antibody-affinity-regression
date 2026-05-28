# ANDD Stratified Train vs Eval/Test Model Fit Diagnosis

## 目的与范围

本分析仅使用已有 checkpoint 进行 inference，不进行新训练，也不修改 dataset 或原有结果文件。

- Dataset: `data/processed_affinity/expanded_affinity_antibody_v2_stratified/`
- Models: `all_cdr_pooled` 与 `all_cdr_cross_attention`
- Inference device in this run: `pooled=cpu, cross_attention=cpu`
- Existing test predictions were read from original output paths; missing train/val predictions were saved only under `fit_diagnosis/predictions/`.
- Low/mid/high bins 与 P10/P90 tails 全部使用 **train target distribution** 定义。

## Metrics By Model And Split

| model | split | rows | MAE | RMSE | Spearman | pred_std_true_std | error_vs_true_Pearson | low_MAE | mid_MAE | high_MAE | below_P10_MAE | above_P90_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_cdr_pooled | train | 934 | 0.7497 | 0.9481 | 0.5831 | 0.4351 | -0.9041 | 0.9195 | 0.3745 | 0.9565 | 1.5772 | 1.3832 |
| all_cdr_pooled | val | 117 | 0.9588 | 1.3931 | 0.4239 | 0.3114 | -0.9504 | 1.0598 | 0.3472 | 1.5491 | 1.7659 | 2.5008 |
| all_cdr_pooled | test | 117 | 0.9373 | 1.3056 | 0.3699 | 0.3170 | -0.9484 | 1.1674 | 0.3773 | 1.3367 | 1.8650 | 2.0887 |
| all_cdr_cross_attention | train | 934 | 0.7597 | 0.9840 | 0.5403 | 0.4584 | -0.8898 | 1.1369 | 0.3494 | 0.7940 | 1.8278 | 1.2357 |
| all_cdr_cross_attention | val | 117 | 0.9825 | 1.3928 | 0.4188 | 0.3831 | -0.9239 | 1.1805 | 0.4800 | 1.3495 | 1.8471 | 2.4690 |
| all_cdr_cross_attention | test | 117 | 0.9523 | 1.3008 | 0.3861 | 0.3925 | -0.9202 | 1.4182 | 0.3876 | 1.1147 | 2.0273 | 1.8483 |

## 1. Train 上是否也有 Regression-To-The-Mean？

- Pooled train: `pred_std / true_std=0.4351`, `error_vs_true_Pearson=-0.9041`。
- Cross-attention train: `pred_std / true_std=0.4584`, `error_vs_true_Pearson=-0.8898`。
- Pooled test: `pred_std / true_std=0.3170`, `error_vs_true_Pearson=-0.9484`。
- Cross-attention test: `pred_std / true_std=0.3925`, `error_vs_true_Pearson=-0.9202`。

两个模型在 train split 上已经出现明显 prediction compression 和负向 residual trend，因此当前证据更像 underfit / representation 或 objective bottleneck，而不是典型的“train 拟合很好、只在 validation/test 崩掉”的 overfit。

## 2. Underfit、Overfit 还是 Representation Bottleneck？

- 如果 train 的 MAE 很低且 prediction spread 健康，而 val/test 才严重压缩，才更符合 overfit。
- 如果 train 自己也有小 `pred_std / true_std` 和强负 `error_vs_true_Pearson`，说明模型从训练数据开始就没有充分表示 target extremes。
- 本次结果应优先解读为：`regression-to-the-mean / representation-or-objective bottleneck`；这比“单纯 overfit”更符合观察。
- Cross-attention 与 pooled 的比较仍然说明 learnable interaction 有帮助，但未完全解决 calibration/tail error。

## 3. Simple Sequence Feature Relationships

### Target vs sequence features：绝对 Spearman 较大的 available relationships

- `val` / `antigen_len`: Pearson=-0.3764, Spearman=-0.5076
- `test` / `antigen_len`: Pearson=-0.3724, Spearman=-0.4407
- `train` / `antigen_len`: Pearson=-0.3766, Spearman=-0.4125
- `train` / `LCDR3_len`: Pearson=0.2240, Spearman=0.2489
- `test` / `LCDR3_len`: Pearson=0.0251, Spearman=0.1419

### Pooled absolute error vs sequence features

- `val` / `total_CDR_len`: Pearson=0.2704, Spearman=0.1351
- `val` / `HCDR3_len`: Pearson=0.0996, Spearman=0.0983
- `test` / `total_CDR_len`: Pearson=0.0525, Spearman=0.0727
- `val` / `antigen_len`: Pearson=-0.0266, Spearman=0.0665
- `test` / `LCDR3_len`: Pearson=-0.0808, Spearman=-0.0533

### Cross-attention absolute error vs sequence features

- `val` / `total_CDR_len`: Pearson=0.3314, Spearman=0.2383
- `val` / `HCDR3_len`: Pearson=0.1471, Spearman=0.1522
- `test` / `total_CDR_len`: Pearson=0.0711, Spearman=0.1396
- `train` / `antigen_len`: Pearson=0.1209, Spearman=0.1210
- `test` / `antigen_len`: Pearson=0.0312, Spearman=0.0692

这些是简单相关分析，不等于因果关系；尤其 antigen groups 和 target 来源可能同时影响这些数字。

## 4. Contact / Structure Feature Availability

本次 stratified dataset 中未发现以下 structure/contact feature columns，因此未进行相应 correlation analysis：

- `contact_count`, `min_distance`, `interface_residue_count`

这不是把它们视为无关，而是说明当前 dataset 尚未提供这些信息。

## 5. 下一步建议

1. **Multi-seed / checkpoint policy**：当前仍是 single-seed baseline，应先验证 compression 与 cross-attention 改善是否稳定。
2. **Tail-aware training 或 calibration**：因为主要错误仍集中在 affinity extremes，可比较 tail-aware weighting、ranking/calibration objective 与按 validation tail 指标选 checkpoint。
3. **Structure/contact-aware features**：如果 train 上也明显压缩，继续只增加 pooled sequence 变体的收益可能有限；CDR-antigen interface/contact information 是更有生物意义的下一步。

## 输出文件

- Metrics: `outputs/andd_antibody_v2_stratified/fit_diagnosis/fit_metrics_by_split.csv`
- Feature correlations: `outputs/andd_antibody_v2_stratified/fit_diagnosis/feature_correlation_summary.csv`
- True-vs-predicted figure: `outputs/final_reports/figures/train_eval_true_predicted_scatter.png`
- Residual figure: `outputs/final_reports/figures/train_eval_residual_scatter.png`
