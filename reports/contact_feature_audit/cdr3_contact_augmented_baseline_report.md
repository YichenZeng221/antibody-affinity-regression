# ANDD Stratified CDR3 Contact Feature Augmented Baseline

## Question

少量真实 CDR3 interface geometry features 是否能在不改深度模型的条件下，为已有 cross-attention predictions 提供增量价值，尤其缓解 tail compression？

## Experimental Boundary

- 这是 **contact-covered subset analysis**，不是 full 1,168-row benchmark。
- 只纳入经过严格 chain/CDR-to-structure mapping validation 的样本；没有处理 ambiguous chain mapping。
- 不训练新的深度模型：sequence predictions 来自已有 cross-attention models。
- Augmentation 方法：在 train subset 上用 `Ridge(alpha=1.0)` 学习 `target - sequence_prediction`，再对同一 subset 的 test rows 做 residual correction。
- 不用 test labels 选择 feature 或模型。
- Tail thresholds 仍来自完整 stratified train split：P10 = `3.3333`，P90 = `6.3204`。

## Subsets and Features

- `HCDR3+LCDR3 contact-safe`: 467 total validated rows；使用 `hcdr3_contact_count_5A`, `lcdr3_contact_count_5A`, `hcdr3_contact_fraction_5A`, `lcdr3_contact_fraction_5A`。
- `All-CDR contact-safe`: 422 total validated rows；在上述特征基础上增加 `cdr_min_distance` 和 `all_cdr_contact_count_5A`。
- `cdr_min_distance` 在当前验证中定义为 all-six-CDR 到 antigen 的最小距离，因此不用于 CDR3-only subset，以免强行补入未验证映射。

## Test Metrics Within The Same Subset

| subset_description | sequence_baseline | method | train_rows | test_rows | MAE | RMSE | Spearman | pred_std_true_std | error_vs_true_Pearson | below_P10_MAE | above_P90_MAE | tail_MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HCDR3+LCDR3 contact-safe subset | unweighted_cross_attention | sequence_only_prediction | 356 | 58 | 1.0074 | 1.3560 | 0.2443 | 0.3553 | -0.9368 | 2.2889 | 2.0558 | 2.1724 |
| HCDR3+LCDR3 contact-safe subset | unweighted_cross_attention | sequence_plus_contact_ridge_residual | 356 | 58 | 1.0312 | 1.3664 | 0.1888 | 0.3555 | -0.9376 | 2.2067 | 2.1849 | 2.1958 |
| All-CDR contact-safe subset | unweighted_cross_attention | sequence_only_prediction | 322 | 49 | 0.9381 | 1.2506 | 0.2299 | 0.4117 | -0.9175 | 2.4309 | 1.2202 | 1.8256 |
| All-CDR contact-safe subset | unweighted_cross_attention | sequence_plus_contact_ridge_residual | 322 | 49 | 0.9188 | 1.2126 | 0.2199 | 0.4074 | -0.9187 | 2.2629 | 1.4322 | 1.8475 |
| HCDR3+LCDR3 contact-safe subset | tailaware_w2_best_val_tail_mae | sequence_only_prediction | 356 | 58 | 0.9740 | 1.2857 | 0.3796 | 0.5021 | -0.8692 | 2.0234 | 1.7167 | 1.8701 |
| HCDR3+LCDR3 contact-safe subset | tailaware_w2_best_val_tail_mae | sequence_plus_contact_ridge_residual | 356 | 58 | 0.9658 | 1.2745 | 0.3984 | 0.4969 | -0.8711 | 2.0123 | 1.6907 | 1.8515 |
| All-CDR contact-safe subset | tailaware_w2_best_val_tail_mae | sequence_only_prediction | 322 | 49 | 0.9278 | 1.2105 | 0.2883 | 0.5837 | -0.8351 | 2.1898 | 0.9423 | 1.5660 |
| All-CDR contact-safe subset | tailaware_w2_best_val_tail_mae | sequence_plus_contact_ridge_residual | 322 | 49 | 0.9016 | 1.1815 | 0.3423 | 0.5699 | -0.8388 | 2.1141 | 0.9943 | 1.5542 |

## Delta After Contact Residual Correction

| subset_description | sequence_baseline | delta_vs_sequence_MAE | delta_vs_sequence_RMSE | delta_vs_sequence_Spearman | delta_vs_sequence_pred_std_true_std | delta_vs_sequence_error_vs_true_Pearson | delta_vs_sequence_tail_MAE |
|---|---|---|---|---|---|---|---|
| HCDR3+LCDR3 contact-safe subset | unweighted_cross_attention | 0.0238 | 0.0104 | -0.0556 | 0.0002 | -0.0008 | 0.0235 |
| All-CDR contact-safe subset | unweighted_cross_attention | -0.0193 | -0.0380 | -0.0100 | -0.0043 | -0.0012 | 0.0220 |
| HCDR3+LCDR3 contact-safe subset | tailaware_w2_best_val_tail_mae | -0.0083 | -0.0112 | 0.0188 | -0.0052 | -0.0019 | -0.0186 |
| All-CDR contact-safe subset | tailaware_w2_best_val_tail_mae | -0.0262 | -0.0290 | 0.0541 | -0.0138 | -0.0037 | -0.0118 |

读法：MAE/RMSE/tail MAE 的 delta 小于 0 为改善；Spearman 的 delta 大于 0 为改善；`pred_std/true_std` 要看是否更接近 1，`error_vs_true_Pearson` 要看是否更接近 0。

## Standardized Ridge Coefficients

这些系数仅用于观察 correction 倾向，不能视为稳定的生物机制解释。
| subset | sequence_baseline | feature | standardized_ridge_coefficient |
|---|---|---|---|
| hcdr3_lcdr3_contact_safe | unweighted_cross_attention | hcdr3_contact_count_5A | 0.1997 |
| hcdr3_lcdr3_contact_safe | unweighted_cross_attention | lcdr3_contact_count_5A | 0.1134 |
| hcdr3_lcdr3_contact_safe | unweighted_cross_attention | hcdr3_contact_fraction_5A | -0.0113 |
| hcdr3_lcdr3_contact_safe | unweighted_cross_attention | lcdr3_contact_fraction_5A | 0.0266 |
| all_cdr_contact_safe | unweighted_cross_attention | hcdr3_contact_count_5A | 0.1384 |
| all_cdr_contact_safe | unweighted_cross_attention | lcdr3_contact_count_5A | 0.1044 |
| all_cdr_contact_safe | unweighted_cross_attention | hcdr3_contact_fraction_5A | -0.1056 |
| all_cdr_contact_safe | unweighted_cross_attention | lcdr3_contact_fraction_5A | -0.0493 |
| all_cdr_contact_safe | unweighted_cross_attention | cdr_min_distance | 0.0476 |
| all_cdr_contact_safe | unweighted_cross_attention | all_cdr_contact_count_5A | 0.1540 |
| hcdr3_lcdr3_contact_safe | tailaware_w2_best_val_tail_mae | hcdr3_contact_count_5A | 0.0592 |
| hcdr3_lcdr3_contact_safe | tailaware_w2_best_val_tail_mae | lcdr3_contact_count_5A | 0.0307 |
| hcdr3_lcdr3_contact_safe | tailaware_w2_best_val_tail_mae | hcdr3_contact_fraction_5A | 0.0576 |
| hcdr3_lcdr3_contact_safe | tailaware_w2_best_val_tail_mae | lcdr3_contact_fraction_5A | 0.0601 |
| all_cdr_contact_safe | tailaware_w2_best_val_tail_mae | hcdr3_contact_count_5A | 0.0372 |
| all_cdr_contact_safe | tailaware_w2_best_val_tail_mae | lcdr3_contact_count_5A | 0.0419 |
| all_cdr_contact_safe | tailaware_w2_best_val_tail_mae | hcdr3_contact_fraction_5A | -0.0223 |
| all_cdr_contact_safe | tailaware_w2_best_val_tail_mae | lcdr3_contact_fraction_5A | -0.0200 |
| all_cdr_contact_safe | tailaware_w2_best_val_tail_mae | cdr_min_distance | 0.0364 |
| all_cdr_contact_safe | tailaware_w2_best_val_tail_mae | all_cdr_contact_count_5A | 0.1308 |

## Honest Interpretation

- HCDR3+LCDR3 contact-safe subset / `unweighted_cross_attention`: improved prediction spread.
- All-CDR contact-safe subset / `unweighted_cross_attention`: improved MAE.
- HCDR3+LCDR3 contact-safe subset / `tailaware_w2_best_val_tail_mae`: improved MAE, Spearman, tail MAE.
- All-CDR contact-safe subset / `tailaware_w2_best_val_tail_mae`: improved MAE, Spearman, tail MAE.

### Primary Reading For Tail-Aware w2

- `HCDR3+LCDR3 contact-safe subset`: MAE `0.9740 -> 0.9658`, Spearman `0.3796 -> 0.3984`, tail MAE `1.8701 -> 1.8515`, pred_std/true_std `0.5021 -> 0.4969`, error-vs-true Pearson `-0.8692 -> -0.8711`.
- `All-CDR contact-safe subset`: MAE `0.9278 -> 0.9016`, Spearman `0.2883 -> 0.3423`, tail MAE `1.5660 -> 1.5542`, pred_std/true_std `0.5837 -> 0.5699`, error-vs-true Pearson `-0.8351 -> -0.8388`.

- 对 tail-aware w2，CDR3 contact correction 在两个 subset 中都给出小幅 MAE/RMSE/Spearman/tail-MAE 改善，说明真实 interface geometry 可能含有增量信号。
- 但是 prediction spread 没有向 1 靠近，error-vs-true Pearson 也没有向 0 靠近；因此这次线性 correction **没有缓解 regression-to-the-mean 核心现象**。
- 对 unweighted baseline 的结果不一致：一个 subset 恶化，另一个只改善部分 error 指标而没有改善 ranking/tail。这进一步说明当前 contact features 是弱增量证据，而不是稳健的通用修正项。

- 即使某个 correction 在 subset 内改善，也不能直接宣称优于 full 1,168-row model：结构覆盖和 mapping 过滤改变了可评估样本集合。
- 如果两种 baseline 在同一 subset 上都显示 tail/spread 改善，说明真实 CDR3 geometry 可能提供增量信息；如果改善不稳定，则应把 contact counts 视为弱特征，而不是新主线模型依据。
- 这是小型 post-hoc linear correction，不是复杂 structure model，也不是最终性能结论。

## Outputs

- Metrics: `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr3_contact_augmented_metrics.csv`
- Figure: `outputs/final_reports/figures/cdr3_contact_augmented_baseline.png`
- Report: `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr3_contact_augmented_baseline_report.md`
