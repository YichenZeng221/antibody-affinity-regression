# ESM2 150M Cross-Attention Interim Summary

## Purpose

This note summarizes the current ESM2 150M all-CDR cross-attention results on the ANDD antibody v2 stratified split.

The goal is to check whether the 150M backbone improvement is reproducible beyond a single seed before making a final multi-seed claim.

## Setup

- Dataset: ANDD antibody v2 stratified antigen-level split
- Input: HCDR1, HCDR2, HCDR3, LCDR1, LCDR2, LCDR3 as query tokens; antigen sequence as key/value tokens
- Model: ESM2 150M + LoRA + all-CDR to antigen cross-attention
- Loss: unweighted MSE
- Checkpoint policy: validation-selected checkpoint, primarily `best_val_tail_mae`
- Evaluation: test split, 117 rows

## Results So Far

| model | seed | MAE | RMSE | Spearman | pred_std/true_std | error_vs_true_Pearson | below_P10_MAE | above_P90_MAE | tail_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8M cross-attention baseline | 42 | 0.9523 | 1.3008 | 0.3861 | 0.3925 | -0.9202 | 2.0273 | 1.8483 | 1.9378 |
| 150M cross-attention | 42 | 0.9047 | 1.2038 | 0.5221 | 0.7341 | -0.7021 | 1.6376 | 1.3923 | 1.5150 |
| 150M cross-attention | 123 | 0.8815 | 1.2241 | 0.5532 | 0.7606 | -0.6833 | 1.6216 | 1.5870 | 1.6043 |

## Interim Interpretation

Across the two available 150M seeds, the larger backbone improves the main symptoms observed with the 8M model:

- MAE improves relative to the 8M baseline in both seed 42 and seed 123.
- Spearman improves substantially in both seeds.
- `pred_std/true_std` is much closer to 1, indicating less prediction compression.
- `error_vs_true_Pearson` is less negative, indicating weaker regression-to-the-mean.
- P10/P90 tail MAE improves relative to the 8M baseline.

This supports the hypothesis that the 8M model had a real capacity / representation bottleneck.

## Current Caution

This is still an interim result.

Only two 150M seeds are currently summarized here. The result is promising, but a stronger claim should wait until seed 2026 is finished and included in the same table.

Also, even with 150M, prediction compression is not fully solved:

- `pred_std/true_std` remains below 1.
- `error_vs_true_Pearson` is still negative.
- Tail MAE remains much higher than mid-range target error.

So the conclusion is not that 150M solves absolute affinity regression. The better conclusion is:

> Scaling from ESM2 8M to ESM2 150M consistently reduces prediction compression and improves ranking/tail behavior across the first two seeds, but residual regression-to-the-mean remains.

## Next Step

Run seed 2026 for the same 150M unweighted cross-attention setting, then generate a final multi-seed summary with mean and standard deviation.

Recommended final claim only after seed 2026:

- If seed 2026 follows the same pattern, report 150M as a stable improvement over 8M.
- If seed 2026 is weaker, report 150M as promising but seed-sensitive.

## Source Files

- Seed 42 report: `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_esm150M_unweighted/esm150M_cross_attention_report.md`
- Seed 42 metrics: `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_esm150M_unweighted/train_val_test_metrics_best_val_tail_mae.csv`
- Seed 123 report: `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_esm150M_unweighted_seed123/esm150M_cross_attention_report.md`
- Seed 123 predictions: `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_esm150M_unweighted_seed123/test_predictions_best_val_tail_mae.csv`
