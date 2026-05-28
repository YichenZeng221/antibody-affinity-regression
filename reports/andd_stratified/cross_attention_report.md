# ANDD Antibody v2 Stratified Split: All-CDR Cross-Attention Report

## Experiment

- Dataset: ANDD antibody v2 stratified antigen-level split.
- Input: six standard AbNumber/IMGT CDRs as query tokens; antigen sequence as key/value tokens.
- Model: shared ESM2 + LoRA with learnable multi-head cross-attention.
- Loss and training hyperparameters: unchanged MSE baseline settings.
- Comparison: all-CDR pooled baseline trained on the same stratified split and test rows.

## Train-Defined Thresholds

- Low target: `target <= 4.5103`
- Mid target: `4.5103 < target <= 5.5218`
- High target: `target > 5.5218`
- Lower tail: `target <= train P10 = 3.3333`
- Upper tail: `target >= train P90 = 6.3204`

## Test Metrics

| model | rows | MAE | RMSE | Spearman | pred std / true std | error vs true Pearson |
|---|---:|---:|---:|---:|---:|---:|
| `cross_attention_all_cdrs` | 117 | 0.9523 | 1.3008 | 0.3861 | 0.3925 | -0.9202 |
| `all_cdrs_pooled` | 117 | 0.9373 | 1.3056 | 0.3699 | 0.3170 | -0.9484 |

## Train-Defined Target-Bin MAE

| model | low rows / MAE | mid rows / MAE | high rows / MAE |
|---|---:|---:|---:|
| `cross_attention_all_cdrs` | 38 / 1.4182 | 42 / 0.3876 | 37 / 1.1147 |
| `all_cdrs_pooled` | 38 / 1.1674 | 42 / 0.3773 | 37 / 1.3367 |

## Train-Defined Tail MAE

| model | below train P10 rows / MAE | above train P90 rows / MAE |
|---|---:|---:|
| `cross_attention_all_cdrs` | 16 / 2.0273 | 12 / 1.8483 |
| `all_cdrs_pooled` | 16 / 1.8650 | 12 / 2.0887 |

## Pooled Baseline Reference

These are the expected headline values supplied for the stratified pooled baseline:

- MAE/RMSE/Spearman: `0.9373` / `1.3056` / `0.3699`
- pred_std/true_std: `0.317`
- error_vs_true_Pearson: `-0.9484`
- low/mid/high target MAE: `1.1674` / `0.3773` / `1.3367`
- below-P10 / above-P90 tail MAE: `1.865` / `2.0887`

## Questions To Answer

1. Does cross-attention lower MAE/RMSE relative to pooled all-CDR on the same stratified test set?
2. Does `pred std / true std` move closer to 1 and does `error vs true Pearson` move closer to 0?
3. Does cross-attention reduce low/high target-bin MAE or P10/P90 tail MAE?
4. Does ranking ability, measured by Spearman, improve?

## Files

- Predictions: `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs/test_predictions.csv`
- Checkpoint: `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs/checkpoints/seqproft_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_lr3e-5_e10.pt`
