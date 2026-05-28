# ANDD Antibody v2 Stratified Split: All-CDR Pooled Baseline

## Question

Does stratified antigen-level split with explicit validation/test tail coverage change the observed regression-to-the-mean behavior?

## Controlled Setting

- Model: existing all-CDR pooled shared ESM2 + LoRA regressor.
- Loss: unchanged MSE loss.
- Learning rate/epochs/seed: unchanged from the prior all-CDR pooled baseline.
- Only the antigen-level split is changed.
- Low/mid/high target bins and P10/P90 tails are defined from each experiment's train split.

## Test Metrics

| split version | MAE | RMSE | Spearman | pred std / true std | error vs true Pearson |
|---|---:|---:|---:|---:|---:|
| original antigen split | 0.9066 | 1.1281 | 0.3817 | 0.3772 | -0.9262 |
| stratified antigen split | 0.9373 | 1.3056 | 0.3699 | 0.3170 | -0.9484 |

## Train-Defined Target-Bin MAE

| split version | low rows / MAE | mid rows / MAE | high rows / MAE |
|---|---:|---:|---:|
| original antigen split | 43 / 1.1683 | 31 / 0.3557 | 43 / 1.0420 |
| stratified antigen split | 38 / 1.1674 | 42 / 0.3773 | 37 / 1.3367 |

## Train-Defined Tail MAE

| split version | below train P10 rows / MAE | above train P90 rows / MAE |
|---|---:|---:|
| original antigen split | 10 / 1.8073 | 17 / 1.3877 |
| stratified antigen split | 16 / 1.8650 | 12 / 2.0887 |

## Reading Guide

- `pred std / true std` closer to 1 indicates less compressed prediction range.
- `error vs true Pearson` closer to 0 indicates weaker systematic regression-to-the-mean.
- Tail MAE now measures behavior in tails represented according to the training distribution.
- If tail coverage improves but compression remains strong, split alone is unlikely to be the main bottleneck.
