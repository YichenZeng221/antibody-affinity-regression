# ESM2 650M Unweighted bs4 Stopped Run Summary

## Setup

- Model: ESM2 650M + LoRA + all-six-CDR cross-attention
- Dataset: ANDD antibody v2 stratified split
- Loss: unweighted regression
- Seed: 2026
- Batch size: 4
- Learning rate: 1e-5
- Planned epochs: 150
- Device: RTX 5090
- Status: stopped early around epoch 43

## Best Observed Validation Result

The strongest observed validation result was around epoch 30:

- val_MAE = 0.9277
- val_RMSE = 1.3669
- val_Spearman = 0.4565
- val_MAE_fold_error ~= 8.5x

## Later Trend

Validation performance weakened after the earlier plateau:

| epoch | val_MAE | val_Spearman |
|---:|---:|---:|
| 40 | 0.9866 | 0.4150 |
| 41 | 0.9691 | 0.4095 |
| 42 | 0.9960 | 0.3800 |

## Comparison Context

| model | MAE | Spearman | tail_MAE |
|---|---:|---:|---:|
| ESM2 150M unweighted, seed 42 | 0.9047 | 0.5221 | 1.5150 |
| ESM2 150M unweighted, seed 123 | 0.8815 | 0.5532 | 1.6043 |
| ESM2 650M unweighted bs14 pilot | approximately 0.99 | approximately 0.45 | not evaluated |
| ESM2 650M unweighted bs4 stopped run | 0.9277 validation MAE | 0.4565 validation Spearman | not evaluated |

The 150M values are test metrics. The 650M bs4 values are best observed validation metrics from the stopped run. They are included as context and should not be interpreted as a direct test-set comparison.

## Conclusion

The ESM2 650M bs4 run improved over the earlier ESM2 650M bs14 pilot, especially in validation MAE, but it still did not show a convincing advantage over the ESM2 150M unweighted baseline. Validation Spearman plateaued around 0.45 and later declined, so the run was stopped early.

This result suggests that scaling from ESM2 150M to ESM2 650M does not improve generalization under the current sequence-only all-six-CDR cross-attention setup.

## Recommended Next Direction

Stop further ESM2 650M scaling for now.

Treat ESM2 150M unweighted cross-attention as the strongest current sequence-only baseline and move toward:

- structure/contact/interface-aware features
- external baseline comparison, such as AbRank
