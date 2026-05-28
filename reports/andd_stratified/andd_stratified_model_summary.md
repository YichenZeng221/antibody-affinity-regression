# ANDD Antibody v2 Stratified Model Summary

## Dataset and Split

The final-stage benchmark uses the ANDD antibody v2 stratified antigen-level split.

| Split | Rows |
|---|---:|
| Train | 934 |
| Validation | 117 |
| Test | 117 |

The split was designed to keep antigen sequences disjoint across train, validation, and test while ensuring validation and test cover target-distribution tails.

## Main Models

| Model | MAE | RMSE | Spearman | pred std / true std | error vs true Pearson | Below P10 MAE | Above P90 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| All-CDR pooled | 0.9373 | 1.3056 | 0.3699 | 0.3170 | -0.9484 | 1.8650 | 2.0887 |
| All-CDR cross-attention | 0.9523 | 1.3008 | 0.3861 | 0.3925 | -0.9202 | 2.0273 | 1.8483 |
| Tail-aware w2 cross-attention, best validation tail MAE | 0.9426 | 1.2938 | 0.4478 | 0.5829 | -0.8222 | 1.5632 | 1.8046 |

## Multi-Seed Summary

Multi-seed validation compared unweighted cross-attention and tail-aware w2 cross-attention using seeds 42, 123, and 2026.

| Group | MAE | Spearman | pred std / true std | P10/P90 tail MAE |
|---|---:|---:|---:|---:|
| Unweighted cross-attention | 0.922 | 0.475 | 0.511 | 1.715 |
| Tail-aware w2 | 0.942 | 0.455 | 0.593 | 1.649 |

Tail-aware w2 improves prediction spread and tail MAE on average, but it does not consistently beat the unweighted baseline on overall MAE or Spearman.

## Interpretation

The results suggest that the main bottleneck is not a simple output activation or split artifact. Prediction compression appears across train, validation, and test. Tail-aware training improves the symptom but does not fully solve the underlying representation, calibration, and label-noise challenges.

## Included Summary Files

- `fit_metrics_by_split.csv`
- `feature_correlation_summary.csv`
- `multiseed_summary.csv`
- `tailaware_w2_checkpoint_comparison.csv`

