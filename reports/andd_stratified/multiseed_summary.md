# ANDD Stratified Multi-Seed Validation: Unweighted vs Tail-Aware W2

## Comparison Design

- Dataset and split are fixed: `expanded_affinity_antibody_v2_stratified`.
- Architecture is fixed: all-CDR cross-attention.
- Training settings are fixed: `lr=3e-5`, `epochs=20`, `batch_size=1`.
- Seeds: `42`, `123`, `2026`.
- Unweighted control uses identical code with `tail_weight=regular_weight=1.0`.
- Tail-aware w2 uses train-P10/P90 tail weight `2.0` and regular weight `1.0`.
- Primary comparison policy: **best validation tail MAE checkpoint**.
- Historical unweighted seed-42 `epochs=10` result is not used in the formal multi-seed aggregate.

## Completion Status

- `unweighted` completed/evaluated seeds found: `[42, 123, 2026]` / expected `[42, 123, 2026]`.
- `tailaware_w2` completed/evaluated seeds found: `[42, 123, 2026]` / expected `[42, 123, 2026]`.

## Primary Policy Summary: Best Validation Tail MAE

| group | n seeds | MAE | RMSE | Spearman | pred std / true std | error vs true Pearson | below P10 MAE | above P90 MAE | tail MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tailaware_w2` | 3 | 0.9420 +/- 0.0020 | 1.2802 +/- 0.0137 | 0.4551 +/- 0.0085 | 0.5935 +/- 0.0350 | -0.8135 +/- 0.0222 | 1.6390 +/- 0.0754 | 1.6581 +/- 0.1270 | 1.6486 +/- 0.0373 |
| `unweighted` | 3 | 0.9224 +/- 0.0091 | 1.2466 +/- 0.0177 | 0.4750 +/- 0.0188 | 0.5106 +/- 0.0345 | -0.8608 +/- 0.0193 | 1.7708 +/- 0.0536 | 1.6601 +/- 0.1405 | 1.7155 +/- 0.0460 |

## Interpretation Rule

Only after all three seeds are present should we claim stability. A convincing w2 result should preserve lower tail MAE, healthier prediction spread, and a Pearson error trend closer to zero without consistently worsening overall MAE or Spearman.

## Outputs

- CSV: `outputs/andd_antibody_v2_stratified/multiseed/multiseed_summary.csv`
- Figure: `outputs/final_reports/figures/multiseed_w2_vs_baseline.png`
