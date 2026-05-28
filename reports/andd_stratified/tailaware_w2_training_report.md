# ANDD Antibody v2 Stratified: Tail-Aware Cross-Attention Weight=2.0

## Experiment

- Model architecture: unchanged all-CDR learnable cross-attention model.
- Loss: tail-weighted MSE; train targets at or below P10 and at or above P90 receive weight `2.0`, other rows receive weight `1.0`.
- Checkpoint selection: independently saved by validation MAE, Spearman, prediction-spread closeness to 1, and validation tail MAE.
- Comparison is on the same stratified antigen-level test split.
- This is a **single-seed experiment**; conclusions are provisional.

## Test Metrics By Validation-Selected Checkpoint

| model                            | kind       |   best_epoch |   rows |    MAE |   RMSE |   Spearman |   prediction_std |   true_std |   pred_std_true_std |   error_vs_true_Pearson |   low_target_MAE |   mid_target_MAE |   high_target_MAE |   below_train_p10_MAE |   above_train_p90_MAE |   tail_MAE |
|:---------------------------------|:-----------|-------------:|-------:|-------:|-------:|-----------:|-----------------:|-----------:|--------------------:|------------------------:|-----------------:|-----------------:|------------------:|----------------------:|----------------------:|-----------:|
| Tail-aware w2: best val MAE      | tailaware  |            6 |    117 | 0.9645 | 1.3215 |     0.3694 |           0.5331 |     1.3705 |              0.3890 |                 -0.9220 |           1.1315 |           0.4442 |            1.3837 |                1.7193 |                2.2178 |     1.9686 |
| Tail-aware w2: best val Spearman | tailaware  |           10 |    117 | 0.9912 | 1.3412 |     0.3872 |           0.6722 |     1.3705 |              0.4905 |                 -0.8759 |           1.4919 |           0.5237 |            1.0076 |                2.0838 |                1.6225 |     1.8532 |
| Tail-aware w2: best val spread   | tailaware  |           20 |    117 | 0.9426 | 1.2938 |     0.4478 |           0.7989 |     1.3705 |              0.5829 |                 -0.8222 |           1.1430 |           0.5486 |            1.1840 |                1.5632 |                1.8046 |     1.6839 |
| Tail-aware w2: best val tail MAE | tailaware  |           20 |    117 | 0.9426 | 1.2938 |     0.4478 |           0.7989 |     1.3705 |              0.5829 |                 -0.8222 |           1.1430 |           0.5486 |            1.1840 |                1.5632 |                1.8046 |     1.6839 |
| Baseline: pooled all-CDR         | baseline   |              |    117 | 0.9373 | 1.3056 |     0.3699 |           0.4345 |     1.3705 |              0.3170 |                 -0.9484 |           1.1674 |           0.3773 |            1.3367 |                1.8650 |                2.0887 |     1.9769 |
| Baseline: cross-attention        | baseline   |              |    117 | 0.9523 | 1.3008 |     0.3861 |           0.5379 |     1.3705 |              0.3925 |                 -0.9202 |           1.4182 |           0.3876 |            1.1147 |                2.0273 |                1.8483 |     1.9378 |
| Tail-aware w3: best val MAE      | comparison |              |    117 | 0.9698 | 1.3295 |     0.3583 |           0.6035 |     1.3705 |              0.4404 |                 -0.9004 |           1.1393 |           0.4787 |            1.3531 |                1.7149 |                2.1527 |     1.9338 |
| Tail-aware w3: best val Spearman | comparison |              |    117 | 0.9698 | 1.3295 |     0.3583 |           0.6035 |     1.3705 |              0.4404 |                 -0.9004 |           1.1393 |           0.4787 |            1.3531 |                1.7149 |                2.1527 |     1.9338 |
| Tail-aware w3: best val spread   | comparison |              |    117 | 1.0058 | 1.3523 |     0.4212 |           1.0218 |     1.3705 |              0.7456 |                 -0.7167 |           1.1979 |           0.7676 |            1.0789 |                1.5714 |                1.5747 |     1.5730 |
| Tail-aware w3: best val tail MAE | comparison |              |    117 | 0.9873 | 1.3442 |     0.4241 |           0.9983 |     1.3705 |              0.7284 |                 -0.7308 |           1.1133 |           0.6975 |            1.1867 |                1.4901 |                1.7391 |     1.6146 |

## Primary Tail-Aware Reading

The checkpoint selected by validation tail MAE is the primary tail-objective reading; it is not selected using test labels.

- Prediction spread: `0.5829` vs baseline `0.3925`; tail-aware improved spread toward 1.
- Error-vs-true Pearson: `-0.8222` vs baseline `-0.9202`; tail-aware improved compression trend toward 0.
- Below-P10 MAE: `1.5632` vs `2.0273`; improved.
- Above-P90 MAE: `1.8046` vs `1.8483`; improved.
- Spearman: `0.4478` vs `0.3861`; improved.
- Overall MAE: `0.9426` vs `0.9523`; improved.

If tail or spread metrics improve while overall MAE gets worse, that is a real tradeoff rather than a universal win.

## Comparison With Prior Tail-Aware Setting

The requested prior reference is `Tail-aware w3: best val tail MAE`. The primary current reading is `Tail-aware w2: best val tail MAE`.

- Overall MAE: current `0.9426` vs prior `0.9873`; improved.
- Spearman: current `0.4478` vs prior `0.4241`; improved.
- Prediction spread ratio: current `0.5829` vs prior `0.7284`.
- Error-vs-true Pearson: current `-0.8222` vs prior `-0.7308`.
- Average P10/P90 tail MAE: current `1.6839` vs prior `1.6146`; did not improve.

- Stability reading: the milder setting is more stable by the requested overall-MAE-and-Spearman criterion.

If the milder weighting preserves spread/tail gains while improving overall MAE or Spearman, it is the more stable variant. If tradeoffs remain, tail-aware objective is useful but not a final solution; structure/contact-aware features remain a serious next step.

## Output Files

- Checkpoint comparison CSV: `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_tailaware_w2/tailaware_w2_checkpoint_comparison.csv`
- Comparison figure: `outputs/final_reports/figures/tailaware_w2_checkpoint_comparison.png`
- Residual figure: `outputs/final_reports/figures/tailaware_w2_residual_vs_true.png`

## Conclusion Boundary

This experiment tests one conservative tail-weighted loss at one seed. It can reveal a promising tradeoff, but it cannot establish a stable improvement until repeated across seeds or validation policies.
