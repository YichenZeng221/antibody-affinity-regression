# Final Results Index

## 1. Important Reports

### Final Project Summaries

| Report | Path |
|---|---|
| English final project summary | `outputs/final_reports/unified_no_high_risk_project_summary.md` |
| Chinese final project summary | `outputs/final_reports/unified_no_high_risk_project_summary_zh.md` |
| Final results index | `outputs/final_reports/final_results_index.md` |

### Main Experiment Reports

| Report | Path |
|---|---|
| Whole-sequence error analysis | `outputs/error_analysis/unified_no_high_risk/error_analysis_report.md` |
| All-CDR pooled baseline report | `outputs/cdr_aware/unified_no_high_risk/cdr_aware_report.md` |
| CDR ablation summary | `outputs/cdr_ablation/unified_no_high_risk/cdr_ablation_summary.md` |
| Simple interaction matrix report | `outputs/interaction_aware/unified_no_high_risk/hcdr3_lcdr3_antigen/interaction_report.md` |
| All-CDR cross-attention report | `outputs/cross_attention/unified_no_high_risk/all_cdrs_antigen/cross_attention_report.md` |
| SeqProFT official GitHub comparison | `outputs/reproducibility/seqproft_github_comparison.md` |

### ANDD Expansion And Today's Follow-Up Reports

| Report | Path |
|---|---|
| ANDD antibody v2 error analysis | `outputs/andd_antibody_v2/error_analysis/error_analysis_report.md` |
| ANDD target distribution diagnosis | `outputs/andd_antibody_v2/target_distribution/target_distribution_report.md` |
| ANDD all-CDR pooled linear calibration | `outputs/andd_antibody_v2/calibration/calibration_report.md` |
| ANDD stratified antigen-level split summary | `data/processed_affinity/expanded_affinity_antibody_v2_stratified/split_summary.md` |
| ANDD stratified all-CDR pooled baseline | `outputs/andd_antibody_v2_stratified/all_cdr_pooled/andd_antibody_v2_stratified_all_cdr_pooled_report.md` |
| ANDD stratified all-CDR cross-attention | `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs/cross_attention_report.md` |

## 2. Core Model Metrics

All metrics below are from the current `unified_no_high_risk` experiment branch under the antigen-sequence group split.

| Model | Main Input | MAE | RMSE | Spearman | pred_std / true_std | error vs true Pearson | high-target MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| Whole-sequence baseline | heavy + light + antigen | 1.1083 | 1.3765 | 0.4557 | 0.4825 | -0.8787 | 1.7949 |
| Pooled all-CDR baseline | HCDR1/2/3 + LCDR1/2/3 + antigen | 0.9975 | 1.2495 | 0.4497 | 0.3651 | -0.9321 | 1.6401 |
| Pooled HCDR3+LCDR3 | HCDR3 + LCDR3 + antigen | 1.0204 | 1.2570 | 0.4438 | 0.3383 | -0.9418 | 1.5780 |
| Pooled heavy CDRs | HCDR1/2/3 + antigen | 1.0462 | 1.3353 | 0.4281 | 0.3251 | -0.9471 | 1.9162 |
| Pooled light CDRs | LCDR1/2/3 + antigen | 1.0554 | 1.3482 | 0.3799 | 0.3538 | -0.9353 | 1.8796 |
| Pooled HCDR3-only | HCDR3 + antigen | 1.0641 | 1.3308 | 0.4176 | 0.4611 | -0.8876 | 1.7880 |
| Simple interaction matrix | HCDR3 + LCDR3 + antigen dot-product summaries | 1.0504 | 1.2839 | 0.4126 | 0.2958 | -0.9560 | 1.6657 |
| All-CDR cross-attention | all six CDR queries -> antigen key/value | 1.0515 | 1.3156 | 0.5018 | 0.7855 | -0.6617 | 1.4740 |

## 3. Current Best MAE/RMSE Model

The current best model for overall absolute error is:

```text
Pooled all-CDR baseline
```

| Metric | Value |
|---|---:|
| MAE | 0.9975 |
| RMSE | 1.2495 |
| Spearman | 0.4497 |

Interpretation:

- Standard IMGT CDR-focused inputs improve overall absolute error over whole-chain pooled inputs.
- Using all six pooled CDRs remains the strongest current choice when MAE/RMSE are the primary goal.

## 4. Current Best Spearman Model

The current best ranking model is:

```text
All-CDR cross-attention
```

| Metric | Value |
|---|---:|
| Spearman | 0.5018 |
| MAE | 1.0515 |
| RMSE | 1.3156 |

Interpretation:

- Learnable CDR-to-antigen interaction improves ranking quality.
- This model does not yet beat pooled all-CDR on MAE/RMSE, so ranking improvement and absolute-error calibration are not yet aligned.

## 5. Current Best Regression-To-Mean Relief

The model that currently most clearly reduces regression-to-mean is:

```text
All-CDR cross-attention
```

| Diagnostic | Value | Why It Matters |
|---|---:|---|
| pred_std / true_std | 0.7855 | Closer to 1 means prediction spread is closer to target spread. |
| error vs true Pearson | -0.6617 | Closer to 0 means weaker systematic overpredict-low / underpredict-high behavior. |
| high-target MAE | 1.4740 | Best current high-target MAE among the listed main models. |

Compared with pooled all-CDR:

| Diagnostic | Pooled all-CDR | All-CDR cross-attention |
|---|---:|---:|
| pred_std / true_std | 0.3651 | 0.7855 |
| error vs true Pearson | -0.9321 | -0.6617 |
| high-target MAE | 1.6401 | 1.4740 |

Interpretation:

- Cross-attention does not yet win average MAE/RMSE.
- It does produce a healthier prediction range and a weaker regression-to-mean pattern.

## 6. ANDD Antibody v2: Original Split Diagnostics

ANDD antibody-only v2 is a separate benchmark from `unified_no_high_risk`, so its numerical results should be interpreted within its own test set rather than directly ranked against the original 605-row benchmark.

### Original ANDD All-CDR Pooled Baseline

| Model / adjustment | MAE | RMSE | Spearman | pred std / true std | error vs true Pearson | low-target MAE | high-target MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw all-CDR pooled | 0.9066 | 1.1281 | 0.3817 | 0.3772 | -0.9262 | 1.2603 | 1.0981 |
| Post-hoc linearly calibrated | 0.9015 | 1.1424 | 0.3817 | 0.5302 | -0.8508 | 1.0173 | 1.2347 |

Linear calibration was fitted using validation predictions only. It slightly improved test MAE, prediction spread, regression-to-the-mean diagnostics, and low-target MAE, but worsened RMSE and high-target MAE. This is a calibration aid, not a complete modeling fix.

### Target Distribution Diagnosis

- Train tertiles are approximately balanced by construction: low/mid/high counts are `312 / 311 / 311`.
- Therefore, coarse low/mid/high imbalance in train does not adequately explain prediction compression.
- Truly extreme tails are still sparse: using train P10/P90, each tail contains about 10% of training rows.
- The original validation split did not have zero high-tail examples under the global-P95 definition: it contained `4` global-P95 tail rows. However, its high-tail coverage was weak and its maximum target was only `6.9811`, while the later stratified validation split reached `11.8155`.

## 7. ANDD Stratified Antigen-Level Split

The stratified split was created to make tail evaluation less accidental while maintaining leakage control.

| Split | Rows | Antigen groups | Target min | Target max | Global P5 tail rows | Global P95 tail rows |
|---|---:|---:|---:|---:|---:|---:|
| train | 934 | 568 | 2.0449 | 8.5343 | 42 | 43 |
| val | 117 | 117 | 2.0031 | 11.8155 | 10 | 9 |
| test | 117 | 117 | 2.0349 | 10.9208 | 7 | 7 |

- Global target P5/P95 thresholds: `2.8900 / 6.5838`.
- Both validation and test explicitly cover both target tails.
- Antigen-sequence overlap is `0` for train/val, train/test, and val/test.

### Same-Split Model Comparison

| Model on stratified test | MAE | RMSE | Spearman | pred std / true std | error vs true Pearson | low-target MAE | high-target MAE | below-P10 MAE | above-P90 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All-CDR pooled | 0.9373 | 1.3056 | 0.3699 | 0.3170 | -0.9484 | 1.1674 | 1.3367 | 1.8650 | 2.0887 |
| All-CDR cross-attention | 0.9523 | 1.3008 | 0.3861 | 0.3925 | -0.9202 | 1.4182 | 1.1147 | 2.0273 | 1.8483 |

Interpretation:

- On the same stratified split, pooled all-CDR remains slightly better by MAE.
- Cross-attention slightly improves RMSE and Spearman.
- Cross-attention improves prediction spread and makes `error vs true Pearson` less negative, indicating reduced regression-to-the-mean.
- Cross-attention improves high-target and upper-tail MAE, but worsens low-target and lower-tail MAE.
- Making the split cover tails did not remove prediction compression in the pooled model; regression-to-the-mean is not merely a split artifact.

## 8. Current Main Conclusions

1. Whole-chain pooled ESM2+LoRA is a useful baseline, but it shows strong regression-to-mean.
2. Standard AbNumber + IMGT CDR extraction enables a biologically focused input representation.
3. On the original `unified_no_high_risk` benchmark, pooled all-CDR input is best for overall MAE/RMSE.
4. Pooled HCDR3+LCDR3 is close to pooled all-CDR, suggesting CDR3 loops carry a large part of the affinity signal.
5. Simple dot-product interaction matrix summary statistics do not improve the pooled CDR3 baseline.
6. Learnable all-CDR cross-attention improves ranking, high-target behavior, prediction spread, and regression-to-mean diagnostics.
7. On ANDD, train tertiles are not obviously imbalanced; extreme tails are limited but coarse target imbalance is not the sole explanation.
8. Stratified antigen splitting gives val/test explicit tail coverage without antigen leakage, yet pooled predictions remain compressed.
9. Linear calibration and cross-attention each relieve part of the compression problem in different ways, but neither fully resolves overall error and both involve trade-offs.
10. The remaining limitation is not an output-head range bug and is not only a split artifact; it points toward interaction representation and tail-aware calibration/training.

## 9. Recommended Next Stage

The next stage should preserve the useful cross-attention interaction signal while improving calibration.

Recommended experiments:

1. Continue learnable interaction modeling on ANDD stratified data, using the same-split pooled comparison as the reference.
2. Calibration-aware or tail-aware training/evaluation, with explicit monitoring of lower and upper tails separately.
3. Weighted loss or sampling variants only with validation-based selection, since the first weighted attempt involved trade-offs.
4. Validation checkpoint selection aligned with the final goal:
   - MAE/RMSE if absolute error is primary,
   - Spearman/high-target metrics if ranking or strong-binding retrieval is primary.
5. Structure/contact-aware supervision or interface features once the data path is reliable.
6. Continued audit of peptide-like, epitope-like, and assay/source-noisy samples.

## 10. Short Meeting Update

> I expanded the benchmark to conservative ANDD antibody data and tested a stratified antigen-level split with explicit target-tail coverage and zero antigen overlap. The pooled all-CDR model still had the better overall MAE, while all-CDR cross-attention improved ranking, prediction spread, and upper-tail error on the same split. Together with the calibration experiment, this suggests regression-to-the-mean is not a code bug or only a split artifact; the next step is better learnable interaction modeling with tail-aware calibration or structure/contact information.
