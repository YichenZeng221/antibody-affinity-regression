# ANDD Antibody v2 Affinity Regression: Phase-1 Final Report

## 1. Executive Summary

This phase established a conservative, leakage-aware ANDD antibody-only affinity
benchmark and used it to diagnose the main limitation of the current
SeqProFT-style ESM2 + LoRA modeling pipeline.

The central finding is:

> The dominant current failure mode is **regression-to-the-mean**: low-affinity
> targets are systematically predicted too high, while high-affinity targets
> are predicted too low.

This conclusion is supported by several independent checks:

- On the original ANDD antibody v2 all-CDR pooled baseline, test
  `pred_std / true_std = 0.3772` and
  `error_vs_true_Pearson = -0.9262`.
- On the stratified antigen-level split, both pooled all-CDR and all-CDR
  cross-attention still produce compressed predictions.
- Train, validation, and test all show compression; this does not look like
  classic overfitting where train fit is healthy and only held-out data fail.
- The regression heads return unrestricted scalar outputs; there is no
  `sigmoid`, `tanh`, output `clamp`, or prediction inverse-transform bug
  constraining the range.
- Stratified splitting improves tail coverage while preserving zero antigen
  leakage, but does not remove compression. The behavior is therefore not a
  simple split artifact.

The modeling results are nuanced:

- Pooled all-CDR remains slightly better for overall MAE on the stratified
  benchmark (`0.9373` versus `0.9523` for cross-attention).
- Cross-attention improves Spearman, prediction spread, residual bias, and
  upper-tail behavior on the same split.
- Tail-aware loss improves spread and combined tail MAE, but three-seed
  validation shows it does not consistently improve MAE/RMSE/Spearman over
  the unweighted model.
- Structure and contact features are technically feasible for a conservative
  subset. CDR3 geometry gives a small incremental improvement over tail-aware
  predictions inside contact-covered subsets, but simple contact counts plus
  Ridge correction do not solve prediction compression.

The honest interpretation is that the remaining bottleneck most likely reflects
a combination of biological representation limits, sparse/noisy tail labels,
assay heterogeneity, and the difficulty of absolute Kd regression. These
results are **consistent with** exploring ranking-oriented objectives in future
work; they do not prove that absolute Kd regression is intrinsically defective.

## 2. Dataset and Task Definition

### 2.1 Prediction Task

- Task: antibody-antigen binding affinity regression.
- Input representation in the formal ANDD baseline: six standard CDR sequences
  (`HCDR1`, `HCDR2`, `HCDR3`, `LCDR1`, `LCDR2`, `LCDR3`) plus
  `antigen_sequence`.
- Target: `neg_log10_affinity_candidate = -log10(Kd[M])`.
- Interpretation: larger target values represent stronger binding.

### 2.2 From ANDD Source to Conservative Antibody Benchmark

| Stage | Rows | Interpretation |
|---|---:|---|
| ANDD source workbook | 48,800 | Heterogeneous antibody/nanobody source data |
| Tier 1 affinity candidates | 4,382 | Experimental quantitative Kd-like rows with usable sequences |
| Antibody candidates | 3,116 | Antibody-only candidate pool |
| Conservative antibody `keep_safe` rows | 1,168 | Formal antibody-only v2 modeling pool |

Important filtering decisions:

- Antibody and nanobody rows were kept as separate tasks.
- Predicted affinity rows, including ANTIPASTI-derived labels, were excluded
  from the primary experimental Kd benchmark.
- Extreme Kd, sequence-quality, duplicate, and overlap concerns were audited
  before defining the conservative 1,168-row set.

### 2.3 Standard CDR Extraction

Formal CDR-aware modeling used standard `AbNumber + IMGT` extraction rather
than source-provided CDR fields or fixed-position slicing.

| Split | Rows | Heavy CDR success | Light CDR success | Both success |
|---|---:|---:|---:|---:|
| Train | 934 | 934 | 934 | 934 |
| Validation | 117 | 117 | 117 | 117 |
| Test | 117 | 117 | 117 | 117 |

Thus the ANDD antibody v2 CDR-aware benchmark has no CDR extraction failure
rows.

### 2.4 Original Antigen-Group Split

| Split | Rows | Antigen groups | Target mean | Target min | Target max |
|---|---:|---:|---:|---:|---:|
| Train | 934 | 644 | 4.9467 | 2.0031 | 11.8155 |
| Validation | 117 | 79 | 4.8944 | 2.0938 | 6.9811 |
| Test | 117 | 79 | 5.0281 | 2.1533 | 8.4692 |

Leakage check:

- Train/validation antigen overlap: `0`
- Train/test antigen overlap: `0`
- Validation/test antigen overlap: `0`

### 2.5 Target Distribution Diagnosis

Using tertiles defined only from the original training targets:

| Split | Low target | Mid target | High target |
|---|---:|---:|---:|
| Train | 312 (33.4%) | 311 (33.3%) | 311 (33.3%) |
| Validation | 38 (32.5%) | 38 (32.5%) | 41 (35.0%) |
| Test | 43 (36.8%) | 31 (26.5%) | 43 (36.8%) |

The coarse low/mid/high ranges are not severely imbalanced. However, true
tails remain sparse: under train P10/P90 definitions, each train tail contains
94 rows. The original validation set also had weak high-end coverage, with
maximum target `6.9811`, despite the training maximum reaching `11.8155`.

## 3. Experimental Timeline

| Phase | Question | Result |
|---|---|---|
| Dataset audit and QC | Can ANDD supply reliable experimental antibody affinity rows? | 1,168 conservative antibody rows selected |
| Standard CDR extraction | Can CDR-aware inputs be standardized safely? | AbNumber + IMGT succeeded for 1,168 / 1,168 rows |
| Original-split pooled baseline | Is the conservative dataset trainable? | MAE `0.9066`, but severe prediction compression |
| Output/head and data checks | Is compression a coding or leakage issue? | No range-limiting output operation found; antigen overlap `0` |
| Linear calibration | Can output-scale adjustment reduce compression? | Spread and low-target error improved; tradeoff remained |
| Stratified antigen split | Does stronger tail coverage remove the issue? | No; compression persisted on leakage-safe tail-covered split |
| Cross-attention baseline | Does learnable CDR-antigen interaction help? | Better ranking/spread/upper-tail behavior, not best MAE |
| Tail-aware w3/w2 | Can objective weighting reduce tails/compression? | Helpful in single seed; w2 more balanced than w3 |
| Multi-seed validation | Is w2 consistently better? | Better tail/spread, not stable overall MAE/Spearman win |
| Contact/interface audit | Are physical interface features available? | Basic features extracted for 472 unambiguous rows |
| CDR mapping validation | Can CDR-specific contacts be trusted? | All-CDR safe: 422 rows; HCDR3+LCDR3 safe: 467 rows |
| CDR3 contact augmentation | Do small true geometry features add value? | Small subset-only gains for tail-aware w2; compression unresolved |

## 4. Main Failure Mode: Regression-to-the-Mean

### 4.1 Original ANDD Pooled Baseline Error Pattern

The original ANDD all-CDR pooled model establishes the problem clearly:

| Metric | Value |
|---|---:|
| Test rows | 117 |
| MAE | 0.9066 |
| RMSE | 1.1281 |
| Spearman | 0.3817 |
| Prediction std | 0.4665 |
| `pred_std / true_std` | 0.3772 |
| `error_vs_true_Pearson` | -0.9262 |

| Target bin | Rows | MAE | Mean prediction error |
|---|---:|---:|---:|
| Low target | 39 | 1.2603 | +1.2603 |
| Mid target | 39 | 0.3614 | +0.0620 |
| High target | 39 | 1.0981 | -1.0981 |

Here, positive low-target errors mean weak-binding examples are overpredicted,
and negative high-target errors mean strong-binding examples are underpredicted.
The model performs well near the center but compresses both extremes toward its
mean prediction.

### 4.2 Train Versus Validation/Test Diagnosis on Stratified Data

Compression is not limited to held-out evaluation rows:

| Model | Split | MAE | Spearman | `pred_std / true_std` | `error_vs_true_Pearson` |
|---|---|---:|---:|---:|---:|
| Pooled all-CDR | Train | 0.7497 | 0.5831 | 0.4351 | -0.9041 |
| Pooled all-CDR | Validation | 0.9588 | 0.4239 | 0.3114 | -0.9504 |
| Pooled all-CDR | Test | 0.9373 | 0.3699 | 0.3170 | -0.9484 |
| All-CDR cross-attention | Train | 0.7597 | 0.5403 | 0.4584 | -0.8898 |
| All-CDR cross-attention | Validation | 0.9825 | 0.4188 | 0.3831 | -0.9239 |
| All-CDR cross-attention | Test | 0.9523 | 0.3861 | 0.3925 | -0.9202 |

Because train predictions themselves are substantially compressed, the evidence
does not fit a simple overfitting story. It is more consistent with an
objective and/or representation bottleneck, compounded by tail-label difficulty.

## 5. What We Ruled Out

### 5.1 Output Activation or Prediction Range Restriction

Code inspection of the pooled CDR and cross-attention regression models shows:

- Pooled CDR model output: one unrestricted `nn.Linear(..., 1)` scalar.
- Cross-attention output: `Linear -> GELU -> Dropout -> Linear -> scalar`.
- Neither prediction path applies `sigmoid`, `tanh`, output `clamp`, or a
  bounded output transform.
- The `clamp(min=1e-9)` operation in pooling protects only the token-count
  denominator from division by zero; it does not clamp predictions.

Therefore, observed prediction compression is not caused by an output
activation or hard range restriction.

### 5.2 Target Transformation or Inverse-Transform Error

- Configured target is consistently `neg_log10_affinity_candidate`.
- Reports and prediction CSVs compare
  `true_neg_log10_affinity` with `predicted_neg_log10_affinity`.
- No inverse-transform step is applied before model metric reporting.

There is no evidence in the reviewed pipeline of a target normalization or
inverse-transform mismatch creating the compression pattern.

### 5.3 Simple Leakage Explanation

Both original and stratified datasets use antigen-sequence group splitting.
For the formal stratified split:

- Train/validation antigen overlap: `0`
- Train/test antigen overlap: `0`
- Validation/test antigen overlap: `0`

Leakage is therefore not a plausible explanation for either the remaining
errors or the observed regression-to-the-mean.

### 5.4 Simple Split Artifact

The stratified split explicitly improved validation and test tail coverage:

| Split | Rows | Antigen groups | Target min | Target max | Global P5 rows | Global P95 rows |
|---|---:|---:|---:|---:|---:|---:|
| Train | 934 | 568 | 2.0449 | 8.5343 | 42 | 43 |
| Validation | 117 | 117 | 2.0031 | 11.8155 | 10 | 9 |
| Test | 117 | 117 | 2.0349 | 10.9208 | 7 | 7 |

Even after this improvement, the pooled model remains strongly compressed
(`pred_std / true_std = 0.3170`, `error_vs_true_Pearson = -0.9484`).
The failure mode is therefore not explained by the original split alone.

## 6. Model Results

### 6.1 Benchmark Boundary

There is no completed ANDD whole-sequence baseline prediction file in the
current ANDD configs; the formal ANDD model baseline begins with standard
all-CDR pooled inputs. Earlier whole-sequence/CDR studies on
`unified_no_high_risk` motivated this representation choice but are not directly
comparable to the ANDD test sets.

### 6.2 Original ANDD Antibody v2 Baseline

| Model | Split | MAE | RMSE | Spearman | `pred_std / true_std` | `error_vs_true_Pearson` | Low MAE | High MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| All-CDR pooled | Original antigen split | 0.9066 | 1.1281 | 0.3817 | 0.3772 | -0.9262 | 1.2603 | 1.0981 |

This model demonstrates that the conservative ANDD set is trainable, but also
that absolute-error improvement alone is insufficient: the prediction range is
severely compressed.

### 6.3 Same-Split Stratified Baselines

The following comparison is fair within the stratified split because both
models use the same dataset split and held-out test rows:

| Model | MAE | RMSE | Spearman | `pred_std / true_std` | `error_vs_true_Pearson` | Low MAE | High MAE | Below P10 MAE | Above P90 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All-CDR pooled | 0.9373 | 1.3056 | 0.3699 | 0.3170 | -0.9484 | 1.1674 | 1.3367 | 1.8650 | 2.0887 |
| All-CDR cross-attention | 0.9523 | 1.3008 | 0.3861 | 0.3925 | -0.9202 | 1.4182 | 1.1147 | 2.0273 | 1.8483 |

Interpretation:

- Pooled all-CDR is slightly better by MAE.
- Cross-attention is slightly better by RMSE and Spearman.
- Cross-attention moves prediction spread closer to the true range and reduces
  the negative residual trend.
- Cross-attention improves high-target and above-P90 error, while worsening
  low-target and below-P10 error.

Thus learnable interaction is useful, particularly for ranking and the
high-affinity tail, but it is not yet a universal absolute-error improvement.

## 7. Calibration Result

Post-hoc linear calibration was fitted using only validation predictions on the
original ANDD split:

```text
calibrated_pred = 1.405797 * raw_pred - 2.274080
```

| Prediction | MAE | RMSE | Spearman | `pred_std / true_std` | `error_vs_true_Pearson` | Low MAE | High MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw pooled prediction | 0.9066 | 1.1281 | 0.3817 | 0.3772 | -0.9262 | 1.2603 | 1.0981 |
| Linearly calibrated | 0.9015 | 1.1424 | 0.3817 | 0.5302 | -0.8508 | 1.0173 | 1.2347 |

Calibration improves scale, residual bias, low-target MAE, and slightly improves
overall MAE. It worsens RMSE and high-target MAE and cannot improve Spearman
when the learned slope is positive. It is useful evidence that output scale is
part of the symptom, but it does not add biological information or solve the
tail problem.

## 8. Tail-Aware Training and Multi-Seed Validation

All experiments below use the stratified antigen-level split and all-CDR
cross-attention architecture. Tail thresholds are computed from training data
only: `P10 = 3.3333`, `P90 = 6.3204`.

### 8.1 Single-Seed Results

| Model / Validation-Selected Checkpoint | MAE | RMSE | Spearman | `pred_std / true_std` | `error_vs_true_Pearson` | Below P10 MAE | Above P90 MAE | Tail MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unweighted cross-attention, e10 | 0.9523 | 1.3008 | 0.3861 | 0.3925 | -0.9202 | 2.0273 | 1.8483 | 1.9378 |
| Tail-aware w3.0, best val tail MAE | 0.9873 | 1.3442 | 0.4241 | 0.7284 | -0.7308 | 1.4901 | 1.7391 | 1.6146 |
| Tail-aware w2.0, best val tail MAE | 0.9426 | 1.2938 | 0.4478 | 0.5829 | -0.8222 | 1.5632 | 1.8046 | 1.6839 |

Single-seed reading:

- Weight `3.0` strongly improves spread and tails but gives up overall MAE/RMSE.
- Weight `2.0` appears more balanced and, at seed 42, improves every headline
  metric relative to the historical e10 unweighted cross-attention reference.
- This observation was promising, but required multi-seed verification before
  becoming a stable claim.

### 8.2 Controlled Multi-Seed Validation

The controlled comparison fixes architecture, split, learning rate
(`3e-5`), epochs (`20`), batch size (`1`), and checkpoint policy
(`best validation tail MAE`), and evaluates seeds `42`, `123`, `2026`.

| Model Group | Seeds | MAE | RMSE | Spearman | `pred_std / true_std` | `error_vs_true_Pearson` | Below P10 MAE | Above P90 MAE | Tail MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unweighted cross-attention, e20 | 3 | **0.9224 +/- 0.0091** | **1.2466 +/- 0.0177** | **0.4750 +/- 0.0188** | 0.5106 +/- 0.0345 | -0.8608 +/- 0.0193 | 1.7708 +/- 0.0536 | 1.6601 +/- 0.1405 | 1.7155 +/- 0.0460 |
| Tail-aware w2.0, e20 | 3 | 0.9420 +/- 0.0020 | 1.2802 +/- 0.0137 | 0.4551 +/- 0.0085 | **0.5935 +/- 0.0350** | **-0.8135 +/- 0.0222** | **1.6390 +/- 0.0754** | **1.6581 +/- 0.1270** | **1.6486 +/- 0.0373** |

| W2 minus unweighted | Delta | Reading |
|---|---:|---|
| MAE | +0.0196 | Worse overall absolute error |
| RMSE | +0.0335 | Worse overall squared error |
| Spearman | -0.0199 | Does not preserve the ranking advantage |
| `pred_std / true_std` | +0.0829 | Healthier, less compressed spread |
| `error_vs_true_Pearson` | +0.0473 | Less systematic residual trend |
| Below P10 MAE | -0.1318 | Meaningful lower-tail improvement |
| Above P90 MAE | -0.0020 | Essentially tied |
| Tail MAE | -0.0669 | Modest combined tail improvement |

Final tail-aware conclusion:

> Tail-aware loss improves prediction spread and slightly improves tail MAE,
> especially below P10, but it does not consistently improve MAE/RMSE/Spearman
> across seeds.

The single-seed w2 result is therefore best viewed as a useful direction, not
as the final best general-purpose model.

## 9. Contact and Interface Feature Analysis

### 9.1 Structure and Chain-Mapping Availability

The SAbDab external structure archive provides broad file coverage, but
chain-level ambiguity limits conservative extraction:

| Availability item | Rows |
|---|---:|
| Total stratified ANDD rows | 1,168 |
| External SAbDab raw/imgt/chothia structure match | 1,168 / 1,168 |
| Complete H/L/antigen chain metadata option | 1,167 / 1,168 |
| Unambiguous viable H/L/antigen chain mapping | 472 / 1,168 |
| Ambiguous multiple viable chain mappings, not guessed | 695 / 1,168 |
| No viable complete chain mapping found | 1 / 1,168 |

This is a crucial boundary: contact analyses below are subset analyses, not
replacement full-benchmark evaluations.

### 9.2 Basic Interface Geometry Pilot

Basic geometry features were successfully extracted for all 472 unambiguous
pilot rows:

- `min_ab_ag_distance`
- `contact_count_4A`, `contact_count_5A`, `contact_count_8A`
- antibody/antigen/heavy/light interface residue counts at 5 A

| Feature vs target affinity | Rows | Pearson | Spearman |
|---|---:|---:|---:|
| `min_ab_ag_distance` | 472 | -0.023 | -0.068 |
| `contact_count_5A` | 472 | 0.198 | 0.173 |

High-tail rows have larger mean `contact_count_5A` than low-tail rows
(`63.586` versus `49.091`), but this is weak exploratory signal and cannot
explain the full compression problem on its own.

## 10. CDR Mapping Validation

Standard IMGT CDR annotations were mapped back to structure residue IDs under
strict sequence/coordinate validation rules.

| Mapping target | Safe rows | Rate within 472-row pilot |
|---|---:|---:|
| All six CDRs usable for preliminary contacts | 422 | 89.41% |
| HCDR3 usable | 468 | 99.15% |
| LCDR3 usable | 470 | 99.58% |
| HCDR3 + LCDR3 jointly usable | 467 | 98.94% |

| Failure reason | Rows |
|---|---:|
| Insertion/deletion ambiguity | 34 |
| Unresolved residues | 16 |
| Chain sequence mismatch | 2 |
| Missing residue numbering | 0 |
| Antigen chain mismatch | 0 |
| Missing CDR annotation | 0 |

CDR3 features are therefore a technically practical first structure-aware
extension because HCDR3 and LCDR3 have substantially higher safe mapping
coverage than the all-six-CDR aggregation.

Preliminary one-variable correlations are weak:

| Feature | Outcome | Rows | Pearson | Spearman |
|---|---|---:|---:|---:|
| `all_cdr_contact_count_5A` | Target affinity | 422 | 0.157 | 0.148 |
| `hcdr3_contact_count_5A` | Target affinity | 468 | 0.110 | 0.059 |
| `lcdr3_contact_count_5A` | Target affinity | 470 | 0.059 | 0.070 |
| `cdr_min_distance` | Target affinity | 422 | -0.004 | -0.028 |

## 11. CDR3-Contact Residual Correction Baseline

### 11.1 Experimental Boundary

- This experiment evaluates only contact-covered rows with validated
  CDR-to-structure mapping.
- It does not retrain the neural sequence models.
- It fits a small `Ridge(alpha=1.0)` residual correction on the relevant
  training subset:

```text
residual = target - sequence_prediction
corrected_prediction = sequence_prediction + predicted_residual
```

- Results must not be compared directly as if they were full 1,168-row
  benchmark results.

### 11.2 Contact-Safe Subsets

| Subset | Total safe rows | Train rows used | Test rows used |
|---|---:|---:|---:|
| HCDR3+LCDR3 contact-safe | 467 | 356 | 58 |
| All-CDR contact-safe | 422 | 322 | 49 |

### 11.3 Tail-Aware w2 Plus Contact Geometry

| Subset | Method | MAE | RMSE | Spearman | `pred_std / true_std` | `error_vs_true_Pearson` | Tail MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| HCDR3+LCDR3 safe | Sequence prediction | 0.9740 | 1.2857 | 0.3796 | 0.5021 | -0.8692 | 1.8701 |
| HCDR3+LCDR3 safe | + Contact Ridge correction | 0.9658 | 1.2745 | 0.3984 | 0.4969 | -0.8711 | 1.8515 |
| All-CDR safe | Sequence prediction | 0.9278 | 1.2105 | 0.2883 | 0.5837 | -0.8351 | 1.5660 |
| All-CDR safe | + Contact Ridge correction | 0.9016 | 1.1815 | 0.3423 | 0.5699 | -0.8388 | 1.5542 |

Interpretation:

- On tail-aware w2 predictions, contact correction gives small improvements in
  MAE/RMSE/Spearman/tail MAE in both contact-safe subsets.
- However, `pred_std / true_std` moves slightly farther from `1`, and
  `error_vs_true_Pearson` becomes slightly more negative.
- Thus real CDR3 interface geometry appears to contain incremental information,
  but simple contact summaries plus linear residual correction do **not**
  resolve regression-to-the-mean.
- On the unweighted baseline, effects are inconsistent across the two subsets,
  reinforcing that this is exploratory evidence rather than a robust new
  benchmark winner.

## 12. Limitations

1. **Benchmark size and tail scarcity.** The formal ANDD benchmark contains
   1,168 rows, but genuinely extreme affinity rows are still limited.
2. **Assay and label heterogeneity.** Experimental absolute Kd values may
   reflect assay conditions, provenance, and measurement noise that are not
   modeled explicitly.
3. **Representation limitations.** Sequence-only CDR and antigen inputs do not
   fully encode three-dimensional interface chemistry or residue contacts.
4. **Structure subset limitation.** Conservative contact extraction is
   currently available for only 472 unambiguous-chain rows, and CDR-safe
   analyses use still smaller subsets.
5. **Calibration/objective tradeoffs.** Calibration and tail-aware loss
   improve some compression diagnostics while sacrificing other metrics.
6. **Multi-seed scope.** Formal multi-seed validation exists for unweighted
   versus tail-aware w2 cross-attention, but not yet for every architecture or
   contact augmentation.
7. **No external held-out benchmark.** Results are internally held out by
   antigen sequence, but have not been validated on an independent external
   data source.
8. **Framing boundary.** The results motivate examining ranking-based
   formulations; they do not prove an inherent defect in absolute Kd
   regression.

## 13. Final Conclusion

The ANDD antibody v2 phase successfully produced a conservative, leakage-aware,
CDR-standardized affinity regression benchmark and clarified the key modeling
limitation.

The current model family is not primarily failing because of an output-layer
bug, target inversion mistake, prediction clamp, or trivial antigen leakage.
Instead, it systematically compresses predictions toward the mean, including
on training data. Cross-attention and tail-aware training reduce parts of this
behavior, and contact geometry adds a small amount of incremental signal in a
strict structure-covered subset. None of these steps yet eliminates the core
compression pattern.

The most defensible current conclusion is:

> The remaining bottleneck is likely a combination of insufficient
> sequence/interface representation, sparse and noisy affinity tails, assay
> heterogeneity, and the challenge of predicting absolute Kd at the extremes.

This outcome is scientifically useful: it identifies what did not fix the
problem, establishes auditable structure-contact feasibility, and points to a
focused next stage rather than uncontrolled model complexity.

## 14. Recommended Next Directions

1. **Richer structure/contact-aware features.** Move beyond simple contact
   counts toward validated per-CDR/per-residue interface features, chemical
   contact types, distance summaries, or learnable structure-conditioned
   fusion, while retaining strict mapping filters.
2. **Ranking-based formulation as future work.** Test ranking or paired
   objectives as an additional task framing, especially for strong/weak binder
   ordering and tail retrieval. This should be framed as a motivated
   experiment, not as proof that absolute Kd regression cannot work.
3. **Larger and cleaner affinity data.** Continue strict expansion with assay
   type, label provenance, unit standardization, and experimental-versus-
   predicted label separation.
4. **Larger protein backbone ablation only after benchmark finalization.**
   Backbone scaling should come after the split, target policy, evaluation
   metrics, and structure-coverage rules are frozen, so added capacity is
   interpretable rather than confounded.

## 15. Key Referenced Outputs and Scripts

### Data and Split Reports

- ANDD source audit:
  `outputs/data_expansion/ANDD_audit/ANDD_data_source_audit.md`
- Conservative antibody candidate QC:
  `outputs/data_expansion/ANDD_antibody_v2_audit/antibody_v2_quality_audit_report.md`
- Original antibody v2 split:
  `data/processed_affinity/expanded_affinity_antibody_v2/split_summary.md`
- Stratified antigen-level split:
  `data/processed_affinity/expanded_affinity_antibody_v2_stratified/split_summary.md`
- Standard CDR extraction:
  `data/processed_affinity/expanded_affinity_antibody_v2_cdr_annotated/cdr_extraction_summary.md`

### Modeling and Diagnosis Reports

- Original ANDD pooled baseline error analysis:
  `outputs/andd_antibody_v2/error_analysis/error_analysis_report.md`
- Target distribution diagnosis:
  `outputs/andd_antibody_v2/target_distribution/target_distribution_report.md`
- Linear calibration:
  `outputs/andd_antibody_v2/calibration/calibration_report.md`
- Stratified pooled baseline:
  `outputs/andd_antibody_v2_stratified/all_cdr_pooled/andd_antibody_v2_stratified_all_cdr_pooled_report.md`
- Stratified cross-attention:
  `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs/cross_attention_report.md`
- Train/validation/test fit diagnosis:
  `outputs/andd_antibody_v2_stratified/fit_diagnosis/fit_diagnosis_report.md`
- Tail-aware w3:
  `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_tailaware/tailaware_training_report.md`
- Tail-aware w2:
  `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_tailaware_w2/tailaware_w2_training_report.md`
- Multi-seed validation:
  `outputs/andd_antibody_v2_stratified/multiseed/multiseed_summary.md`

### Contact/Structure Reports

- Contact feature availability audit:
  `outputs/andd_antibody_v2_stratified/contact_feature_audit/contact_feature_audit_report.md`
- Basic interface geometry pilot:
  `outputs/andd_antibody_v2_stratified/contact_feature_audit/basic_interface_feature_report.md`
- CDR-to-structure mapping validation:
  `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr_mapping_validation_report.md`
- CDR3 contact-augmented residual baseline:
  `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr3_contact_augmented_baseline_report.md`

### Machine-Readable Metric Tables

- Calibration metrics:
  `outputs/andd_antibody_v2/calibration/calibration_metrics.csv`
- Stratified fit metrics by model and split:
  `outputs/andd_antibody_v2_stratified/fit_diagnosis/fit_metrics_by_split.csv`
- Tail-aware multi-seed metrics:
  `outputs/andd_antibody_v2_stratified/multiseed/multiseed_summary.csv`
- Basic interface geometry features:
  `outputs/andd_antibody_v2_stratified/contact_feature_audit/basic_interface_features.csv`
- Preliminary CDR contact features:
  `outputs/andd_antibody_v2_stratified/contact_feature_audit/preliminary_cdr_contact_features.csv`
- CDR3 contact-augmented comparison metrics:
  `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr3_contact_augmented_metrics.csv`

### Figures

- Original ANDD target distribution:
  `outputs/andd_antibody_v2/target_distribution/target_distribution_histogram.png`
- Calibration true-versus-predicted:
  `outputs/andd_antibody_v2/calibration/true_vs_predicted_raw_vs_calibrated.png`
- Tail-aware w2 comparison:
  `outputs/final_reports/figures/tailaware_w2_checkpoint_comparison.png`
- Multi-seed comparison:
  `outputs/final_reports/figures/multiseed_w2_vs_baseline.png`
- Basic interface feature correlations:
  `outputs/final_reports/figures/basic_interface_feature_correlations.png`
- CDR3 contact augmented baseline:
  `outputs/final_reports/figures/cdr3_contact_augmented_baseline.png`

### Main Scripts

- `scripts/audit_andd_data_source.py`
- `scripts/audit_andd_antibody_v2_candidates.py`
- `scripts/build_andd_antibody_v2_stratified_split.py`
- `scripts/analyze_andd_antibody_v2_target_distribution.py`
- `scripts/calibrate_andd_antibody_v2_predictions.py`
- `scripts/analyze_andd_stratified_model_fit.py`
- `scripts/summarize_andd_stratified_cross_attention_multiseed.py`
- `scripts/audit_andd_stratified_contact_feature_availability.py`
- `scripts/extract_andd_stratified_basic_interface_features.py`
- `scripts/validate_andd_stratified_cdr_structure_mapping.py`
- `scripts/analyze_andd_stratified_cdr3_contact_augmented_baseline.py`

## 16. Three-Minute Presentation Summary

This phase focused on building a careful ANDD antibody-only affinity regression
benchmark and understanding why our sequence-based models still struggle at
affinity extremes. Starting from 48,800 ANDD rows, I separated antibody from
nanobody data, excluded predicted labels from the primary benchmark, audited
label and sequence quality, and obtained 1,168 conservative antibody rows. I
then applied standard AbNumber and IMGT CDR extraction with complete success,
and built antigen-group splits with zero antigen overlap across train,
validation, and test.

The first all-CDR pooled ESM2 plus LoRA baseline was trainable and achieved test
MAE 0.9066 on the original ANDD split, but the important finding was not just
the MAE. Its prediction spread was only 37.7 percent of the true spread, and
its residuals were strongly negatively correlated with target affinity. In
plain terms, weak binders were predicted too strong and strong binders were
predicted too weak. I checked whether this was caused by the regression output
layer, an activation clamp, a target transform error, or antigen leakage. The
head is an unrestricted scalar regressor, the target pipeline is consistent,
and the antigen-group split has zero overlap, so those simple explanations do
not account for the issue.

I next investigated whether the split itself made the problem look worse. I
created a new stratified antigen-level split that explicitly places both low and
high target tails in validation and test while still keeping antigen overlap at
zero. On this same split, pooled all-CDR gave the better MAE at 0.9373, while
all-CDR cross-attention improved Spearman, prediction spread, residual bias, and
upper-tail error. This suggested that learnable CDR-antigen interaction helps
ranking and strong-binding behavior, although it does not yet optimize overall
absolute error.

Because the main symptom was tail compression, I then tried tail-aware
cross-attention loss. In a single seed, a moderate weight of 2.0 looked very
promising: it improved MAE, Spearman, prediction spread, and tail error relative
to the historical unweighted model. But I did not stop there. In a controlled
three-seed validation, tail-aware w2 consistently improved prediction spread,
reduced residual bias, and modestly reduced combined tail MAE, particularly on
the lower tail. However, the unweighted model remained better on average MAE,
RMSE, and Spearman. Therefore the honest conclusion is that tail-aware loss
helps the symptom but is not a stable replacement for the baseline.

Finally, I audited physical interface information. All 1,168 benchmark samples
match files in the external SAbDab structure archive, but only 472 have
unambiguous chain mappings suitable for conservative interface extraction.
Basic contact features were successfully extracted for all of those rows. After
strict residue-level validation, 467 rows were safe for HCDR3 plus LCDR3
contact features and 422 were safe for all-CDR contact features. Adding these
CDR3 geometry features as a small Ridge residual correction to tail-aware
predictions gave modest improvements in MAE, Spearman, and tail MAE inside the
contact-covered subsets, but it did not improve prediction spread or remove
regression-to-the-mean.

So the final conclusion is that the current bottleneck is not a simple coding
bug and not only a split problem. It more likely reflects limited biological
representation, sparse and heterogeneous affinity tails, and the difficulty of
calibrating absolute Kd predictions. The next serious direction is richer,
validated structure/contact-aware modeling, with ranking-based objectives as a
carefully evaluated future option, and any larger backbone experiment deferred
until the benchmark design is fully frozen.
