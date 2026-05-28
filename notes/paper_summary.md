# Paper / Project Reading Summary

## SeqProFT-Style Motivation

This project was inspired by the idea of parameter-efficient protein model fine-tuning, especially the SeqProFT-style workflow: start from a pretrained protein language model, adapt it with a small number of trainable parameters, and evaluate whether task-specific biological signals can be recovered without full model retraining.

In this project, the concrete use case was antibody-antigen binding affinity prediction. I used ESM2 with LoRA-style fine-tuning and compared several input/modeling choices:

- whole antibody chain + antigen sequence
- standard IMGT CDRs + antigen sequence
- all-CDR pooled representation
- all-CDR cross-attention
- tail-aware loss
- CDR3 contact feature augmentation

## Main Task Framing

The initial task was absolute affinity regression:

```text
input: antibody sequence representation + antigen sequence
target: -log10(Kd)
```

Higher `-log10(Kd)` means stronger binding.

This is a valid benchmark, but the experiments showed that calibrated absolute affinity prediction is difficult under noisy labels, heterogeneous assays, limited data, and imperfect sequence/structure representations.

## Key Empirical Observation

The main failure mode was regression-to-the-mean:

- low-affinity examples were often predicted too high
- high-affinity examples were often predicted too low
- prediction standard deviation was much smaller than true target standard deviation
- the same compression appeared on train, validation, and test splits

This suggests the issue is not simply classic overfitting. It is more likely a combination of representation bottleneck, label noise, target imbalance/tail scarcity, and the difficulty of absolute Kd regression.

## CDR-Aware Modeling Lesson

Standard CDR extraction with AbNumber / IMGT was important. Fixed-index slicing is not reliable for antibody CDR annotation.

CDR-aware inputs improved the representation compared with whole-chain pooled inputs. In particular:

- all six CDRs provided the best overall pooled baseline
- HCDR3 + LCDR3 captured much of the signal
- cross-attention improved ranking and prediction spread in some settings

This supports the biological intuition that binding signal is concentrated around CDR-antigen interaction regions.

## Tail-Aware Loss Lesson

Tail-aware loss helped reduce some symptoms:

- improved prediction spread
- improved P10/P90 tail MAE in some settings
- reduced residual bias in single-seed runs

However, multi-seed validation showed that it did not consistently improve overall MAE or Spearman. This was a useful reminder that single-seed improvements can be misleading.

## Contact / Interface Feature Lesson

The contact/interface audit showed that structure information is technically usable for a conservative subset:

- all 1,168 ANDD antibody v2 stratified rows had external structure file matches
- 472 rows had unambiguous H/L/antigen chain mapping
- basic interface features were extracted successfully for all 472 rows
- HCDR3/LCDR3 contact-safe mapping covered 467 rows
- all-CDR contact-safe mapping covered 422 rows

Simple scalar contact features gave small subset gains through Ridge residual correction, but they did not solve prediction compression. This suggests that richer structure/contact-aware representations may be more useful than raw contact counts.

## AbRank Reflection

AbRank helped me think more carefully about task framing. If the practical goal is antibody screening, the key question may not be:

```text
What is the exact calibrated Kd?
```

but rather:

```text
Which antibody is more likely to bind better?
```

My results do not prove that absolute Kd regression is invalid. Instead, they are consistent with the motivation behind ranking-based formulations: under noisy and heterogeneous affinity labels, ranking or candidate prioritization may be more practical than exact absolute Kd prediction.

## What I Learned

- Biomedical ML needs careful definition of input, target, split unit, leakage risk, and metrics.
- Error analysis can be more informative than immediately changing model architecture.
- Multi-seed validation is essential before trusting a result.
- CDR extraction must use standard numbering tools rather than fixed slicing.
- Contact/interface modeling requires chain and residue mapping validation before feature extraction.
- Task framing matters: absolute regression, ranking, calibration, and screening prioritization are related but different problems.

## Recommended Next Direction

The natural next project is an AbRank-inspired antibody binder ranking benchmark:

- convert affinity labels into pairwise or listwise preferences
- compare regression-derived ranking vs direct ranking models
- evaluate pairwise accuracy, Spearman, NDCG, and top-k enrichment
- test whether CDR/contact-aware representations improve candidate prioritization
