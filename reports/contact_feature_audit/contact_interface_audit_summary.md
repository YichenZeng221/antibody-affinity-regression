# Contact / Interface Feature Audit Summary

## Motivation

After sequence-only and tail-aware models still showed regression-to-the-mean, the next question was whether real antibody-antigen interface geometry could explain the remaining errors.

## Contact Availability Audit

| Step | Rows |
|---|---:|
| Total ANDD antibody v2 stratified rows | 1,168 |
| External structure file match | 1,168 |
| Complete H/L/antigen chain metadata option | 1,167 |
| Unambiguous viable H/L/antigen chain mapping | 472 |
| Basic interface feature-ready rows | 472 |

Ambiguous chain mappings were not used for contact extraction.

## Basic Interface Geometry Extraction

Basic geometry features were extracted for all 472 pilot-eligible rows.

Extracted features included:

- minimum antibody-antigen distance
- contact counts at 4 A, 5 A, and 8 A
- antibody and antigen interface residue counts
- heavy-chain and light-chain interface residue counts

The single-feature relationship with affinity was weak:

| Feature | Pearson vs target | Spearman vs target |
|---|---:|---:|
| contact_count_5A | 0.198 | 0.173 |
| min_ab_ag_distance | -0.023 | -0.068 |

## CDR-to-Structure Mapping Validation

CDR residue mapping was validated on the 472-row unambiguous pilot subset.

| Mapping subset | Rows |
|---|---:|
| HCDR3-only contact-safe | 468 |
| LCDR3-only contact-safe | 470 |
| HCDR3+LCDR3 jointly contact-safe | 467 |
| All-six-CDR contact-safe | 422 |

Main failure reasons were insertion/deletion ambiguity, unresolved residues, and chain sequence mismatch.

## CDR3 Contact Residual-Correction Baseline

A small post-hoc Ridge residual-correction experiment tested whether CDR3 contact features add value beyond sequence model predictions.

For the tail-aware w2 sequence baseline:

| Subset | Method | MAE | Spearman | Tail MAE | pred std / true std |
|---|---|---:|---:|---:|---:|
| HCDR3+LCDR3 safe | Sequence only | 0.9740 | 0.3796 | 1.8701 | 0.5021 |
| HCDR3+LCDR3 safe | Sequence + CDR3 contact | 0.9658 | 0.3984 | 1.8515 | 0.4969 |
| All-CDR safe | Sequence only | 0.9278 | 0.2883 | 1.5660 | 0.5837 |
| All-CDR safe | Sequence + CDR3 contact | 0.9016 | 0.3423 | 1.5542 | 0.5699 |

## Interpretation

Contact/interface features are technically feasible and contain weak incremental signal. However, simple scalar contact counts and linear residual correction do not solve prediction compression. The next step should use richer structure/contact-aware representations rather than only scalar contact-count features.

## Included Summary Files

- `contact_feature_availability.csv`
- `basic_interface_features.csv`
- `cdr_mapping_availability.csv`
- `cdr3_contact_augmented_metrics.csv`

