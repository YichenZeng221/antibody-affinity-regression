# Data Summary

## ANDD Antibody v2 Candidate Construction

The project constructed a conservative antibody-only benchmark from ANDD. Nanobody samples and predicted affinity labels were treated separately from the main antibody-only regression task.

The final conservative antibody-only split used experimental Kd-like labels, available antibody heavy/light sequences, antigen sequences, and leakage-aware antigen-level splitting.

## Stratified Split

The final ANDD antibody v2 stratified split contains:

| Split | Rows |
|---|---:|
| Train | 934 |
| Validation | 117 |
| Test | 117 |

The split uses antigen sequence as the grouping unit to avoid antigen leakage across train, validation, and test.

## Included Data Reports

- `split_summary.md`
- `split_summary.json`
- `expanded_affinity_v2_candidate_summary.md`
- `antibody_v2_quality_audit_report.md`

Large raw data files are intentionally not included in this GitHub export.

