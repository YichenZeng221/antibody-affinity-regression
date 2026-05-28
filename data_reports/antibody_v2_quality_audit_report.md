# ANDD Antibody-only v2 Candidate Quality Audit

## Scope

This audit prepares antibody-only `expanded_affinity_dataset_v2` candidates for manual review.

- No model was trained.
- No final train/val/test split was created.
- Existing `unified_no_high_risk` data was not modified.
- Original candidate rows were not deleted; quality issues are stored as flags.

## 1. Overall Counts

- Total antibody candidate rows: `3116`
- Conservative `keep_safe` rows: `1168`
- Rows with extreme Kd flags: `754`
- Rows with sequence issue flags: `738`
- Rows with duplicate / exact-triplet overlap flags: `688`
- Rows with antigen overlap flags: `612`
- Rows overlapping current unified test antigens: `62`

## 2. Kd Value Audit

- `affinity_kd_m`: count=3116, min=8.023e-16, median=1.277e-05, mean=2.418e+21, max=5.56e+24, std=1.044e+23
- `neg_log10_affinity_candidate`: count=3116, min=-24.75, median=4.894, mean=3.041, max=15.1, std=5.756

Extreme Kd rows are not removed here. They are flagged for manual review because very large or very tiny Kd values can dominate regression loss and may reflect unit/provenance problems.

## 3. Source / Provenance Audit

| Source | Rows | Extreme Kd | Sequence Issue | Duplicate | Antigen Overlap | Keep Safe |
|---|---:|---:|---:|---:|---:|---:|
| `SabDab_ab` | 3108 | 754 | 730 | 685 | 610 | 1168 |
| `PDB-bind_ab` | 8 | 0 | 8 | 3 | 2 | 0 |

## 4. Sequence Quality Audit

- Heavy length: min=97.0, median=121.0, mean=122.61, max=231.0
- Light length: min=93.0, median=108.0, mean=109.11, max=220.0
- Antigen length: min=5.0, median=231.0, mean=381.76, max=2040.0
- Nonstandard heavy AA rows: `1`
- Nonstandard light AA rows: `0`
- Nonstandard antigen AA rows: `64`
- Heavy/light identical rows: `0`

## 5. CDR Field Audit

- CDR columns present: `True`
- Important: ANDD CDR fields should still be treated as source-provided annotations. For a formal benchmark, CDRs should be regenerated with one consistent method, preferably AbNumber + IMGT, before CDR-aware modeling.

- `HCDR1` length: min=0, median=7.0, mean=7.18, max=14, unique_lengths=[0, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]
- `HCDR2` length: min=0, median=6.0, mean=5.83, max=20, unique_lengths=[0, 4, 5, 6, 7, 8, 10, 11, 13, 15, 16, 20]
- `HCDR3` length: min=0, median=12.0, mean=12.84, max=63, unique_lengths=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- `LCDR1` length: min=0, median=11.0, mean=12.24, max=19, unique_lengths=[0, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]
- `LCDR2` length: min=0, median=7.0, mean=7.01, max=12, unique_lengths=[0, 3, 5, 7, 11, 12]
- `LCDR3` length: min=0, median=9.0, mean=9.31, max=19, unique_lengths=[0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19]

## 6. Duplicate / Leakage Audit

- Exact triplet duplicates within ANDD candidates: `675`
- Exact triplet overlap with current unified_no_high_risk: `15`
- Antigen sequence overlap with current unified_no_high_risk: `612`
- Antigen sequence overlap with current unified_no_high_risk test split: `62`
- Source ID overlap with current unified_no_high_risk: `450`

Antigen overlap rows are flagged but not deleted, because final antigen-group split design needs this information.

## 7. Conservative Filtering Proposal

`keep_safe=True` means the row currently has no extreme Kd, sequence issue, exact duplicate/triplet overlap, or antigen overlap flag.

Suggested exclusions before building a formal antibody-only v2 training dataset:

1. Exclude invalid/nonpositive Kd rows if any appear.
2. Exclude or manually inspect extreme Kd rows (`Kd < 1e-12 M` or `Kd > 1e-2 M`).
3. Exclude sequence issue rows with nonstandard amino acids, abnormal lengths, or identical heavy/light chains.
4. Remove exact heavy+light+antigen duplicates and exact overlaps with current unified data.
5. Keep antigen overlap information for split design; do not mix overlapping antigens across train/val/test.
6. Re-run standard CDR extraction with AbNumber + IMGT before CDR-aware v2 experiments.

## 8. Recommendation

Yes, it is worth building a conservative antibody-only v2 next. Start from the `1168` `keep_safe` rows, then do manual review of high-value flagged rows if more data is needed.

Do not mix antibody and nanobody in this antibody-only v2. Nanobody should be a separate task because the input structure is different.

## 9. Output Files

- `/Users/yichenzeng/seqproft_reproduce/outputs/data_expansion/ANDD_antibody_v2_audit/expanded_affinity_antibody_v2_audited_flags.csv`
- `/Users/yichenzeng/seqproft_reproduce/outputs/data_expansion/ANDD_antibody_v2_audit/antibody_v2_flag_summary.csv`
- `/Users/yichenzeng/seqproft_reproduce/outputs/data_expansion/ANDD_antibody_v2_audit/antibody_v2_source_summary.csv`
- `/Users/yichenzeng/seqproft_reproduce/outputs/data_expansion/ANDD_antibody_v2_audit/antibody_v2_kd_distribution_summary.csv`
- `/Users/yichenzeng/seqproft_reproduce/outputs/data_expansion/ANDD_antibody_v2_audit/antibody_v2_quality_audit_report.md`
