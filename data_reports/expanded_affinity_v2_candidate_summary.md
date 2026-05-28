# ANDD expanded_affinity_dataset_v2 Candidate Summary

## Scope

This is a conservative candidate dataset, not a final train/val/test dataset.

- No model was trained.
- No final split was created.
- Current `unified_no_high_risk` was not modified.
- Antibody and nanobody candidates are intentionally separated.

## 1. Candidate Counts

- Tier 1 rows from ANDD audit: `4382`
- Tier 1 rows after predicted/ANTIPASTI filter: `4382`
- Antibody candidate rows: `3116`
- Nanobody candidate rows: `1102`
- Excluded or unsupported Tier 1 rows: `164`

## 2. Overlap And Duplicate Checks

- Antibody exact heavy+light+antigen triplet overlaps with unified_no_high_risk: `15`
- Antibody antigen_sequence overlap rows: `612`
- Nanobody antigen_sequence overlap rows: `31`
- Antibody exact triplet duplicates within ANDD candidates: `423`
- Nanobody+antigen duplicates within ANDD candidates: `534`
- Total rows written to overlap file: `654`

Duplicate antigen_sequence rows are flagged but not removed, because future antigen-group split design needs this information.

## 3. Kd And Target Distribution

| Candidate group | Kd(M) summary | neg_log10 target summary |
|---|---|---|
| antibody | count=3116, min=8.023e-16, median=1.277e-05, mean=2.418e+21, max=5.56e+24, std=1.044e+23 | count=3116, min=-24.75, median=4.894, mean=3.041, max=15.1, std=5.756 |
| nanobody | count=1102, min=7.192e-16, median=5.428e-09, mean=5.444e+15, max=3e+18, std=1.277e+17 | count=1102, min=-18.48, median=8.265, mean=8.23, max=15.14, std=2.278 |

Kd is retained in M as `affinity_kd_m`, and `neg_log10_affinity_candidate` is computed as `-log10(Kd)`.
Rows with very tiny or very large Kd are not removed here; they are flagged in `risk_flags` for human review.

## 4. Top Sources

### Antibody candidates

| Source | Rows |
|---|---:|
| `SabDab_ab` | 3108 |
| `PDB-bind_ab` | 8 |

### Nanobody candidates

| Source | Rows |
|---|---:|
| `SabDab_nano` | 854 |
| `SabDab_ab` | 216 |
| `sdAB-DB` | 32 |

## 5. Risk Flags

### Antibody risk flags

| Flag | Rows |
|---|---:|
| `very_weak_or_large_kd` | 850 |
| `kd_greater_than_1e-2_M` | 752 |
| `overlap_antigen_sequence` | 612 |
| `overlap_source_id` | 450 |
| `duplicate_exact_triplet_within_ANDD` | 423 |
| `overlap_exact_triplet` | 15 |
| `kd_less_than_1e-12_M` | 2 |
| `very_strong_or_tiny_kd` | 2 |

### Nanobody risk flags

| Flag | Rows |
|---|---:|
| `duplicate_nanobody_antigen_within_ANDD` | 534 |
| `kd_less_than_1e-12_M` | 53 |
| `very_strong_or_tiny_kd` | 53 |
| `overlap_antigen_sequence` | 31 |
| `very_weak_or_large_kd` | 16 |
| `kd_greater_than_1e-2_M` | 10 |

## 6. Recommendation

Do not mix antibody and nanobody into one main task yet.

Recommended next step:

1. Build `expanded_affinity_antibody_v2` first if the next model is heavy+light+antigen.
2. Build `expanded_affinity_nanobody_v2` separately for VHH/nanobody input mode.
3. Keep predicted labels out of the primary supervised benchmark.
4. Review extreme Kd flags and source/provenance before final split.
5. Only after manual review, create antigen-sequence group split for each task.

## 7. Output Files

- `outputs/data_expansion/ANDD_v2_candidates/expanded_affinity_antibody_v2_candidates.csv`
- `outputs/data_expansion/ANDD_v2_candidates/expanded_affinity_nanobody_v2_candidates.csv`
- `outputs/data_expansion/ANDD_v2_candidates/excluded_or_flagged_rows.csv`
- `outputs/data_expansion/ANDD_v2_candidates/overlap_with_unified_no_high_risk.csv`
- `data/processed_affinity/expanded_affinity_dataset_v2_candidates/expanded_affinity_antibody_v2_candidates.csv`
- `data/processed_affinity/expanded_affinity_dataset_v2_candidates/expanded_affinity_nanobody_v2_candidates.csv`