# ANDD Stratified CDR-to-Structure Mapping Validation

## Scope and Safety Rules

- 输入样本仅限 basic interface pilot 中 chain mapping 已无歧义的 472 rows。
- 本脚本不训练模型，不修改 dataset，不处理 695 条 ambiguous chain mappings。
- CDR annotations 来源于标准 AbNumber + IMGT extraction；接触坐标来源于只读 SAbDab raw PDB。
- CDR contact 只在 full heavy/light sequence 到结构 chain 的 strict mapping 通过后作为 preliminary pilot 输出。

## Mapping Validation Rule

1. 每个 CDR sequence 必须在对应 full chain sequence 中唯一出现。
2. Full chain sequence 对齐到结构 chain 时，alignment identity 必须至少 95%，coverage 至少 80%。
3. 每个 CDR residue 必须映射到结构中存在、且 amino acid 字符一致的 coordinate-bearing residue。
4. 如果多个同分 alignment 对某个 CDR 给出不同 mapping，则记录为 insertion/deletion ambiguity，不计算 contact。

## Overall Result

- Pilot rows validated: **472**.
- Rows safe for preliminary CDR-antigen contact features: **422 / 472** (89.41%).
- Failed mapping rows: **50 / 472**.
- Eligible rows by split: `{'test': 49, 'train': 322, 'val': 51}`.
- HCDR3-only contact-safe rows: **468 / 472**.
- LCDR3-only contact-safe rows: **470 / 472**.
- HCDR3+LCDR3 jointly contact-safe rows: **467 / 472**.

## CDR Mapping Success Rate

| CDR | success_rows | total_rows | success_rate |
|---|---|---|---|
| LCDR3 | 470 | 472 | 99.58% |
| HCDR2 | 469 | 472 | 99.36% |
| HCDR3 | 468 | 472 | 99.15% |
| HCDR1 | 466 | 472 | 98.73% |
| LCDR1 | 465 | 472 | 98.52% |
| LCDR2 | 435 | 472 | 92.16% |

## Failure Reasons

| reason | rows |
|---|---|
| chain_sequence_mismatch | 2 |
| missing_residue_numbering | 0 |
| insertion_deletion_ambiguity | 34 |
| unresolved_residues | 16 |
| antigen_chain_mismatch | 0 |
| missing_CDR_annotation | 0 |

### Failure Reason Co-occurrence Patterns

| reason | rows |
|---|---|
| insertion_deletion_ambiguity | 32 |
| unresolved_residues | 14 |
| insertion_deletion_ambiguity;unresolved_residues | 2 |
| chain_sequence_mismatch | 2 |

## Preliminary CDR Contact Features

- `all_cdr_contact_count_5A`: all six mapped CDRs 与 antigen 的 5 A residue-pair contact 数。
- `hcdr3_contact_count_5A`, `lcdr3_contact_count_5A`: CDR3 loops 的 5 A residue-pair contact 数。
- `hcdr3_contact_fraction_5A`, `lcdr3_contact_fraction_5A`: 至少接触 antigen 的 CDR3 residues 占比。
- `cdr_interface_residue_count_5A`: 接触 antigen 的 CDR residues 数。
- `cdr_min_distance`: all mapped CDR residues 到 antigen 的最小非氢原子距离。
- All-CDR aggregate features 只在六段 CDR 均通过时有值；HCDR3/LCDR3 features 则在相应 loop 自身安全映射时保留，因此可评估 CDR3-only 路线。
- Preliminary contact table rows: **471**（至少一个 CDR3 loop 可安全计算）；其中 all-six features **422** rows，HCDR3 features **468** rows，LCDR3 features **470** rows。

### Exploratory Correlations

| feature | outcome | n | pearson | spearman |
|---|---|---|---|---|
| all_cdr_contact_count_5A | target_affinity | 422 | 0.157 | 0.148 |
| all_cdr_contact_count_5A | unweighted_cross_attention_absolute_error | 49 | -0.287 | -0.204 |
| all_cdr_contact_count_5A | tailaware_w2_best_val_tail_mae_absolute_error | 49 | -0.195 | -0.128 |
| hcdr3_contact_count_5A | target_affinity | 468 | 0.110 | 0.059 |
| hcdr3_contact_count_5A | unweighted_cross_attention_absolute_error | 58 | -0.181 | -0.069 |
| hcdr3_contact_count_5A | tailaware_w2_best_val_tail_mae_absolute_error | 58 | -0.128 | 0.036 |
| lcdr3_contact_count_5A | target_affinity | 470 | 0.059 | 0.070 |
| lcdr3_contact_count_5A | unweighted_cross_attention_absolute_error | 58 | 0.090 | 0.088 |
| lcdr3_contact_count_5A | tailaware_w2_best_val_tail_mae_absolute_error | 58 | 0.111 | 0.074 |
| cdr_min_distance | target_affinity | 422 | -0.004 | -0.028 |
| cdr_min_distance | unweighted_cross_attention_absolute_error | 49 | -0.046 | -0.080 |
| cdr_min_distance | tailaware_w2_best_val_tail_mae_absolute_error | 49 | 0.023 | -0.077 |

### Tail Pattern

- Train-defined tail thresholds: P10 = **3.3333**, P90 = **6.3204**.
| target_tail | n | all_cdr_contact_count_5A_mean | hcdr3_contact_count_5A_mean | lcdr3_contact_count_5A_mean | cdr_min_distance_mean |
|---|---|---|---|---|---|
| above_train_P90 | 49 | 51.673 | 19.490 | 6.388 | 2.524 |
| below_train_P10 | 41 | 43.512 | 16.171 | 5.854 | 2.567 |
| middle_P10_to_P90 | 332 | 54.633 | 20.663 | 7.660 | 2.507 |

## Answers

1. Reliable CDR-contact rows: **422 / 472** pilot samples.
2. Most stable CDR mapping: **LCDR3** based on strict mapping success rate.
3. HCDR3/LCDR3 availability: HCDR3 **468 / 472**, LCDR3 **470 / 472**.
4. Main failure reasons: `[{'reason': 'chain_sequence_mismatch', 'rows': 2}, {'reason': 'missing_residue_numbering', 'rows': 0}, {'reason': 'insertion_deletion_ambiguity', 'rows': 34}, {'reason': 'unresolved_residues', 'rows': 16}, {'reason': 'antigen_chain_mismatch', 'rows': 0}, {'reason': 'missing_CDR_annotation', 'rows': 0}]`.
5. Modeling recommendation: CDR-contact-aware modeling is technically feasible. A conservative next experiment can prioritize HCDR3/LCDR3 contact features because their mapping coverage is higher than all-six-CDR coverage. Preliminary weak correlations mean this should remain an incremental, controlled baseline rather than a claimed solution.
6. If coverage is inadequate, the missing ingredient is reliable residue-level mapping for affected structures, including unresolved residues, indels, and chain assignment confirmation.

## Outputs

- Mapping availability: `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr_mapping_availability.csv`
- Preliminary CDR contacts: `outputs/andd_antibody_v2_stratified/contact_feature_audit/preliminary_cdr_contact_features.csv`
- This report: `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr_mapping_validation_report.md`
