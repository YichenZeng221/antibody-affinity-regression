# Reproduction Plan

This repository is a curated portfolio export. Large raw datasets, structure archives, checkpoints, and local prediction dumps are intentionally excluded. The goal of this plan is to document how the main experiments can be reproduced when the required local data are available.

## 1. Environment Setup

Create a Python environment for model training:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For standard CDR extraction, use a separate environment with AbNumber, ANARCI, and HMMER available. In the original project this was kept separate from the model-training environment.

## 2. Required Local Inputs

The following files are not included in this GitHub export:

- ANDD v2 spreadsheet.
- SAbDab all-structures archive.
- Processed train / validation / test CSV files.
- Model checkpoints.
- Large prediction files.

The final reports and summary CSVs included in this repository document the resulting benchmark and model behavior.

## 3. Dataset Preparation

The final benchmark used in the main stage was the ANDD antibody v2 stratified antigen-level split. The relevant scripts are:

```text
scripts/audit_andd_data_source.py
scripts/build_andd_v2_candidates.py
scripts/audit_andd_antibody_v2_candidates.py
scripts/build_andd_antibody_v2_stratified_split.py
scripts/extract_standard_cdrs_andd_antibody_v2.py
```

The split was designed so that train, validation, and test antigen sequences do not overlap. The stratified version was used to make validation and test cover target-distribution tails.

## 4. Main Model Runs

The main final-stage models are represented by these configs:

```text
configs/config_affinity_andd_antibody_v2_stratified_all_cdr_pooled_lr3e-5_e10.yaml
configs/config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_lr3e-5_e10.yaml
configs/config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_tailaware_w2_lr3e-5_e20.yaml
configs/config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_unweighted_s42_lr3e-5_e20.yaml
```

Example training commands:

```bash
./.venv/bin/python run_train_affinity_cdr.py \
  --config configs/config_affinity_andd_antibody_v2_stratified_all_cdr_pooled_lr3e-5_e10.yaml

./.venv/bin/python run_train_affinity_cross_attention.py \
  --config configs/config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_lr3e-5_e10.yaml

./.venv/bin/python run_train_affinity_cross_attention_tailaware.py \
  --config configs/config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_tailaware_w2_lr3e-5_e20.yaml
```

## 5. Evaluation and Diagnostics

The key diagnostic scripts are:

```text
scripts/evaluate_andd_antibody_v2_stratified_cdr_test_set.py
scripts/evaluate_andd_antibody_v2_stratified_cross_attention_test_set.py
scripts/evaluate_andd_antibody_v2_stratified_tailaware_checkpoints.py
scripts/analyze_andd_stratified_model_fit.py
scripts/summarize_andd_stratified_cross_attention_multiseed.py
```

The final presentation figures were generated from existing metrics and prediction files with:

```text
scripts/make_andd_final_presentation_clean_figures.py
scripts/make_contact_interface_audit_figures.py
```

## 6. Contact / Interface Analysis

The structure-aware audit was intentionally kept as an analysis stage rather than a full deep structure model. The relevant scripts are:

```text
scripts/audit_andd_stratified_contact_feature_availability.py
scripts/extract_andd_stratified_basic_interface_features.py
scripts/validate_andd_stratified_cdr_structure_mapping.py
scripts/analyze_andd_stratified_cdr3_contact_augmented_baseline.py
```

The final report distinguishes full-benchmark results from contact-covered subset analysis.

## 7. Reading Order

Recommended reading order for reviewers:

1. `README.md`
2. `reports/final_reports/final_results_index.md`
3. `reports/final_reports/andd_v2_affinity_regression_final_report.md`
4. `reports/andd_stratified/andd_stratified_model_summary.md`
5. `reports/contact_feature_audit/contact_interface_audit_summary.md`
6. `docs/SCRIPT_GUIDE.md`

## 8. Main Reproduction Caveat

Exact numerical reproduction requires the same local datasets, preprocessing choices, random seeds, model checkpoints, and Apple Silicon / MPS training setup used in the original project. This export is meant to document the research workflow and provide executable code structure, not to bundle restricted or very large raw data.
