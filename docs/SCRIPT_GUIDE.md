# Script Guide

This repository keeps the scripts from the local project because they show the full research workflow. The most important scripts are grouped conceptually below.

## Dataset and Audit

- `scripts/audit_andd_data_source.py`
- `scripts/build_andd_v2_candidates.py`
- `scripts/audit_andd_antibody_v2_candidates.py`
- `scripts/build_andd_antibody_v2_stratified_split.py`

Purpose: inspect ANDD, define conservative antibody-only candidates, and create the stratified antigen-level split.

## CDR Extraction and CDR-Aware Modeling

- `scripts/extract_standard_cdrs_andd_antibody_v2.py`
- `scripts/check_andd_antibody_v2_cdr_batch_shapes.py`
- `run_train_affinity_cdr.py`
- `scripts/evaluate_andd_antibody_v2_stratified_cdr_test_set.py`

Purpose: use standard AbNumber / IMGT CDR extraction and train/evaluate all-CDR pooled baselines.

## Cross-Attention and Tail-Aware Training

- `run_train_affinity_cross_attention.py`
- `run_train_affinity_cross_attention_tailaware.py`
- `scripts/check_andd_stratified_cross_attention_tailaware_batch_shapes.py`
- `scripts/evaluate_andd_antibody_v2_stratified_cross_attention_test_set.py`
- `scripts/evaluate_andd_antibody_v2_stratified_tailaware_checkpoints.py`
- `scripts/summarize_andd_stratified_cross_attention_multiseed.py`

Purpose: train all-CDR cross-attention models, test tail-aware loss, and aggregate multi-seed results.

## Error Analysis and Figures

- `scripts/analyze_andd_stratified_model_fit.py`
- `scripts/make_andd_final_presentation_figure_set.py`
- `scripts/make_contact_interface_audit_figures.py`

Purpose: diagnose prediction compression, regression-to-the-mean, and generate final presentation figures.

## Contact / Interface Feature Audit

- `scripts/audit_andd_stratified_contact_feature_availability.py`
- `scripts/extract_andd_stratified_basic_interface_features.py`
- `scripts/validate_andd_stratified_cdr_structure_mapping.py`
- `scripts/analyze_andd_stratified_cdr3_contact_augmented_baseline.py`

Purpose: check structure availability, extract safe basic interface features, validate CDR-to-structure mapping, and test CDR3 contact residual correction.

## Notes

Some scripts are retained for traceability from earlier project stages. The final reports and selected configs identify the main experiments that should be read first.
