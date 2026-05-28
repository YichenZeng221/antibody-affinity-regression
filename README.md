# SeqProFT-style Antibody-Antigen Affinity Regression

This repository is a research portfolio export of a first-stage antibody-antigen affinity regression project using ESM2 + LoRA-style fine-tuning.

It is intentionally organized to show the full research loop: dataset audit, split design, model baselines, error diagnosis, multi-seed validation, contact/interface feature audit, and final scientific reporting.

The main benchmark in the final stage is **ANDD antibody v2 stratified split**, with target:

```text
y = -log10(Kd)
```

Higher values mean stronger binding.

## Project Summary

The project started from sequence-only affinity regression and progressed through:

1. ANDD antibody-only data audit and conservative benchmark construction.
2. Standard CDR extraction using AbNumber / IMGT.
3. All-CDR pooled and all-CDR cross-attention baselines.
4. Regression-to-the-mean diagnosis across train / validation / test.
5. Tail-aware loss experiments and multi-seed validation.
6. Contact/interface feature availability audit.
7. Basic interface geometry extraction.
8. CDR-to-structure mapping validation.
9. CDR3 contact residual-correction subset analysis.

The central finding is that absolute affinity regression shows systematic prediction compression: low-affinity examples tend to be overpredicted and high-affinity examples tend to be underpredicted. Tail-aware loss and CDR3 contact features provide partial improvements, but simple scalar contact features do not fully solve the compression problem.

## Why This Project Matters

This project is not just a leaderboard experiment. It demonstrates:

- Careful biomedical dataset curation and leakage-aware splitting.
- Standard CDR extraction rather than fixed-index slicing.
- Sequence-only, CDR-aware, cross-attention, and tail-aware modeling.
- Systematic error analysis instead of blindly changing architectures.
- Multi-seed validation before trusting a single positive result.
- Structure/contact feature feasibility checks before claiming structure-aware modeling.
- Scientific honesty about what the model did and did not solve.

## Repository Contents

```text
src/                          Core model, dataset, training, and evaluation code
scripts/                      Data audit, CDR extraction, evaluation, plotting scripts
configs/                      Four selected final experiment configs
reports/final_reports/        Final English reports
reports/final_reports/figures/ Presentation-ready figures
reports/andd_stratified/      Key ANDD stratified model reports and summary CSVs
reports/contact_feature_audit/ Contact/interface audit reports and lightweight feature CSVs
data_reports/                 Dataset split and audit summary reports
notes/                        Paper notes and reproduction plan
docs/                         GitHub export manifest and script guide
```

## Important Reports

- `reports/final_reports/andd_v2_affinity_regression_final_report.md`
- `reports/final_reports/final_results_index.md`
- `reports/andd_stratified/andd_stratified_model_summary.md`
- `reports/contact_feature_audit/contact_interface_audit_summary.md`

## Presentation Figures

- `reports/final_reports/figures/final_fig1_prediction_compression_across_splits.png`
- `reports/final_reports/figures/final_fig2_residual_trend_regression_to_mean.png`
- `reports/final_reports/figures/final_fig3_multiseed_tailaware_w2.png`
- `reports/final_reports/figures/final_fig4_cdr3_contact_augmentation.png`
- `reports/final_reports/figures/final_fig5_contact_interface_availability_funnel.png`
- `reports/final_reports/figures/final_fig6_cdr_mapping_validation.png`

## Selected Final Configs

The repo keeps only the key final configs, rather than every intermediate sweep file:

- `configs/config_affinity_andd_antibody_v2_stratified_all_cdr_pooled_lr3e-5_e10.yaml`
- `configs/config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_lr3e-5_e10.yaml`
- `configs/config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_tailaware_w2_lr3e-5_e20.yaml`
- `configs/config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_unweighted_s42_lr3e-5_e20.yaml`

## Data Availability

Large raw datasets and structures are intentionally **not included** in this GitHub export.

Local-only inputs used during the project included:

- ANDD v2 spreadsheet
- SAbDab summary files
- SAbDab all-structures archive
- processed train / validation / test CSVs
- model checkpoints

The code expects local data paths matching the original project structure. See `data_reports/` for dataset construction and split summaries.

## Reproducibility Notes

Training used the project `.venv` environment. Standard CDR extraction used a separate `abnumber-cdr` conda environment with AbNumber / ANARCI / HMMER.

Example commands from the final stage:

```bash
./.venv/bin/python run_train_affinity_cross_attention.py \
  --config configs/config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_lr3e-5_e10.yaml

./.venv/bin/python scripts/evaluate_andd_antibody_v2_stratified_cross_attention_test_set.py \
  --config configs/config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_lr3e-5_e10.yaml
```

For GitHub users, paths in configs may need to be adjusted to local data locations.

## What Is Not Included

To keep the repository lightweight and GitHub-safe, this export excludes:

- raw ANDD spreadsheets
- SAbDab all-structures archive
- local PDB cache
- model checkpoints
- virtual environments
- large prediction dumps

The final figures and summary CSVs needed to understand the project are included.

## Main Scientific Takeaway

The model did not simply fail; rather, the absolute affinity regression setup exposed a systematic bottleneck. Sequence-only models compressed predictions toward the mean. Tail-aware loss improved prediction spread and tail MAE in some settings, but multi-seed validation showed the gains were not uniformly stable. Contact/interface features were technically feasible and gave small subset gains, but simple contact counts were not enough to solve the core compression issue.

This motivates future work on richer structure/contact-aware representations and ranking-based antibody binder prioritization.

## Recommended Next Step

The natural continuation is a ranking-based antibody binder prioritization project inspired by AbRank-style task framing. The current project shows why exact calibrated affinity regression is difficult; the next project can ask whether pairwise or listwise ranking better supports candidate prioritization.
