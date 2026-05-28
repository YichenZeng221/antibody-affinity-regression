# Final Results Index

## Key Final Reports

| Report | Path |
|---|---|
| Final project report | `reports/final_reports/andd_v2_affinity_regression_final_report.md` |
| ANDD stratified model summary | `reports/andd_stratified/andd_stratified_model_summary.md` |
| Contact/interface audit summary | `reports/contact_feature_audit/contact_interface_audit_summary.md` |
| GitHub export manifest | `docs/GITHUB_EXPORT_MANIFEST.md` |
| Script guide | `docs/SCRIPT_GUIDE.md` |

## Final Presentation Figures

| Figure | Path | Purpose |
|---|---|---|
| Figure 1 | `reports/final_reports/figures/final_fig1_prediction_compression_across_splits.png` | Shows prediction compression on train/validation/test. |
| Figure 2 | `reports/final_reports/figures/final_fig2_residual_trend_regression_to_mean.png` | Shows systematic regression-to-the-mean. |
| Figure 3 | `reports/final_reports/figures/final_fig3_multiseed_tailaware_w2.png` | Shows multi-seed tail-aware w2 trade-offs. |
| Figure 4 | `reports/final_reports/figures/final_fig4_cdr3_contact_augmentation.png` | Shows CDR3 contact residual-correction subset results. |
| Figure 5 | `reports/final_reports/figures/final_fig5_contact_interface_availability_funnel.png` | Shows structure/contact feature availability. |
| Figure 6 | `reports/final_reports/figures/final_fig6_cdr_mapping_validation.png` | Shows CDR-to-structure mapping success. |

## Core Takeaway

The main failure mode is systematic prediction compression in absolute affinity regression. Tail-aware loss and contact/interface features provide partial relief, but the remaining bottleneck likely requires richer interaction representation, better calibration, cleaner labels, or ranking-based task formulation.

