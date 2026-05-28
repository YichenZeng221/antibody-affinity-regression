# ANDD Antibody v2 Stratified Basic Interface Geometry Feature Pilot

## Scope

- 本 pilot 只处理 contact availability audit 中 `basic_interface_features_ready_for_extraction=True` 的无歧义 chain-mapping rows。
- 没有处理 ambiguous chain mappings，没有猜测 chain ID，没有训练模型或修改 dataset。
- 结构来源：只读访问 `/Users/yichenzeng/Downloads/all_structures/raw/`。
- `contact_count_*` 的定义是 antibody-antigen **residue pair** contact count：任意一对非氢原子距离小于等于 cutoff，即计为一个接触 residue pair。
- 本轮暂不计算 CDR-specific contacts，因为尚未完成 IMGT CDR residue 到结构 residue 的映射验证。

## Extraction Result

- Pilot-eligible rows from audit: **472**.
- Successfully extracted basic geometry features: **472 / 472**.
- Failed rows: **0 / 472**.
- Successful rows by split: `{'test': 58, 'train': 360, 'val': 54}`.

### Failure Reasons

- None. All eligible rows were parsed successfully.

## Extracted Features

- `min_ab_ag_distance`: antibody heavy/light chains 到 antigen chain(s) 的最小非氢原子距离。
- `contact_count_4A`, `contact_count_5A`, `contact_count_8A`: 不同 cutoff 下的 interface residue-pair 数。
- `antibody_interface_residue_count_5A`, `antigen_interface_residue_count_5A`: 5 A 内涉及的两侧 residue 数。
- `heavy_interface_residue_count_5A`, `light_interface_residue_count_5A`: 5 A 内分别来自 heavy/light chain 的 interface residue 数。

## Feature vs Target Affinity

| feature | n | pearson | spearman |
|---|---|---|---|
| min_ab_ag_distance | 472 | -0.023 | -0.068 |
| contact_count_5A | 472 | 0.198 | 0.173 |

相关性是探索性诊断，不代表因果关系；interface 大小和 affinity 也可能受 antigen 类型、assay noise、结构构象和 label source 共同影响。
- 初步观察：`contact_count_5A` 与 target 只有弱关系（Spearman = 0.173），`min_ab_ag_distance` 基本无单变量关系 （Spearman = -0.068）。
- Geometry QC 注意：有 **1** row 的最小距离 `< 1.0 A`；该异常短距离应在进入建模前核查结构 alternate locations、链选择或坐标质量。

## Feature vs Existing Prediction Error

| model | feature | outcome | n | pearson | spearman |
|---|---|---|---|---|---|
| unweighted_cross_attention | min_ab_ag_distance | absolute_error | 58 | 0.061 | -0.017 |
| unweighted_cross_attention | min_ab_ag_distance | error | 58 | -0.143 | -0.171 |
| tailaware_w2_best_val_tail_mae | min_ab_ag_distance | absolute_error | 58 | 0.073 | -0.049 |
| tailaware_w2_best_val_tail_mae | min_ab_ag_distance | error | 58 | -0.156 | -0.172 |
| unweighted_cross_attention | contact_count_5A | absolute_error | 58 | -0.214 | -0.261 |
| unweighted_cross_attention | contact_count_5A | error | 58 | -0.211 | -0.145 |
| tailaware_w2_best_val_tail_mae | contact_count_5A | absolute_error | 58 | -0.125 | -0.179 |
| tailaware_w2_best_val_tail_mae | contact_count_5A | error | 58 | -0.232 | -0.174 |

## Tail Contact Pattern Audit

- Tail thresholds are defined from train targets only: P10 = **3.3333**, P90 = **6.3204**.
| target_tail | n | target_mean | min_distance_mean | contact_count_5A_mean | antibody_interface_residue_count_5A_mean | antigen_interface_residue_count_5A_mean |
|---|---|---|---|---|---|---|
| above_train_P90 | 58 | 6.943 | 2.483 | 63.586 | 26.448 | 22.121 |
| below_train_P10 | 44 | 2.778 | 2.551 | 49.091 | 20.932 | 17.886 |
| middle_P10_to_P90 | 370 | 4.990 | 2.485 | 62.541 | 24.308 | 22.992 |

- 在这批安全子集里，high-tail 的平均 `contact_count_5A` 为 **63.586**，low-tail 为 **49.091**；这是值得验证的模式，但尚不足以单独解释 affinity tail。
- 特别是 error correlation 仅基于 58 条 test pilot rows，因此不能据此声称 contact features 已经解决 regression-to-the-mean。

## Should These Features Enter a Next Model?

- 这些 features 值得进入下一步 **分析/小型增量 baseline**，因为它们提供了 sequence-only 模型没有看到的真实界面几何信息，并且能够按 `sample_id` 接到已有 residual。
- 但这仍是 pilot 子集：只有无歧义 chain mapping 的样本被纳入。若直接训练，必须清楚说明 subset selection 会改变 benchmark，并先验证 geometry-error relationship 是否稳定。
- 推荐下一步先在当前提取成功的 test pilot rows 上解读相关性，再决定是否为 train/val/test 全部可解析 subset 建独立 contact-feature benchmark。

## CDR-specific Contacts: Still Missing

- CSV 中已有 AbNumber + IMGT 的 CDR sequences；结构中也有 IMGT 文件可供核查。
- 仍需验证 heavy/light sequence 与结构 residue numbering/alignment 的一一对应，特别是 insertion、missing residues、多模型/多复合物链的处理。
- 在该验证通过前，不生成 `CDR-antigen contact count`、`HCDR3 contact fraction` 或 `LCDR3 contact fraction`，以免把错误链或错误 residue 范围作为生物学信号。

## Outputs

- Features: `outputs/andd_antibody_v2_stratified/contact_feature_audit/basic_interface_features.csv`
- Figure: `outputs/final_reports/figures/basic_interface_feature_correlations.png`
- Report: `outputs/andd_antibody_v2_stratified/contact_feature_audit/basic_interface_feature_report.md`
