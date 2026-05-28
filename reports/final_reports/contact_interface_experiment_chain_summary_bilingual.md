# Contact / Interface Feature Experiment Chain Summary

## One-Line Logic

Sequence-only 和 tail-aware 模型都还有明显 regression-to-the-mean，所以我开始检查：真实 antibody-antigen interface / contact 信息是否能解释剩余 error。  
这条实验链路是：

1. **Contact / Interface Availability Audit**：先看结构和 chain mapping 能不能用。
2. **Basic Interface Geometry Extraction**：再提取最基础的 whole-interface 几何特征。
3. **CDR-to-Structure Mapping Validation**：再验证 CDR residue 能不能安全映射到结构坐标。
4. **CDR3 Contact Residual-Correction Baseline**：最后测试 CDR3 contact features 是否能给 sequence model 提供增量价值。

核心结论：contact/interface features 技术上可行，也有一点增量信号，但简单 contact count / Ridge correction 还没有解决 prediction compression。

## 0. Starting Problem

### 解决了什么问题

前面的 sequence-only / tail-aware 阶段说明：

- regression-to-the-mean 同时出现在 train / val / test；
- 这不像典型 overfitting；
- tail-aware loss 能改善 prediction spread 和 tail MAE，但 multi-seed 后没有稳定全面提升 MAE / Spearman；
- 所以问题可能不只是 loss，而是 representation 里缺少真实 interface 信息。

因此下一步不是继续盲目调模型，而是先做 structure/contact feature audit。

## 1. Contact / Interface Availability Audit

### 解决了什么问题

先回答一个最基础的问题：  
**ANDD antibody v2 stratified 里的样本到底有没有可用结构？有没有 reliable heavy/light/antigen chain mapping？**

如果这一步不成立，后面所有 contact feature 都不能做。

### 具体参数 / 规则

- 数据集：
  - `data/processed_affinity/expanded_affinity_antibody_v2_stratified/train.csv`
  - `data/processed_affinity/expanded_affinity_antibody_v2_stratified/val.csv`
  - `data/processed_affinity/expanded_affinity_antibody_v2_stratified/test.csv`

- 总样本数：
  - 1,168 rows

- 结构来源：
  - external SAbDab all-structures archive
  - `/Users/yichenzeng/Downloads/all_structures`

- 检查的结构目录：
  - `raw/`
  - `imgt/`
  - `chothia/`

- metadata 来源：
  - `data/raw/sabdab_summary.tsv`

- 检查内容：
  - PDB structure file 是否存在；
  - Hchain / Lchain / antigen_chain metadata 是否完整；
  - raw PDB 里是否真的有这些 chain；
  - 是否只有一个 viable H/L/antigen chain mapping；
  - 如果有多个 viable mapping，标记 ambiguous，不猜 chain。

- 安全规则：
  - 不计算 contact；
  - 不训练模型；
  - 不修改 dataset；
  - 不复制 31GB structure archive；
  - ambiguous chain mapping 不处理。

### 关键结果

| Item | Result |
|---|---:|
| Total stratified rows | 1,168 |
| External structure file match | 1,168 / 1,168 |
| Complete H/L/antigen chain metadata option | 1,167 / 1,168 |
| Unambiguous viable H/L/antigen mapping | 472 / 1,168 |
| Ambiguous multiple viable mappings | 695 / 1,168 |
| No viable complete chain mapping | 1 / 1,168 |
| Ready for conservative basic interface extraction | 472 / 1,168 |

### 承接关系

这一步解决了“结构到底能不能用”的问题。  
结果是：结构文件覆盖很好，但安全 chain mapping 只覆盖 472 条。因此后续 contact extraction 必须先从这 472 条 conservative pilot subset 开始，不能直接对全部 1,168 条做。

### 输出文件

- `outputs/andd_antibody_v2_stratified/contact_feature_audit/contact_feature_audit_report.md`
- `outputs/andd_antibody_v2_stratified/contact_feature_audit/contact_feature_availability.csv`

## 2. Basic Interface Geometry Extraction

### 解决了什么问题

在 472 条 chain mapping 无歧义的样本上，下一步要回答：  
**基础 antibody-antigen interface geometry 能不能稳定提取？这些简单几何特征和 affinity / error 有没有关系？**

这一步先不做 CDR-specific contact，因为 CDR residue 到结构 residue 的 mapping 还没验证。

### 具体参数 / 规则

- 输入样本：
  - 只处理 `contact_feature_availability.csv` 中 basic-interface-feature-ready 的 472 rows；
  - 不处理 ambiguous chain mapping；
  - 不猜 chain ID。

- 结构来源：
  - `/Users/yichenzeng/Downloads/all_structures/raw/`

- contact 定义：
  - residue-pair contact；
  - 如果 antibody residue 和 antigen residue 之间任意一对 non-hydrogen atoms 距离小于等于 cutoff，就算一个 residue-pair contact。

- distance cutoffs：
  - 4 Å
  - 5 Å
  - 8 Å

- 提取 features：
  - `min_ab_ag_distance`
  - `contact_count_4A`
  - `contact_count_5A`
  - `contact_count_8A`
  - `antibody_interface_residue_count_5A`
  - `antigen_interface_residue_count_5A`
  - `heavy_interface_residue_count_5A`
  - `light_interface_residue_count_5A`

- 暂时不提取：
  - CDR-antigen contact count；
  - HCDR3 contact fraction；
  - LCDR3 contact fraction。

原因：CDR-to-structure residue mapping 尚未验证。

### 关键结果

| Item | Result |
|---|---:|
| Pilot-eligible rows | 472 |
| Successfully extracted | 472 / 472 |
| Failed rows | 0 / 472 |
| Train / val / test successful rows | 360 / 54 / 58 |

Feature vs target affinity:

| Feature | Pearson | Spearman |
|---|---:|---:|
| `contact_count_5A` vs target | 0.198 | 0.173 |
| `min_ab_ag_distance` vs target | -0.023 | -0.068 |

Tail pattern:

| Target group | n | mean `contact_count_5A` |
|---|---:|---:|
| below train P10 | 44 | 49.091 |
| middle P10-P90 | 370 | 62.541 |
| above train P90 | 58 | 63.586 |

### 承接关系

这一步解决了“基础 interface features 能不能提取”的问题。  
结果是：能稳定提取，472 / 472 成功。但 whole-interface features 和 target 的相关性很弱，只能说明 structure 可能有信号，不能解释主要 error。

所以自然引出下一步：也许 signal 不是 whole-interface level，而是在 CDR-antigen contact，尤其 CDR3-antigen contact。

### 输出文件

- `outputs/andd_antibody_v2_stratified/contact_feature_audit/basic_interface_features.csv`
- `outputs/andd_antibody_v2_stratified/contact_feature_audit/basic_interface_feature_report.md`
- `outputs/final_reports/figures/basic_interface_feature_correlations.png`

## 3. CDR-to-Structure Mapping Validation

### 解决了什么问题

Basic interface features 信号较弱，所以需要进一步问：  
**能不能把 sequence-level 的 CDR annotation 安全映射到 PDB structure residue 上？**

如果 CDR residue mapping 不可靠，就不能计算 CDR-specific contact features。

### 具体参数 / 规则

- 输入样本：
  - 472 条 unambiguous chain-mapping pilot samples。

- CDR annotation 来源：
  - AbNumber + IMGT；
  - 使用已有 `HCDR1/HCDR2/HCDR3/LCDR1/LCDR2/LCDR3`。

- 结构来源：
  - raw SAbDab PDB。

- strict mapping rules：
  1. 每个 CDR sequence 必须在对应 full heavy/light sequence 中唯一出现；
  2. full chain sequence 到 structure chain sequence 的 alignment identity 至少 95%；
  3. alignment coverage 至少 80%；
  4. 每个 CDR residue 必须能映射到有坐标的 structure residue；
  5. amino acid identity 必须一致；
  6. insertion/deletion ambiguity 不强行处理，直接标记失败。

- 计算的 preliminary CDR features：
  - `all_cdr_contact_count_5A`
  - `hcdr3_contact_count_5A`
  - `lcdr3_contact_count_5A`
  - `hcdr3_contact_fraction_5A`
  - `lcdr3_contact_fraction_5A`
  - `cdr_interface_residue_count_5A`
  - `cdr_min_distance`

### 关键结果

| Item | Result |
|---|---:|
| Pilot rows validated | 472 |
| All-six-CDR contact-safe rows | 422 / 472 |
| HCDR3-only contact-safe rows | 468 / 472 |
| LCDR3-only contact-safe rows | 470 / 472 |
| HCDR3+LCDR3 jointly contact-safe rows | 467 / 472 |
| Failed mapping rows | 50 / 472 |
| All-six-CDR train / val / test rows | 322 / 51 / 49 |

CDR mapping success rate:

| CDR | Success rate |
|---|---:|
| LCDR3 | 99.58% |
| HCDR2 | 99.36% |
| HCDR3 | 99.15% |
| HCDR1 | 98.73% |
| LCDR1 | 98.52% |
| LCDR2 | 92.16% |

Failure reasons:

| Reason | Rows |
|---|---:|
| insertion/deletion ambiguity | 34 |
| unresolved residues | 16 |
| chain sequence mismatch | 2 |
| missing CDR annotation | 0 |
| antigen chain mismatch | 0 |

### 承接关系

这一步解决了“CDR-specific contact features 是否可信”的问题。  
结果是：HCDR3/LCDR3 mapping 非常稳定，joint HCDR3+LCDR3 contact-safe 有 467 rows；all-six-CDR 也有 422 rows。

因此下一步可以做一个小型 CDR3-contact feature augmented baseline，验证 CDR3 geometry 是否有增量价值。

### 输出文件

- `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr_mapping_validation_report.md`
- `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr_mapping_availability.csv`
- `outputs/andd_antibody_v2_stratified/contact_feature_audit/preliminary_cdr_contact_features.csv`

## 4. CDR3 Contact Residual-Correction Baseline

### 解决了什么问题

在验证 CDR3 contact-safe subset 后，最后要回答：  
**真实 CDR3 interface geometry 是否能在 sequence model prediction 之外提供增量信息？**

这里不训练新的 deep model，只做 post-hoc residual correction，避免把 architecture 复杂化。

### 具体参数 / 规则

- 不训练新的 neural network。

- 方法：
  - 先使用已有 sequence model prediction；
  - 定义 residual：
    - `residual = target - sequence_prediction`
  - 在 train subset 上用 Ridge 学 residual；
  - 在 test subset 上修正 prediction：
    - `corrected_prediction = sequence_prediction + predicted_residual`

- residual model：
  - `Ridge(alpha=1.0)`

- tail thresholds：
  - 从 full stratified train split 计算；
  - train P10 = 3.3333；
  - train P90 = 6.3204。

- sequence baselines：
  - `unweighted_cross_attention`
  - `tailaware_w2_best_val_tail_mae`

- subset 1：
  - `HCDR3+LCDR3 contact-safe`
  - total rows = 467
  - train rows = 356
  - test rows = 58
  - features:
    - `hcdr3_contact_count_5A`
    - `lcdr3_contact_count_5A`
    - `hcdr3_contact_fraction_5A`
    - `lcdr3_contact_fraction_5A`

- subset 2：
  - `All-CDR contact-safe`
  - total rows = 422
  - train rows = 322
  - test rows = 49
  - features:
    - `hcdr3_contact_count_5A`
    - `lcdr3_contact_count_5A`
    - `hcdr3_contact_fraction_5A`
    - `lcdr3_contact_fraction_5A`
    - `cdr_min_distance`
    - `all_cdr_contact_count_5A`

### 关键结果：Tail-aware w2

HCDR3+LCDR3 safe subset:

| Metric | Sequence only | Sequence + CDR3 contact |
|---|---:|---:|
| MAE | 0.9740 | 0.9658 |
| RMSE | 1.2857 | 1.2745 |
| Spearman | 0.3796 | 0.3984 |
| pred std / true std | 0.5021 | 0.4969 |
| tail MAE | 1.8701 | 1.8515 |

All-CDR safe subset:

| Metric | Sequence only | Sequence + CDR3 contact |
|---|---:|---:|
| MAE | 0.9278 | 0.9016 |
| RMSE | 1.2105 | 1.1815 |
| Spearman | 0.2883 | 0.3423 |
| pred std / true std | 0.5837 | 0.5699 |
| tail MAE | 1.5660 | 1.5542 |

### 承接关系

这一步解决了“CDR3 geometry 是否有增量价值”的问题。  
结果是：有小幅增量价值，尤其在 tail-aware w2 setting 下，两个 subset 的 MAE、RMSE、Spearman、tail MAE 都有改善。

但是：

- pred std / true std 没有改善；
- error-vs-true bias 没有明显接近 0；
- 所以 CDR3 contact Ridge correction 没有解决 prediction compression。

因此结论不是“contact features 已经解决问题”，而是：

> CDR3 contact features 技术上可提取，也有弱增量信号；但简单 scalar contact count + linear residual correction 不够，下一步需要 richer structure/contact-aware representation。

### 结论边界

这是 contact-covered subset analysis，不是 full 1,168-row benchmark。  
不能直接说它优于 full benchmark model，因为 subset selection 改变了评估样本。

### 输出文件

- `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr3_contact_augmented_baseline_report.md`
- `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr3_contact_augmented_metrics.csv`
- `outputs/final_reports/figures/final_fig4_cdr3_contact_augmentation.png`

## 5. Final Report / Bilingual Summary

### 解决了什么问题

最终报告把两条线连起来：

1. sequence-only / tail-aware 模型为什么仍然有 regression-to-the-mean；
2. contact/interface features 是否能解释或者缓解这个问题。

最终结论是：

- sequence-only 模型有系统性 prediction compression；
- tail-aware loss 能缓解 spread/tail，但 multi-seed 后不是稳定全面提升；
- structure/contact 信息技术上可提取；
- basic whole-interface features 信号弱；
- CDR3 contact features 有小幅 subset gains；
- 但 simple contact counts / Ridge residual correction 没有解决核心 compression；
- 下一步应该做 richer structure/contact-aware modeling，而不是继续只加 scalar features。

### 相关最终文件

- `outputs/final_reports/andd_v2_affinity_regression_final_report.md`
- `outputs/final_reports/andd_v2_affinity_regression_final_report_zh.md`
- `outputs/final_reports/andd_v2_four_figure_presentation_summary_bilingual.md`
- `outputs/final_reports/figures/final_fig1_prediction_compression_across_splits.png`
- `outputs/final_reports/figures/final_fig2_residual_trend_regression_to_mean.png`
- `outputs/final_reports/figures/final_fig3_multiseed_tailaware_w2.png`
- `outputs/final_reports/figures/final_fig4_cdr3_contact_augmentation.png`

## Short Version To Send

Sequence-only 和 tail-aware 模型仍然有 regression-to-the-mean，所以我开始检查真实 antibody-antigen interface 信息是否能解释剩余 error。第一步做 contact/interface availability audit：SAbDab structure archive 覆盖全部 1,168 条 ANDD antibody v2 stratified rows，但只有 472 条有无歧义 H/L/antigen chain mapping，可以安全做 contact extraction。第二步在这 472 条上提取 basic interface geometry，全部成功，但 whole-interface contact_count 和 target 只有弱相关。第三步做 CDR-to-structure mapping validation：HCDR3+LCDR3 contact-safe 有 467 条，all-CDR contact-safe 有 422 条，说明 CDR-specific contact features 技术上可行。最后用 CDR3 contact features 做 Ridge residual correction，在 contact-covered subset 上 MAE、Spearman、tail MAE 有小幅改善，但 prediction spread 没有改善。因此结论是：contact features 有弱增量信号，但 simple scalar contact counts 还不能解决 prediction compression；下一步应该做更丰富的 structure/contact-aware representation。
