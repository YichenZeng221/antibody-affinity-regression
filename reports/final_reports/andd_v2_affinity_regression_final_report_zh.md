# ANDD Antibody v2 Affinity Regression：第一阶段最终中文报告

## 1. Executive Summary / 执行摘要

这一阶段的目标，是把 ANDD 中可用的 antibody-antigen affinity 数据整理成一个保守、可复现、尽量避免 leakage 的 antibody-only benchmark，并用它系统诊断当前 SeqProFT-style ESM2 + LoRA affinity regression pipeline 的主要瓶颈。

最核心结论是：

> 当前主要 failure mode 是 **regression-to-the-mean**：低 affinity target 被模型预测得偏高，高 affinity target 被模型预测得偏低。模型不是完全不会排序或不会学习，而是预测范围明显被压缩到中间。

这个结论不是来自单一图或单一模型，而是由多组检查共同支持：

- 原始 ANDD antibody v2 all-CDR pooled baseline 上，test `pred_std / true_std = 0.3772`，`error_vs_true_Pearson = -0.9262`。
- 在新的 stratified antigen-level split 上，pooled all-CDR 和 all-CDR cross-attention 仍然都有 prediction compression。
- train、validation、test 上都能看到 compression，因此不像典型 overfit。典型 overfit 应该是 train 拟合很好、val/test 才崩；这里 train 本身也没有学出完整 target range。
- regression head 输出是 unrestricted scalar，没有 `sigmoid`、`tanh`、prediction `clamp`，也没有发现 target inverse-transform 错误。
- stratified split 保证 val/test 覆盖 P5/P95 tail，并且 train/val/test antigen overlap 仍然为 0，但 compression 仍然存在，所以这也不是简单 split artifact。

模型结果需要诚实解释：

- 在 stratified benchmark 上，pooled all-CDR 的 overall MAE 略好：`0.9373`，cross-attention 为 `0.9523`。
- 但 cross-attention 改善了 Spearman、prediction spread、residual bias 和 upper-tail behavior。
- tail-aware loss 可以缓解 prediction spread 和 tail MAE，但三随机种子验证后发现，它没有稳定全面超过 unweighted baseline 的 MAE/RMSE/Spearman。
- contact/interface features 技术上可提取；CDR3 geometry 在 contact-covered subset 中对 tail-aware w2 有小幅增量收益。
- 但 simple contact count + Ridge residual correction 仍然没有解决 prediction compression。

因此，当前最稳妥的解释是：瓶颈更可能来自 sequence/structure representation 不够充分、tail data scarcity、label noise / assay heterogeneity，以及 absolute Kd regression framing 本身比较难校准。这个结果支持未来探索 ranking-based objective，但不能过度声称“已经证明 absolute Kd regression 本身有固有缺陷”。

## 2. Dataset and Task Definition / 数据集与任务定义

### 2.1 任务定义

本项目任务是 antibody-antigen binding affinity regression。

- 输入：抗体 CDR 序列和 antigen sequence。
- 主要正式 ANDD baseline 输入：`HCDR1`、`HCDR2`、`HCDR3`、`LCDR1`、`LCDR2`、`LCDR3` 加 `antigen_sequence`。
- target：`neg_log10_affinity_candidate = -log10(Kd[M])`。
- target 解释：数值越大，代表 binding 越强。

这里预测的是 absolute affinity/Kd，而不是二分类，也不是 pairwise ranking。这一点很重要，因为 absolute Kd regression 对 label noise、assay difference 和 calibration 非常敏感。

### 2.2 从 ANDD 原始数据到 conservative antibody benchmark

| 阶段 | rows | 解释 |
|---|---:|---|
| ANDD 原始 workbook | 48,800 | 混合 antibody、nanobody、多来源、多 label 类型 |
| Tier 1 affinity candidates | 4,382 | experimental quantitative Kd-like rows，且有可用 sequence |
| Antibody candidates | 3,116 | antibody-only candidate pool |
| Conservative antibody `keep_safe` rows | 1,168 | 最终 ANDD antibody v2 modeling pool |

关键筛选原则：

- antibody 和 nanobody 分开处理，不混成一个主任务。
- predicted affinity rows，包括 ANTIPASTI-derived labels，不进入主 supervised benchmark。
- 对 extreme Kd、sequence issue、duplicate、overlap 进行审计后，只用 `keep_safe` rows 构建 antibody-only v2。

### 2.3 标准 CDR 提取

正式 CDR-aware modeling 使用 `AbNumber + IMGT` 重新提取 CDR，不直接依赖 ANDD source-provided CDR 字段，也不使用固定 index slicing。

| Split | Rows | Heavy CDR success | Light CDR success | Both success |
|---|---:|---:|---:|---:|
| Train | 934 | 934 | 934 | 934 |
| Validation | 117 | 117 | 117 | 117 |
| Test | 117 | 117 | 117 | 117 |

这说明 ANDD antibody v2 在 CDR extraction 层面非常干净：1,168 条全部成功。

### 2.4 原始 antigen-group split

| Split | Rows | Antigen groups | Target mean | Target min | Target max |
|---|---:|---:|---:|---:|---:|
| Train | 934 | 644 | 4.9467 | 2.0031 | 11.8155 |
| Validation | 117 | 79 | 4.8944 | 2.0938 | 6.9811 |
| Test | 117 | 79 | 5.0281 | 2.1533 | 8.4692 |

Leakage check：

- train vs validation antigen overlap：`0`
- train vs test antigen overlap：`0`
- validation vs test antigen overlap：`0`

### 2.5 Target distribution 诊断

用 train tertiles 定义 low/mid/high bins：

| Split | Low target | Mid target | High target |
|---|---:|---:|---:|
| Train | 312 (33.4%) | 311 (33.3%) | 311 (33.3%) |
| Validation | 38 (32.5%) | 38 (32.5%) | 41 (35.0%) |
| Test | 43 (36.8%) | 31 (26.5%) | 43 (36.8%) |

粗粒度 low/mid/high 并没有严重不平衡。但真正极端的 tail 仍然有限：按 train P10/P90 定义，每个 tail 在 train 中只有 94 条。原始 validation set 的 high-end coverage 也偏弱，最大 target 只有 `6.9811`，而 train 最大值达到 `11.8155`。

## 3. Experimental Timeline / 实验流程

| 阶段 | 问题 | 结果 |
|---|---|---|
| ANDD audit 和 QC | ANDD 是否有可靠 experimental antibody affinity rows？ | 得到 1,168 条 conservative antibody rows |
| 标准 CDR extraction | CDR-aware input 是否能统一标准化？ | AbNumber + IMGT 对 1,168 / 1,168 成功 |
| 原始 split pooled baseline | 数据是否可训练？ | MAE `0.9066`，但 prediction compression 明显 |
| output/head/data 排查 | compression 是代码或 leakage 问题吗？ | 没有 output range 限制；antigen overlap 为 0 |
| Linear calibration | 后处理校准能否缓解 compression？ | spread 和 low-target error 改善，但有 tradeoff |
| Stratified antigen split | 更好的 tail coverage 能否解决问题？ | 不能；leakage-safe tail-covered split 上仍有 compression |
| Cross-attention baseline | learnable CDR-antigen interaction 是否有帮助？ | 改善 ranking/spread/upper-tail，但不是最佳 MAE |
| Tail-aware w3/w2 | loss weighting 能否缓解 tail/compression？ | single seed 有帮助；w2 比 w3 更平衡 |
| Multi-seed validation | w2 是否稳定更好？ | 改善 tail/spread，但没有稳定赢 overall MAE/Spearman |
| Contact/interface audit | 是否有可用结构界面信息？ | 472 条 unambiguous rows 可提取 basic interface features |
| CDR mapping validation | CDR-specific contacts 是否可信？ | all-CDR safe 422 条；HCDR3+LCDR3 safe 467 条 |
| CDR3 contact augmentation | 少量真实 geometry features 是否有增量？ | 对 tail-aware w2 有小幅 subset gain，但 compression 未解决 |

## 4. Main Failure Mode: Regression-to-the-Mean / 主要失败模式

### 4.1 原始 ANDD all-CDR pooled baseline 的错误模式

| Metric | Value |
|---|---:|
| Test rows | 117 |
| MAE | 0.9066 |
| RMSE | 1.1281 |
| Spearman | 0.3817 |
| Prediction std | 0.4665 |
| `pred_std / true_std` | 0.3772 |
| `error_vs_true_Pearson` | -0.9262 |

按 target bin 看：

| Target bin | Rows | MAE | Mean prediction error |
|---|---:|---:|---:|
| Low target | 39 | 1.2603 | +1.2603 |
| Mid target | 39 | 0.3614 | +0.0620 |
| High target | 39 | 1.0981 | -1.0981 |

解释：

- low target mean error 为正，说明 weak-binding / low-affinity 样本被预测得太高。
- high target mean error 为负，说明 strong-binding / high-affinity 样本被预测得太低。
- mid target 表现最好，说明模型更愿意预测中间区域。

这就是典型 regression-to-the-mean：模型不是完全随机，而是过度保守，避免预测极端值。

### 4.2 Train/validation/test 上都出现 compression

在 stratified split 上，train 本身也存在明显 compression：

| Model | Split | MAE | Spearman | `pred_std / true_std` | `error_vs_true_Pearson` |
|---|---|---:|---:|---:|---:|
| Pooled all-CDR | Train | 0.7497 | 0.5831 | 0.4351 | -0.9041 |
| Pooled all-CDR | Validation | 0.9588 | 0.4239 | 0.3114 | -0.9504 |
| Pooled all-CDR | Test | 0.9373 | 0.3699 | 0.3170 | -0.9484 |
| All-CDR cross-attention | Train | 0.7597 | 0.5403 | 0.4584 | -0.8898 |
| All-CDR cross-attention | Validation | 0.9825 | 0.4188 | 0.3831 | -0.9239 |
| All-CDR cross-attention | Test | 0.9523 | 0.3861 | 0.3925 | -0.9202 |

这说明问题不像典型 overfit。典型 overfit 应该是 train 拟合得很好，validation/test 才变差；但这里 train prediction range 也没有展开。更合理的解释是：当前 objective / representation / label tail 本身让模型不容易学习极端 affinity。

## 5. What We Ruled Out / 已排除的问题

### 5.1 不是 output activation 或 prediction clamp

检查模型代码后确认：

- pooled CDR model 最后一层是 unrestricted `nn.Linear(..., 1)`。
- cross-attention model 输出是 `Linear -> GELU -> Dropout -> Linear -> scalar`。
- 没有 `sigmoid`、`tanh`、prediction `clamp` 或 bounded output transform。
- pooling 里的 `clamp(min=1e-9)` 只是防止 padding token count 除以 0，不会限制 prediction。

所以 prediction compression 不是输出层把范围硬压住。

### 5.2 不是 target normalization / inverse transform 错误

- config 中 target 一直是 `neg_log10_affinity_candidate`。
- evaluation 比较的是 `true_neg_log10_affinity` 和 `predicted_neg_log10_affinity`。
- metric 计算前没有发现错误 inverse transform。

所以目前没有证据说明 target transform pipeline 出错。

### 5.3 不是简单 leakage 问题

原始 split 和 stratified split 都使用 antigen-sequence group split。正式 stratified split 中：

- train/validation antigen overlap：`0`
- train/test antigen overlap：`0`
- validation/test antigen overlap：`0`

因此，当前问题不是因为相同 antigen 泄漏导致的虚假表现。

### 5.4 不是简单 split artifact

stratified split 明确增强了 val/test tail coverage：

| Split | Rows | Antigen groups | Target min | Target max | Global P5 rows | Global P95 rows |
|---|---:|---:|---:|---:|---:|---:|
| Train | 934 | 568 | 2.0449 | 8.5343 | 42 | 43 |
| Validation | 117 | 117 | 2.0031 | 11.8155 | 10 | 9 |
| Test | 117 | 117 | 2.0349 | 10.9208 | 7 | 7 |

即使 val/test 都覆盖 tail，pooled model 仍然有 `pred_std / true_std = 0.3170` 和 `error_vs_true_Pearson = -0.9484`。所以 split 本身不是唯一原因。

## 6. Model Results / 模型结果

### 6.1 Benchmark boundary / 可比性边界

目前 ANDD branch 没有完成 whole-sequence baseline prediction file；正式 ANDD modeling 从标准 all-CDR pooled input 开始。之前 `unified_no_high_risk` 上的 whole-sequence/CDR 结果是设计动机，但不能直接和 ANDD test set 横向比较。

### 6.2 原始 ANDD antibody v2 baseline

| Model | Split | MAE | RMSE | Spearman | `pred_std / true_std` | `error_vs_true_Pearson` | Low MAE | High MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| All-CDR pooled | Original antigen split | 0.9066 | 1.1281 | 0.3817 | 0.3772 | -0.9262 | 1.2603 | 1.0981 |

这个结果说明 ANDD antibody v2 是可训练的，但也说明只看 MAE 会掩盖问题：模型预测范围严重被压缩。

### 6.3 Stratified split 上的同 split 对比

这个表里的两个模型是在同一个 stratified test set 上评估，因此可以公平比较：

| Model | MAE | RMSE | Spearman | `pred_std / true_std` | `error_vs_true_Pearson` | Low MAE | High MAE | Below P10 MAE | Above P90 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All-CDR pooled | 0.9373 | 1.3056 | 0.3699 | 0.3170 | -0.9484 | 1.1674 | 1.3367 | 1.8650 | 2.0887 |
| All-CDR cross-attention | 0.9523 | 1.3008 | 0.3861 | 0.3925 | -0.9202 | 1.4182 | 1.1147 | 2.0273 | 1.8483 |

解读：

- pooled all-CDR 的 MAE 略好。
- cross-attention 的 RMSE 和 Spearman 略好。
- cross-attention 的 prediction spread 更健康，`error_vs_true_Pearson` 也更接近 0。
- cross-attention 改善 high-target / above-P90，但 worsens low-target / below-P10。

所以 cross-attention 不是简单“全面更好”，而是更像：它在 ranking、prediction spread 和 high-affinity tail 上有价值，但 overall MAE 还没赢。

## 7. Calibration Result / 后处理校准结果

在原始 ANDD split 上，用 validation predictions 拟合 post-hoc linear calibration：

```text
calibrated_pred = 1.405797 * raw_pred - 2.274080
```

| Prediction | MAE | RMSE | Spearman | `pred_std / true_std` | `error_vs_true_Pearson` | Low MAE | High MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw pooled prediction | 0.9066 | 1.1281 | 0.3817 | 0.3772 | -0.9262 | 1.2603 | 1.0981 |
| Linearly calibrated | 0.9015 | 1.1424 | 0.3817 | 0.5302 | -0.8508 | 1.0173 | 1.2347 |

解读：

- Calibration 改善了 prediction spread、residual bias 和 low-target MAE。
- MAE 也略微改善。
- 但 RMSE 和 high-target MAE 变差。
- Spearman 没变，因为线性校准 slope 为正，不改变 ranking。

这说明 output scale 是症状的一部分，但校准本身不增加 biological information，也不能真正解决 tail prediction。

## 8. Tail-Aware Training and Multi-Seed Validation / Tail-aware 训练与多随机种子验证

以下实验都基于 stratified antigen-level split 和 all-CDR cross-attention。Tail thresholds 只用 train split 计算：

- Train P10 = `3.3333`
- Train P90 = `6.3204`

### 8.1 Single-seed 结果

| Model / Validation-selected checkpoint | MAE | RMSE | Spearman | `pred_std / true_std` | `error_vs_true_Pearson` | Below P10 MAE | Above P90 MAE | Tail MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unweighted cross-attention, e10 | 0.9523 | 1.3008 | 0.3861 | 0.3925 | -0.9202 | 2.0273 | 1.8483 | 1.9378 |
| Tail-aware w3.0, best val tail MAE | 0.9873 | 1.3442 | 0.4241 | 0.7284 | -0.7308 | 1.4901 | 1.7391 | 1.6146 |
| Tail-aware w2.0, best val tail MAE | 0.9426 | 1.2938 | 0.4478 | 0.5829 | -0.8222 | 1.5632 | 1.8046 | 1.6839 |

single-seed 读法：

- w3.0 明显拉开 prediction spread，也改善 tail MAE，但 overall MAE/RMSE 变差，说明太激进。
- w2.0 在 seed 42 上看起来很漂亮，同时改善 MAE、RMSE、Spearman、spread、bias 和 tail MAE。
- 但 single-seed 不足以证明稳定性，因此后面做了 multi-seed validation。

### 8.2 Controlled multi-seed validation

控制变量：

- architecture 相同：all-CDR cross-attention
- split 相同：ANDD antibody v2 stratified antigen split
- lr 相同：`3e-5`
- epochs 相同：`20`
- batch size 相同：`1`
- checkpoint policy 相同：best validation tail MAE
- seeds：`42`、`123`、`2026`

| Model Group | Seeds | MAE | RMSE | Spearman | `pred_std / true_std` | `error_vs_true_Pearson` | Below P10 MAE | Above P90 MAE | Tail MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unweighted cross-attention, e20 | 3 | **0.9224 +/- 0.0091** | **1.2466 +/- 0.0177** | **0.4750 +/- 0.0188** | 0.5106 +/- 0.0345 | -0.8608 +/- 0.0193 | 1.7708 +/- 0.0536 | 1.6601 +/- 0.1405 | 1.7155 +/- 0.0460 |
| Tail-aware w2.0, e20 | 3 | 0.9420 +/- 0.0020 | 1.2802 +/- 0.0137 | 0.4551 +/- 0.0085 | **0.5935 +/- 0.0350** | **-0.8135 +/- 0.0222** | **1.6390 +/- 0.0754** | **1.6581 +/- 0.1270** | **1.6486 +/- 0.0373** |

W2 minus unweighted：

| Metric | Delta | 解读 |
|---|---:|---|
| MAE | +0.0196 | w2 overall MAE 更差 |
| RMSE | +0.0335 | w2 squared-error scale 更差 |
| Spearman | -0.0199 | w2 ranking 不如 unweighted 稳定 |
| `pred_std / true_std` | +0.0829 | w2 prediction spread 更健康 |
| `error_vs_true_Pearson` | +0.0473 | w2 regression-to-the-mean 较弱 |
| Below P10 MAE | -0.1318 | w2 明显改善 lower tail |
| Above P90 MAE | -0.0020 | upper tail 基本打平，w2 略好 |
| Tail MAE | -0.0669 | w2 modestly 改善 combined tail |

最终 tail-aware 结论：

> tail-aware loss 可以改善 prediction spread，并略微改善 tail MAE，尤其是 lower tail；但它没有在多随机种子下稳定改善 MAE/RMSE/Spearman。

所以 w2 是有价值的 objective variant，但不能说它已经稳定替代 unweighted baseline。

## 9. Contact/Interface Feature Analysis / 接触界面特征分析

### 9.1 Structure 和 chain mapping availability

SAbDab external structure archive 覆盖很好，但 chain mapping ambiguity 限制了保守提取：

| Availability item | Rows |
|---|---:|
| Total stratified ANDD rows | 1,168 |
| External SAbDab raw/imgt/chothia structure match | 1,168 / 1,168 |
| Complete H/L/antigen chain metadata option | 1,167 / 1,168 |
| Unambiguous viable H/L/antigen chain mapping | 472 / 1,168 |
| Ambiguous multiple viable chain mappings, not guessed | 695 / 1,168 |
| No viable complete chain mapping found | 1 / 1,168 |

这里必须强调：contact analysis 是 subset analysis，不是 full 1,168-row benchmark。

### 9.2 Basic interface geometry pilot

对 472 条 unambiguous rows 成功提取 basic geometry features：

- `min_ab_ag_distance`
- `contact_count_4A`
- `contact_count_5A`
- `contact_count_8A`
- antibody/antigen/heavy/light interface residue counts at 5 A

| Feature vs target affinity | Rows | Pearson | Spearman |
|---|---:|---:|---:|
| `min_ab_ag_distance` | 472 | -0.023 | -0.068 |
| `contact_count_5A` | 472 | 0.198 | 0.173 |

观察：

- `contact_count_5A` 和 target 只有弱相关。
- high-tail samples 的平均 `contact_count_5A` 高于 low-tail：`63.586` vs `49.091`。
- 这说明 interface geometry 有一点 signal，但 basic whole-interface features 不足以解释主要 error。

## 10. CDR Mapping Validation / CDR 结构映射验证

用严格规则把 IMGT CDR annotations 映射回 structure residue IDs。

| Mapping target | Safe rows | Rate within 472-row pilot |
|---|---:|---:|
| All six CDRs usable for preliminary contacts | 422 | 89.41% |
| HCDR3 usable | 468 | 99.15% |
| LCDR3 usable | 470 | 99.58% |
| HCDR3 + LCDR3 jointly usable | 467 | 98.94% |

失败原因：

| Failure reason | Rows |
|---|---:|
| Insertion/deletion ambiguity | 34 |
| Unresolved residues | 16 |
| Chain sequence mismatch | 2 |
| Missing residue numbering | 0 |
| Antigen chain mismatch | 0 |
| Missing CDR annotation | 0 |

因此 HCDR3/LCDR3 是最适合先做 contact-aware pilot 的 CDR 子集：mapping coverage 高，生物意义也强。

Preliminary one-variable correlations 很弱：

| Feature | Outcome | Rows | Pearson | Spearman |
|---|---|---:|---:|---:|
| `all_cdr_contact_count_5A` | Target affinity | 422 | 0.157 | 0.148 |
| `hcdr3_contact_count_5A` | Target affinity | 468 | 0.110 | 0.059 |
| `lcdr3_contact_count_5A` | Target affinity | 470 | 0.059 | 0.070 |
| `cdr_min_distance` | Target affinity | 422 | -0.004 | -0.028 |

这说明单个简单 contact feature 不能单独解释 binding affinity，但可以作为小型增量特征继续验证。

## 11. CDR3-Contact Residual Correction Baseline / CDR3 contact 残差修正 baseline

### 11.1 实验边界

这个实验只在 contact-covered subset 内进行：

- 不训练新的 neural network。
- 不改 dataset。
- 不处理 ambiguous chain mappings。
- 用已有 sequence model predictions。
- 在 train subset 上训练一个小的 `Ridge(alpha=1.0)` residual correction：

```text
residual = target - sequence_prediction
corrected_prediction = sequence_prediction + predicted_residual
```

这个实验不能直接和 full 1,168-row benchmark 横向比较。

### 11.2 Contact-safe subset size

| Subset | Total safe rows | Train rows used | Test rows used |
|---|---:|---:|---:|
| HCDR3+LCDR3 contact-safe | 467 | 356 | 58 |
| All-CDR contact-safe | 422 | 322 | 49 |

### 11.3 Tail-aware w2 + contact geometry

| Subset | Method | MAE | RMSE | Spearman | `pred_std / true_std` | `error_vs_true_Pearson` | Tail MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| HCDR3+LCDR3 safe | Sequence prediction | 0.9740 | 1.2857 | 0.3796 | 0.5021 | -0.8692 | 1.8701 |
| HCDR3+LCDR3 safe | + Contact Ridge correction | 0.9658 | 1.2745 | 0.3984 | 0.4969 | -0.8711 | 1.8515 |
| All-CDR safe | Sequence prediction | 0.9278 | 1.2105 | 0.2883 | 0.5837 | -0.8351 | 1.5660 |
| All-CDR safe | + Contact Ridge correction | 0.9016 | 1.1815 | 0.3423 | 0.5699 | -0.8388 | 1.5542 |

解读：

- 对 tail-aware w2，CDR3 contact correction 在两个 subset 中都小幅改善 MAE、RMSE、Spearman 和 tail MAE。
- 但是 `pred_std / true_std` 没有向 1 靠近，`error_vs_true_Pearson` 也没有向 0 靠近。
- 所以真实 CDR3 interface geometry 确实可能有增量信息，但 simple contact count + Ridge residual correction 没有解决核心 prediction compression。
- 对 unweighted baseline，contact correction 的效果不稳定，因此更不能过度宣称。

## 12. What I Learned / 我学到了什么

这个项目让我对 ML research workflow 有了更完整的理解。最重要的不是“跑出了一个分数”，而是学会了如何一步一步判断问题在哪里。

### 12.1 关于任务定义

我学到 affinity prediction 不是简单地把序列塞进模型：

- 需要明确 sample 是什么。
- 需要明确 target 是 Kd、`-log10(Kd)`，还是 ranking label。
- 需要明确 split unit，尤其 antibody-antigen 任务里 antigen leakage 很容易造成虚高结果。
- 需要区分 regression metrics 和 ranking metrics：MAE/RMSE 好，不代表 Spearman 一定好；Spearman 好，也不代表 absolute calibration 好。

### 12.2 关于 baseline thinking

我学到模型改进要有层次：

- 先做可解释的 baseline。
- 再做 CDR-aware input。
- 再做 cross-attention。
- 再做 tail-aware loss。
- 最后才进入 structure/contact feature。

这样每一步都在回答一个具体问题，而不是盲目叠复杂度。

### 12.3 关于 error analysis

我学到只看 overall MAE 很危险。这个项目最关键的 insight 来自 error analysis：

- `pred_std / true_std` 可以看 prediction range 是否被压缩。
- `error_vs_true_Pearson` 可以量化 regression-to-the-mean。
- low/mid/high target bin MAE 可以告诉我们模型到底错在哪里。
- train/val/test 同时看，能区分 overfit 和 representation/objective bottleneck。

### 12.4 关于生物表示

我学到 whole-chain sequence 不是最干净的 antibody binding 表示。CDR 是更接近 binding interface 的区域，但也不能用固定 index 硬切。标准 AbNumber + IMGT extraction 很重要，因为错误 CDR annotation 会把模型输入本身搞坏。

CDR3 loops 很重要，但 all-CDR、antigen、结构接触信息也都有可能贡献 signal。

### 12.5 关于 loss 和 objective

我学到 tail-aware loss 可以改变模型行为，但不是魔法。w2 single-seed 很好看，但 multi-seed 后发现它不能稳定赢 overall MAE/Spearman。这提醒我：

- single-seed 不能过度解读。
- checkpoint policy 很重要。
- objective weighting 可能改善 symptom，但不一定解决 root cause。

### 12.6 关于结构/contact 特征

我学到结构特征的难点不只是“有没有 PDB 文件”，更难的是：

- chain mapping 是否无歧义；
- sequence-to-structure residue mapping 是否可靠；
- CDR residue 是否能对应到有坐标的 residue；
- ambiguous chains 不能乱猜。

这部分让我更理解为什么 structure-aware modeling 要非常谨慎。

## 13. Limitations / 局限性

当前阶段还有明显局限：

1. **数据仍然偏小。** 1,168 rows 对 deep learning 来说不大，真正 tail 样本更少。
2. **label noise 和 assay heterogeneity 难以避免。** 不同实验条件、assay method、source provenance 都会影响 Kd。
3. **absolute Kd regression 很难校准。** 模型在中间区域表现较好，但极端 high/low affinity 很难。
4. **sequence-only representation 不够充分。** 即使使用 CDR 和 cross-attention，也没有真正显式建模 3D interface chemistry。
5. **contact subset 不是 full benchmark。** contact-covered subset 只有 472 条 unambiguous rows，CDR3-safe 或 all-CDR-safe subset 更小，不能直接替代 full 1,168-row benchmark。
6. **tail-aware loss 有 tradeoff。** 它改善 spread 和 tail，但 multi-seed 后不稳定提升 MAE/Spearman。
7. **contact correction 只是小型 post-hoc baseline。** Ridge residual correction 有小幅增益，但没有解决 compression。
8. **还没有外部 held-out validation。** 当前都是内部 antigen-group split，还需要更独立的数据源验证。

## 14. Final Conclusion / 最终结论

ANDD antibody v2 第一阶段已经完成了一个比较完整的闭环：

- 从 ANDD 原始数据中筛出 conservative antibody-only affinity benchmark；
- 使用 antigen-group split 控制 leakage；
- 用 AbNumber + IMGT 标准化 CDR；
- 训练和评估 pooled all-CDR、cross-attention、tail-aware loss；
- 做 target distribution、calibration、train-vs-test fit diagnosis；
- 审计 structure/contact availability；
- 验证 CDR structure mapping；
- 做小型 CDR3 contact residual correction baseline。

当前最稳妥的科学结论是：

> 模型主要问题不是输出层 bug、target transform bug、prediction clamp 或简单 split leakage，而是系统性 prediction compression。这个瓶颈很可能来自 representation 不够、tail 数据稀疏、label noise / assay heterogeneity，以及 absolute Kd regression 在极端区域本身很难校准。

Cross-attention、tail-aware loss 和 CDR3 contact geometry 都能缓解一部分症状，但都没有彻底解决 regression-to-the-mean。

这不是坏结果。相反，它说明我们已经从“能不能训练”推进到了“知道模型为什么错、知道下一步该在哪里加信息”的阶段。

## 15. Recommended Next Directions / 下一步建议

### 15.1 Richer structure/contact-aware features

下一步最值得做的是更丰富、更严格的 structure/contact-aware features：

- per-CDR contact counts
- HCDR3/LCDR3 contact fractions
- chemical contact type
- distance distribution
- interface residue features
- learnable structure-conditioned fusion

但必须保留严格 mapping 规则，不能为了覆盖率乱猜 ambiguous chain。

### 15.2 Ranking-based formulation as future work

当前结果支持未来探索 ranking-based objective，例如：

- pairwise ranking loss
- strong/weak binder ordering
- within-antigen ranking
- hybrid regression + ranking objective

但要注意措辞：目前只能说这些结果 motivate ranking-based formulation，不能说已经证明 absolute Kd regression 不可行。

### 15.3 Larger / cleaner affinity dataset

继续扩展数据是必要的，但要更重视：

- experimental vs predicted label 分开；
- assay method 分层；
- source/provenance 保留；
- unit standardization；
- antibody 和 nanobody 分开建任务；
- exact triplet duplicate 和 antigen leakage 检查。

### 15.4 Larger protein backbone ablation only after benchmark finalized

更大的 ESM backbone 可能有帮助，但不应该太早做。最好先冻结：

- dataset version；
- split policy；
- target policy；
- evaluation metrics；
- contact subset rule；
- checkpoint selection rule。

否则换大模型后很难判断提升来自 backbone，还是来自数据和评估变化。

## 16. Key Referenced Outputs and Scripts / 关键引用文件

### Data and split reports

- ANDD source audit:
  `outputs/data_expansion/ANDD_audit/ANDD_data_source_audit.md`
- Conservative antibody candidate QC:
  `outputs/data_expansion/ANDD_antibody_v2_audit/antibody_v2_quality_audit_report.md`
- Original antibody v2 split:
  `data/processed_affinity/expanded_affinity_antibody_v2/split_summary.md`
- Stratified antigen-level split:
  `data/processed_affinity/expanded_affinity_antibody_v2_stratified/split_summary.md`
- Standard CDR extraction:
  `data/processed_affinity/expanded_affinity_antibody_v2_cdr_annotated/cdr_extraction_summary.md`

### Modeling and diagnosis reports

- Original ANDD pooled baseline error analysis:
  `outputs/andd_antibody_v2/error_analysis/error_analysis_report.md`
- Target distribution diagnosis:
  `outputs/andd_antibody_v2/target_distribution/target_distribution_report.md`
- Linear calibration:
  `outputs/andd_antibody_v2/calibration/calibration_report.md`
- Stratified pooled baseline:
  `outputs/andd_antibody_v2_stratified/all_cdr_pooled/andd_antibody_v2_stratified_all_cdr_pooled_report.md`
- Stratified cross-attention:
  `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs/cross_attention_report.md`
- Train/validation/test fit diagnosis:
  `outputs/andd_antibody_v2_stratified/fit_diagnosis/fit_diagnosis_report.md`
- Tail-aware w3:
  `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_tailaware/tailaware_training_report.md`
- Tail-aware w2:
  `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_tailaware_w2/tailaware_w2_training_report.md`
- Multi-seed validation:
  `outputs/andd_antibody_v2_stratified/multiseed/multiseed_summary.md`

### Contact / structure reports

- Contact feature availability audit:
  `outputs/andd_antibody_v2_stratified/contact_feature_audit/contact_feature_audit_report.md`
- Basic interface geometry pilot:
  `outputs/andd_antibody_v2_stratified/contact_feature_audit/basic_interface_feature_report.md`
- CDR-to-structure mapping validation:
  `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr_mapping_validation_report.md`
- CDR3 contact-augmented residual baseline:
  `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr3_contact_augmented_baseline_report.md`

### Machine-readable metric tables

- Calibration metrics:
  `outputs/andd_antibody_v2/calibration/calibration_metrics.csv`
- Stratified fit metrics by model and split:
  `outputs/andd_antibody_v2_stratified/fit_diagnosis/fit_metrics_by_split.csv`
- Tail-aware multi-seed metrics:
  `outputs/andd_antibody_v2_stratified/multiseed/multiseed_summary.csv`
- Basic interface geometry features:
  `outputs/andd_antibody_v2_stratified/contact_feature_audit/basic_interface_features.csv`
- Preliminary CDR contact features:
  `outputs/andd_antibody_v2_stratified/contact_feature_audit/preliminary_cdr_contact_features.csv`
- CDR3 contact-augmented comparison metrics:
  `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr3_contact_augmented_metrics.csv`

### Figures

- Original ANDD target distribution:
  `outputs/andd_antibody_v2/target_distribution/target_distribution_histogram.png`
- Calibration true-vs-predicted:
  `outputs/andd_antibody_v2/calibration/true_vs_predicted_raw_vs_calibrated.png`
- Tail-aware w2 comparison:
  `outputs/final_reports/figures/tailaware_w2_checkpoint_comparison.png`
- Multi-seed comparison:
  `outputs/final_reports/figures/multiseed_w2_vs_baseline.png`
- Basic interface feature correlations:
  `outputs/final_reports/figures/basic_interface_feature_correlations.png`
- CDR3 contact augmented baseline:
  `outputs/final_reports/figures/cdr3_contact_augmented_baseline.png`

### Main scripts

- `scripts/audit_andd_data_source.py`
- `scripts/audit_andd_antibody_v2_candidates.py`
- `scripts/build_andd_antibody_v2_stratified_split.py`
- `scripts/analyze_andd_antibody_v2_target_distribution.py`
- `scripts/calibrate_andd_antibody_v2_predictions.py`
- `scripts/analyze_andd_stratified_model_fit.py`
- `scripts/summarize_andd_stratified_cross_attention_multiseed.py`
- `scripts/audit_andd_stratified_contact_feature_availability.py`
- `scripts/extract_andd_stratified_basic_interface_features.py`
- `scripts/validate_andd_stratified_cdr_structure_mapping.py`
- `scripts/analyze_andd_stratified_cdr3_contact_augmented_baseline.py`

## 17. 3-Minute 中文 Presentation Summary

这一阶段我主要完成了 ANDD antibody-only affinity regression benchmark 的构建和系统诊断。最开始 ANDD 有 48,800 条记录，但里面混合了 antibody、nanobody、predicted affinity、多种来源和不同质量的 label。我先把 antibody 和 nanobody 分开，排除 predicted affinity，只保留 experimental quantitative Kd-like labels，并做 sequence quality、extreme Kd、duplicate 和 overlap 审计，最后得到 1,168 条 conservative antibody-only rows。

然后我用 AbNumber + IMGT 重新做标准 CDR extraction，而不是直接使用 source-provided CDR 或固定 index slicing。这个步骤在 1,168 条上全部成功。接着我建立了 antigen-group split，保证 train、validation、test 之间 antigen sequence overlap 为 0。

第一个 ANDD all-CDR pooled ESM2 + LoRA baseline 在原始 split 上可以训练，test MAE 是 0.9066。但真正重要的发现不是 MAE，而是 prediction compression：模型的 prediction std 只有 true std 的 37.7%，而且 error 和 true target 的 Pearson 是 -0.9262。这意味着 low-affinity 样本被预测得太高，high-affinity 样本被预测得太低，也就是 regression-to-the-mean。

我排查了几个简单原因。模型最后输出是 unrestricted scalar，没有 sigmoid、tanh 或 clamp；target transform 也没有发现 inverse-transform 错误；split 中 antigen overlap 为 0。所以这不是简单代码 bug 或 leakage。为了进一步排查 split 问题，我又构建了 stratified antigen-level split，让 validation 和 test 都覆盖 P5/P95 tails，同时仍然保持 antigen overlap 为 0。但在这个 split 上，pooled all-CDR 仍然有明显 compression，说明 split 不是唯一原因。

之后我测试了 all-CDR cross-attention。它在 stratified split 上没有赢 overall MAE，pooled all-CDR 是 0.9373，cross-attention 是 0.9523。但 cross-attention 改善了 Spearman、prediction spread、error-vs-true Pearson，以及 high-target / upper-tail error。这说明 learnable CDR-antigen interaction 对 ranking 和强 binding tail 有帮助，但还没有完全解决 absolute error。

因为主要问题是 tail compression，我又做了 tail-aware loss。Single-seed 下 w2.0 看起来很好，MAE、Spearman、spread、tail MAE 都改善。但我没有停在 single-seed，而是做了 3 个 seed 的 controlled validation。结果是：tail-aware w2 稳定改善 prediction spread、降低 residual bias，并改善 combined tail MAE，尤其 lower tail；但 unweighted cross-attention 在 MAE、RMSE、Spearman 上平均更好。所以 tail-aware loss 是有价值的症状缓解方法，但不是稳定替代 baseline 的最终方案。

最后我进入 structure/contact audit。所有 1,168 条样本在外部 SAbDab structure archive 中都有结构文件，但只有 472 条有无歧义的 heavy/light/antigen chain mapping。对这 472 条，我成功提取了 basic interface geometry features。进一步做 CDR-to-structure mapping validation 后，HCDR3+LCDR3 contact-safe subset 有 467 条，all-CDR contact-safe subset 有 422 条。然后我做了一个很小的 post-hoc CDR3 contact residual correction baseline，不训练新的神经网络，只用 Ridge 学习 sequence prediction residual。结果显示，CDR3 geometry 对 tail-aware w2 有小幅 MAE、Spearman 和 tail MAE 增益，但没有改善 prediction spread，也没有解决 regression-to-the-mean。

所以最后结论是：当前瓶颈不是简单代码问题，也不是简单 split artifact。更可能是 representation 不够、tail 数据稀疏、label noise / assay heterogeneity，以及 absolute Kd regression 在极端 affinity 上很难校准。下一步最值得做的是更严格、更丰富的 structure/contact-aware features，同时可以把 ranking-based objective 作为 future work，但不能过度声称我们已经证明 absolute Kd regression 本身不可行。
