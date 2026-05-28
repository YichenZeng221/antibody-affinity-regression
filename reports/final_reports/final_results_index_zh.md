# 最终结果索引

## 1. 重要报告路径

### 最终项目总结

| 报告 | 路径 |
|---|---|
| 英文最终项目总结 | `outputs/final_reports/unified_no_high_risk_project_summary.md` |
| 中文最终项目总结 | `outputs/final_reports/unified_no_high_risk_project_summary_zh.md` |
| 英文 Week 1 总结 | `outputs/final_reports/week1_project_closure_summary.md` |
| 中文 Week 1 总结 | `outputs/final_reports/week1_project_closure_summary_zh.md` |
| 英文结果索引 | `outputs/final_reports/final_results_index.md` |
| 中文结果索引 | `outputs/final_reports/final_results_index_zh.md` |

### 主要建模报告

| 报告 | 路径 |
|---|---|
| Whole-sequence error analysis | `outputs/error_analysis/unified_no_high_risk/error_analysis_report.md` |
| All-CDR pooled baseline | `outputs/cdr_aware/unified_no_high_risk/cdr_aware_report.md` |
| CDR ablation 汇总 | `outputs/cdr_ablation/unified_no_high_risk/cdr_ablation_summary.md` |
| Simple interaction matrix report | `outputs/interaction_aware/unified_no_high_risk/hcdr3_lcdr3_antigen/interaction_report.md` |
| All-CDR cross-attention report | `outputs/cross_attention/unified_no_high_risk/all_cdrs_antigen/cross_attention_report.md` |
| SeqProFT official GitHub 对照报告 | `outputs/reproducibility/seqproft_github_comparison.md` |

### ANDD 数据扩展与最新诊断报告

| 报告 | 路径 |
|---|---|
| ANDD antibody v2 error analysis | `outputs/andd_antibody_v2/error_analysis/error_analysis_report.md` |
| ANDD target distribution diagnosis | `outputs/andd_antibody_v2/target_distribution/target_distribution_report.md` |
| ANDD all-CDR pooled linear calibration | `outputs/andd_antibody_v2/calibration/calibration_report.md` |
| ANDD stratified antigen-level split summary | `data/processed_affinity/expanded_affinity_antibody_v2_stratified/split_summary.md` |
| ANDD stratified all-CDR pooled baseline | `outputs/andd_antibody_v2_stratified/all_cdr_pooled/andd_antibody_v2_stratified_all_cdr_pooled_report.md` |
| ANDD stratified all-CDR cross-attention | `outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs/cross_attention_report.md` |

## 2. `unified_no_high_risk` 主要模型指标

以下指标来自原始 `unified_no_high_risk` benchmark，使用 antigen-sequence group split。

| 模型 | 输入 | MAE | RMSE | Spearman | pred_std / true_std | error vs true Pearson | high-target MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| Whole-sequence baseline | heavy + light + antigen | 1.1083 | 1.3765 | 0.4557 | 0.4825 | -0.8787 | 1.7949 |
| Pooled all-CDR baseline | HCDR1/2/3 + LCDR1/2/3 + antigen | 0.9975 | 1.2495 | 0.4497 | 0.3651 | -0.9321 | 1.6401 |
| Pooled HCDR3+LCDR3 | HCDR3 + LCDR3 + antigen | 1.0204 | 1.2570 | 0.4438 | 0.3383 | -0.9418 | 1.5780 |
| Pooled heavy CDRs | HCDR1/2/3 + antigen | 1.0462 | 1.3353 | 0.4281 | 0.3251 | -0.9471 | 1.9162 |
| Pooled light CDRs | LCDR1/2/3 + antigen | 1.0554 | 1.3482 | 0.3799 | 0.3538 | -0.9353 | 1.8796 |
| Pooled HCDR3-only | HCDR3 + antigen | 1.0641 | 1.3308 | 0.4176 | 0.4611 | -0.8876 | 1.7880 |
| Simple interaction matrix | CDR3-antigen dot-product summaries | 1.0504 | 1.2839 | 0.4126 | 0.2958 | -0.9560 | 1.6657 |
| All-CDR cross-attention | 六条 CDR query -> antigen key/value | 1.0515 | 1.3156 | 0.5018 | 0.7855 | -0.6617 | 1.4740 |

## 3. 当前最佳模型结论

### Overall MAE / RMSE 最佳

当前总体绝对误差最好的模型是：

```text
Pooled all-CDR baseline
```

| 指标 | 数值 |
|---|---:|
| MAE | 0.9975 |
| RMSE | 1.2495 |
| Spearman | 0.4497 |

解释：标准 IMGT CDR-focused input 比 whole-chain pooled input 更适合降低总体预测误差；如果目标优先是 MAE/RMSE，当前应以 pooled all-CDR 为主参考模型。

### Spearman 最佳

当前排序能力最好的模型是：

```text
All-CDR cross-attention
```

| 指标 | 数值 |
|---|---:|
| Spearman | 0.5018 |
| MAE | 1.0515 |
| RMSE | 1.3156 |

解释：learnable CDR-to-antigen interaction 有助于排序，但还没有同时改善整体 MAE/RMSE。

### 最能缓解 Regression-To-Mean

当前最明显缓解预测范围压缩的模型也是：

```text
All-CDR cross-attention
```

| 诊断指标 | 数值 | 含义 |
|---|---:|---|
| pred_std / true_std | 0.7855 | 越接近 1，prediction spread 越接近真实 target spread。 |
| error vs true Pearson | -0.6617 | 越接近 0，低值高估、高值低估的系统偏差越弱。 |
| high-target MAE | 1.4740 | 当前主要模型中 high-target MAE 最好。 |

## 4. ANDD Antibody v2：原始 Split 诊断与 Calibration

ANDD antibody-only v2 是独立的新 benchmark，不能把它的 test metrics 与原始 605-row benchmark 当作同一个 test set 直接比较。

### 原始 ANDD All-CDR Pooled Baseline

| 模型 / 处理 | MAE | RMSE | Spearman | pred_std / true_std | error vs true Pearson | low-target MAE | high-target MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw all-CDR pooled | 0.9066 | 1.1281 | 0.3817 | 0.3772 | -0.9262 | 1.2603 | 1.0981 |
| Post-hoc linear calibration | 0.9015 | 1.1424 | 0.3817 | 0.5302 | -0.8508 | 1.0173 | 1.2347 |

linear calibration 只在 validation predictions 上拟合，然后应用到 test。它略微改善了 MAE、prediction spread、regression-to-the-mean 指标和 low-target MAE，但 RMSE 与 high-target MAE 变差。因此 calibration 有帮助，但不是完整解决方案。

### Target Distribution Diagnosis

- Train 的 low/mid/high tertiles 按定义近似均衡：`312 / 311 / 311`。
- 所以 regression-to-the-mean 不能简单归因于 train 中粗粒度 low/mid/high 数量不均。
- 真正极端的 tails 仍然样本有限：以 train P10/P90 定义时，每一端约占训练数据的 10%。
- 原始 validation 并非完全没有 high-affinity tail：按 global P95 口径有 `4` 条；但覆盖较弱，最大 target 仅为 `6.9811`，而新的 stratified validation 达到 `11.8155`。

## 5. ANDD Stratified Antigen-Level Split

新的 stratified split 目标是让 tail evaluation 更稳定，同时继续严格控制 antigen leakage。

| Split | Rows | Antigen groups | Target min | Target max | Global P5 tail rows | Global P95 tail rows |
|---|---:|---:|---:|---:|---:|---:|
| train | 934 | 568 | 2.0449 | 8.5343 | 42 | 43 |
| val | 117 | 117 | 2.0031 | 11.8155 | 10 | 9 |
| test | 117 | 117 | 2.0349 | 10.9208 | 7 | 7 |

- Global target P5/P95 thresholds：`2.8900 / 6.5838`。
- Validation 与 test 都明确覆盖低端和高端 target tails。
- train/val、train/test、val/test 的 `antigen_sequence` overlap 均为 `0`。

### 同一 Stratified Split 下的公平比较

| 模型 | MAE | RMSE | Spearman | pred_std / true_std | error vs true Pearson | low-target MAE | high-target MAE | below-P10 MAE | above-P90 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All-CDR pooled | 0.9373 | 1.3056 | 0.3699 | 0.3170 | -0.9484 | 1.1674 | 1.3367 | 1.8650 | 2.0887 |
| All-CDR cross-attention | 0.9523 | 1.3008 | 0.3861 | 0.3925 | -0.9202 | 1.4182 | 1.1147 | 2.0273 | 1.8483 |

解释：

- 在同一个 stratified test set 上，pooled all-CDR 的 overall MAE 略好。
- Cross-attention 略微改善 RMSE 与 Spearman。
- Cross-attention 提高 prediction spread，并使 `error vs true Pearson` 没有那么负，说明 regression-to-the-mean 有所缓解。
- Cross-attention 改善 high-target 和 upper-tail MAE，但 low-target 与 lower-tail MAE 变差。
- 即使 split 明确覆盖 tails，pooled 模型仍有明显 prediction compression。因此 regression-to-the-mean 不是单纯由 split 造成的。

## 6. 最终诚实结论

1. 在原始 `unified_no_high_risk` benchmark 上，pooled all-CDR 仍然是 overall MAE/RMSE 最好的模型。
2. HCDR3+LCDR3 pooled 很接近 all-CDR，说明 CDR3 loops 承载了大量 affinity signal。
3. 简单 dot-product interaction summary 没有带来提升，说明交互信息不能过早被压缩成粗略统计量。
4. All-CDR cross-attention 提高了 Spearman、prediction spread 与高端 affinity 表现，证明 learnable interaction 值得继续探索。
5. ANDD 扩展数据可用，但必须保留 antibody-only、experimental-label-only 与 antigen-group leakage control 的保守设计。
6. Calibration 能部分拉开 prediction range，但在 low/high tail 之间存在 trade-off。
7. Stratified antigen-level split 让 tail evaluation 更可信，但没有消除 regression-to-the-mean。
8. 当前瓶颈不是 regression head 输出范围 bug，也不只是 split artifact，而是 interaction representation、tail calibration 与 label/data complexity 的综合问题。

## 7. 下一阶段建议

1. 在 ANDD stratified benchmark 上继续优化 learnable cross-attention 或 interaction-aware representation。
2. 将 calibration、tail-aware loss 与 checkpoint selection 结合起来，并分别监控 low/high tails。
3. 对 weighted loss / sampling 方法只在清晰的 validation policy 下比较，避免为一个 tail 改善而牺牲另一端。
4. 如果 sequence-only interaction 的收益到顶，进一步接入 structure/contact-aware features 或 interface supervision。
5. 继续保持 antibody 与 nanobody 分任务、experimental 与 predicted label 分层的审计原则。

## 8. 简短会议更新

> 这一阶段我把项目从原始的 605-row curated benchmark 扩展到了保守筛选的 ANDD antibody-only benchmark，并专门检查了模型为什么会向均值收缩。原始 ANDD validation 对 high-affinity tail 的覆盖较弱，因此我新建了 antigen-level stratified split，使 validation/test 都覆盖 P5/P95 tails，同时保持 antigen overlap 为 0。在同一 stratified test set 上，pooled all-CDR 的 overall MAE 略好，而 cross-attention 提高了 Spearman、prediction spread 和 upper-tail 表现。结合 linear calibration 的结果，现在可以比较明确地说：regression-to-the-mean 不是代码 bug，也不只是 split 问题，下一步应该继续做 learnable interaction 与 tail-aware calibration/training，并考虑 structure/contact-aware information。
