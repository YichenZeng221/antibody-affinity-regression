# ANDD Antibody v2 Affinity Regression: Four-Figure Presentation Summary
# ANDD Antibody v2 亲和力回归：四图汇报总结

## 1. Project Goal / 项目目标

**English**

This project studies antibody-antigen binding affinity prediction on the ANDD antibody v2 benchmark. The target is `-log10(Kd)`, where a higher value means stronger binding. The model input is based on antibody CDR sequences and antigen sequence. The main goal of this stage was not only to improve metrics, but to diagnose why sequence-based models struggle with affinity extremes.

**中文**

这个项目研究的是 ANDD antibody v2 数据集上的抗体-抗原结合亲和力预测。预测目标是 `-log10(Kd)`，数值越高表示结合越强。模型输入主要是抗体 CDR 序列和 antigen sequence。本阶段的目标不只是提高指标，而是诊断为什么 sequence-based 模型在亲和力极端样本上表现困难。

## 2. What I Did / 我做了什么

**English**

I built a conservative antibody-only benchmark from ANDD, used an antigen-level stratified split, extracted standard CDRs with AbNumber/IMGT, and trained sequence-based baselines including all-CDR pooled models and all-CDR cross-attention models. I then diagnosed prediction compression, tested tail-aware loss with multi-seed validation, audited structure/contact feature availability, and tested a small CDR3 contact residual-correction baseline on contact-covered subsets.

**中文**

我从 ANDD 中构建了一个保守的 antibody-only benchmark，使用 antigen-level stratified split，基于 AbNumber/IMGT 提取标准 CDR，并训练了 all-CDR pooled 和 all-CDR cross-attention 等 sequence-based baseline。之后我诊断了 prediction compression，做了 tail-aware loss 的 multi-seed 验证，审计了 structure/contact feature 的可用性，并在 contact-covered subset 上测试了 CDR3 contact residual correction baseline。

## 3. Figure 1: Prediction Compression Across Splits / 图 1：不同 split 上的预测压缩

Figure: `outputs/final_reports/figures/final_fig1_prediction_compression_across_splits.png`

**English**

Figure 1 plots true vs predicted `-log10(Kd)` for train, validation, and test using the all-CDR cross-attention model. The dashed diagonal is the ideal `y=x` line, while the red line is the fitted trend. If the model were well calibrated across the full target range, the fitted trend would be close to the diagonal.

The fitted trend is much flatter than `y=x` on all splits. The train split also shows compression, so the problem is not classic overfitting where train predictions are good but validation/test predictions fail.

Key values:

| Split | pred std / true std | trend slope |
|---|---:|---:|
| Train | 0.458 | 0.25 |
| Validation | 0.383 | 0.13 |
| Test | 0.392 | 0.13 |

**中文**

图 1 展示了 all-CDR cross-attention 模型在 train、validation 和 test 上的 true vs predicted `-log10(Kd)`。虚线是理想的 `y=x`，红线是拟合趋势线。如果模型能很好覆盖整个 target range，趋势线应该接近对角线。

但实际趋势线明显比 `y=x` 平，说明预测范围被压缩。更重要的是，train split 上也存在同样现象，所以这不像典型 overfitting，也就是不是“训练集很好、验证/测试集很差”的情况。

关键数值：

| Split | pred std / true std | trend slope |
|---|---:|---:|
| Train | 0.458 | 0.25 |
| Validation | 0.383 | 0.13 |
| Test | 0.392 | 0.13 |

## 4. Figure 2: Residual Trend Shows Regression-to-the-Mean / 图 2：残差趋势显示向均值回归

Figure: `outputs/final_reports/figures/final_fig2_residual_trend_regression_to_mean.png`

**English**

Figure 2 plots residual vs true target, where residual is `prediction - true`. A downward residual trend means that low target values are overpredicted and high target values are underpredicted.

This is exactly what appears across train, validation, and test. The error is systematic rather than random: the model pulls predictions toward the middle of the target distribution.

Key values:

| Split | residual trend slope |
|---|---:|
| Train | -0.75 |
| Validation | -0.87 |
| Test | -0.87 |

**中文**

图 2 展示的是 residual vs true target，其中 residual 定义为 `prediction - true`。如果 residual trend 是向下的，说明低 target 被高估，高 target 被低估。

这正是 train、validation、test 上都出现的现象。也就是说，误差不是随机噪声，而是系统性地把预测拉向 target distribution 的中间区域。

关键数值：

| Split | residual trend slope |
|---|---:|
| Train | -0.75 |
| Validation | -0.87 |
| Test | -0.87 |

## 5. Figure 3: Multi-Seed Validation of Tail-Aware W2 / 图 3：Tail-aware W2 多 seed 验证

Figure: `outputs/final_reports/figures/final_fig3_multiseed_tailaware_w2.png`

**English**

Figure 3 compares unweighted all-CDR cross-attention with tail-aware w2 cross-attention under three random seeds. The checkpoint policy shown here is best validation tail MAE.

Tail-aware w2 improves prediction spread and tail MAE on average, but unweighted cross-attention remains better on overall MAE and Spearman. This means tail-aware loss helps the symptom, especially tail behavior, but it is not a universal fix.

Key values:

| Metric | Unweighted cross-attention | Tail-aware w2 |
|---|---:|---:|
| MAE | 0.922 | 0.942 |
| Spearman | 0.475 | 0.455 |
| pred std / true std | 0.511 | 0.593 |
| P10/P90 Tail MAE | 1.715 | 1.649 |

**中文**

图 3 比较了 unweighted all-CDR cross-attention 和 tail-aware w2 cross-attention 的三 seed 结果。这里展示的是 best validation tail MAE checkpoint policy。

Tail-aware w2 平均上改善了 prediction spread 和 tail MAE，但 unweighted cross-attention 在整体 MAE 和 Spearman 上仍然更好。这说明 tail-aware loss 可以缓解部分症状，尤其是 tail behavior，但不是一个全面解决方案。

关键数值：

| Metric | Unweighted cross-attention | Tail-aware w2 |
|---|---:|---:|
| MAE | 0.922 | 0.942 |
| Spearman | 0.475 | 0.455 |
| pred std / true std | 0.511 | 0.593 |
| P10/P90 Tail MAE | 1.715 | 1.649 |

## 6. Figure 4: CDR3 Contact Augmentation / 图 4：CDR3 接触特征增量实验

Figure: `outputs/final_reports/figures/final_fig4_cdr3_contact_augmentation.png`

**English**

Figure 4 tests whether simple CDR3 contact geometry features add value on contact-covered subsets. This is not the full 1,168-row benchmark. It is a subset analysis only, because reliable structure/contact mapping is available for only part of the data.

The method uses sequence model predictions as the baseline, then adds a Ridge residual correction using CDR3 contact features. The result shows small gains in MAE, Spearman, and tail MAE in the tail-aware w2 setting. However, pred std / true std does not improve, so simple scalar contact features do not solve prediction compression.

HCDR3+LCDR3 safe subset:

| Metric | Sequence only | Sequence + CDR3 contact |
|---|---:|---:|
| MAE | 0.974 | 0.966 |
| Spearman | 0.380 | 0.398 |
| Tail MAE | 1.870 | 1.851 |
| pred std / true std | 0.502 | 0.497 |

All-CDR safe subset:

| Metric | Sequence only | Sequence + CDR3 contact |
|---|---:|---:|
| MAE | 0.928 | 0.902 |
| Spearman | 0.288 | 0.342 |
| Tail MAE | 1.566 | 1.554 |
| pred std / true std | 0.584 | 0.570 |

Important boundary: this is contact-covered subset analysis only and is not directly comparable to the full 1,168-row benchmark.

**中文**

图 4 测试的是简单 CDR3 contact geometry features 是否能在 contact-covered subset 上提供增量价值。这里不是完整的 1,168-row benchmark，而只是 subset analysis，因为只有部分样本有可靠的 structure/contact mapping。

方法是先使用 sequence model prediction 作为 baseline，然后用 CDR3 contact features 通过 Ridge residual correction 修正残差。结果显示，在 tail-aware w2 setting 下，MAE、Spearman 和 tail MAE 有小幅改善。但是 pred std / true std 没有改善，所以简单 scalar contact features 仍然没有解决 prediction compression。

HCDR3+LCDR3 safe subset:

| Metric | Sequence only | Sequence + CDR3 contact |
|---|---:|---:|
| MAE | 0.974 | 0.966 |
| Spearman | 0.380 | 0.398 |
| Tail MAE | 1.870 | 1.851 |
| pred std / true std | 0.502 | 0.497 |

All-CDR safe subset:

| Metric | Sequence only | Sequence + CDR3 contact |
|---|---:|---:|
| MAE | 0.928 | 0.902 |
| Spearman | 0.288 | 0.342 |
| Tail MAE | 1.566 | 1.554 |
| pred std / true std | 0.584 | 0.570 |

重要边界：这是 contact-covered subset analysis，不能直接和完整的 1,168-row benchmark 横向比较。

## 7. What AbRank Helped Me Realize / AbRank 给我的启发

**English**

AbRank helped me think about the task framing. Instead of treating antibody-antigen affinity prediction only as exact absolute Kd regression, AbRank frames the problem more as ranking or metric learning: which antibody is more likely to bind better?

That framing is close to many real screening scenarios, where the practical goal is to prioritize better candidates rather than perfectly calibrate absolute Kd for every pair. My results do not prove that absolute Kd regression is invalid. Absolute regression is still a legitimate benchmark. But the results are consistent with the motivation behind ranking-based formulations such as AbRank: calibrated absolute affinity prediction is difficult under noisy labels, heterogeneous assays, limited tail data, and imperfect sequence/structure representation.

**中文**

AbRank 给我的主要启发是 task framing。它不是只把抗体-抗原亲和力预测当成精确的 absolute Kd regression，而是更偏向 ranking 或 metric learning：哪个 antibody 更可能结合得更好？

这个 framing 更接近很多真实 screening 场景，因为实际目标往往是优先筛出更强 binder，而不是对每一对 antibody-antigen 都精确校准绝对 Kd。我的结果不能证明 absolute Kd regression 不成立；它仍然是一个合法 benchmark。更准确地说，我的结果和 AbRank 这类 ranking-based formulation 的动机是一致的：在 label noisy、assay heterogeneous、tail data 有限、sequence/structure representation 不完美的情况下，精确校准 absolute affinity 很难。

## 8. What I Learned / 我学到了什么

**English**

1. Biomedical ML starts with defining `X`, `y`, split unit, leakage risk, and metrics.
2. Train/validation/test diagnostics can distinguish overfitting from representation, objective, or data bottlenecks.
3. Single-seed results can be misleading; multi-seed validation makes conclusions more reliable.
4. Error analysis is more important than blindly changing model architecture.
5. Structure/contact features require a mapping audit before modeling.
6. Task framing matters: absolute regression, ranking, calibration, and screening prioritization are related but different problems.

**中文**

1. Biomedical ML 首先要定义清楚 `X`、`y`、split unit、leakage risk 和 metrics。
2. Train/validation/test 诊断可以帮助区分 overfitting 和 representation/objective/data bottleneck。
3. Single-seed 结果可能误导结论；multi-seed validation 会让结论更可靠。
4. Error analysis 比盲目换模型结构更重要。
5. Structure/contact features 在建模前必须先做 mapping availability audit。
6. Task framing 很重要：absolute regression、ranking、calibration 和 screening prioritization 是相关但不同的问题。

## 9. Final Takeaway / 最终结论

**English**

The main conclusion is not that the model simply failed, but that absolute affinity regression under this dataset shows systematic prediction compression. Tail-aware loss and CDR3 contact features provide partial improvements, but neither solves the core issue. This experience suggests that antibody affinity modeling should carefully consider representation, label quality, and task framing. AbRank's ranking-based formulation is a useful reference because it targets candidate prioritization rather than exact calibrated Kd prediction.

**中文**

最终结论不是“模型失败了”，而是这个数据和任务设定下，absolute affinity regression 出现了系统性预测压缩。Tail-aware loss 和 CDR3 contact features 都有部分帮助，但都没有根治。这个项目让我意识到，抗体亲和力建模不能只看模型结构，还要看数据质量、表示方式和任务定义。AbRank 的 ranking framing 给我的启发是：如果真实目标是筛选更强 binder，排序问题可能比精确预测绝对 Kd 更实用。

## 10. 3-Minute English Talk Track

This project focuses on antibody-antigen affinity prediction using the ANDD antibody v2 benchmark. The target is `-log10(Kd)`, so higher values mean stronger binding. I built a conservative antibody-only benchmark, used antigen-level splitting, extracted standard CDRs with AbNumber and IMGT, and tested sequence-based models including all-CDR pooled and all-CDR cross-attention models.

The key issue I found is prediction compression. In Figure 1, true vs predicted plots show that the fitted prediction trend is much flatter than the ideal `y=x` line. This happens not only on validation and test, but also on the train split. The train pred std / true std is 0.458, and validation and test are 0.383 and 0.392. That means the model is not simply overfitting; even on training data, predictions are pulled toward the middle of the target range.

Figure 2 confirms this with residuals. The residual trend is negative across all splits: about -0.75 on train and -0.87 on validation and test. This means low target values are overpredicted and high target values are underpredicted. So the error pattern is systematic regression-to-the-mean, not random noise.

I then tested whether tail-aware training can help. Figure 3 compares unweighted cross-attention with tail-aware w2 over three seeds. Tail-aware w2 improves pred std / true std from 0.511 to 0.593 and improves P10/P90 tail MAE from 1.715 to 1.649. However, unweighted cross-attention remains better on overall MAE and Spearman. So tail-aware loss helps the symptom, but it is not a universal fix.

Finally, I tested whether real CDR3 contact features add information. Figure 4 shows a contact-covered subset analysis only. Adding CDR3 contact residual correction gives small improvements in MAE, Spearman, and tail MAE, especially in the all-CDR safe subset, where MAE improves from 0.928 to 0.902 and Spearman improves from 0.288 to 0.342. But pred std / true std does not improve, so simple contact-count features do not solve compression.

The broader lesson for me is task framing. AbRank helped me realize that antibody affinity modeling may be better framed as ranking or metric learning when the real goal is candidate prioritization. My results do not prove absolute Kd regression is invalid, but they are consistent with AbRank's motivation: exact calibrated affinity prediction is difficult with noisy labels, heterogeneous assays, limited tail data, and incomplete biological representation.

## 11. 3-Minute 中文汇报稿

这个项目研究的是 ANDD antibody v2 benchmark 上的抗体-抗原亲和力预测。预测目标是 `-log10(Kd)`，数值越高表示结合越强。我构建了一个保守的 antibody-only benchmark，使用 antigen-level split，通过 AbNumber 和 IMGT 提取标准 CDR，并测试了 all-CDR pooled 和 all-CDR cross-attention 等 sequence-based 模型。

我发现的核心问题是 prediction compression。图 1 的 true vs predicted scatter 显示，模型拟合出来的趋势线明显比理想的 `y=x` 平。这个现象不仅出现在 validation 和 test，也出现在 train split。train 的 pred std / true std 是 0.458，validation 和 test 分别是 0.383 和 0.392。这说明模型不是简单 overfitting，因为即使在训练集上，预测也被拉向 target range 的中间。

图 2 从 residual 角度进一步确认了这个问题。三个 split 上 residual trend 都是负的：train 约为 -0.75，validation 和 test 约为 -0.87。这意味着 low target 被高估，high target 被低估。所以误差不是随机的，而是系统性的 regression-to-the-mean。

接着我测试了 tail-aware training 是否能缓解这个问题。图 3 比较了 unweighted cross-attention 和 tail-aware w2 的三 seed 结果。Tail-aware w2 把 pred std / true std 从 0.511 提高到 0.593，也把 P10/P90 tail MAE 从 1.715 降到 1.649。但是 unweighted cross-attention 在整体 MAE 和 Spearman 上仍然更好。所以 tail-aware loss 能缓解 tail behavior，但不是全面解决方案。

最后我测试了真实 CDR3 contact features 是否能提供增量信息。图 4 是 contact-covered subset analysis，不是完整 1,168-row benchmark。加入 CDR3 contact residual correction 后，MAE、Spearman 和 tail MAE 有小幅改善。比如 all-CDR safe subset 上，MAE 从 0.928 降到 0.902，Spearman 从 0.288 提高到 0.342。但是 pred std / true std 没有改善，所以简单 contact count 类特征仍然没有解决 prediction compression。

这个项目给我的最大启发是 task framing。AbRank 让我意识到，抗体亲和力建模不一定只能被定义成精确的 absolute Kd regression。如果真实目标是筛选更强 binder，那么 ranking 或 metric learning 可能更贴近实际需求。我的结果不能证明 absolute Kd regression 不成立，但它们和 AbRank 的动机是一致的：在 label noisy、assay heterogeneous、tail data 有限、representation 不完整的情况下，精确校准 absolute affinity 是很难的。

## Referenced Files / 引用文件

- `outputs/final_reports/figures/final_fig1_prediction_compression_across_splits.png`
- `outputs/final_reports/figures/final_fig2_residual_trend_regression_to_mean.png`
- `outputs/final_reports/figures/final_fig3_multiseed_tailaware_w2.png`
- `outputs/final_reports/figures/final_fig4_cdr3_contact_augmentation.png`
- `outputs/andd_antibody_v2_stratified/fit_diagnosis/fit_metrics_by_split.csv`
- `outputs/andd_antibody_v2_stratified/multiseed/multiseed_summary.csv`
- `outputs/andd_antibody_v2_stratified/contact_feature_audit/cdr3_contact_augmented_metrics.csv`
