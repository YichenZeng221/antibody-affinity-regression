"""评估代码：训练过程中用它来计算验证集 accuracy。

人话解释：
训练时我们不仅要看 loss，还要看模型在验证集上答对了多少。
这里先只实现最简单的分类准确率 accuracy。
"""

import torch


def evaluate_accuracy(model, dataloader, device) -> float:
    """计算简单分类准确率。

    人话解释：
    accuracy = 答对的样本数 / 总样本数

    例如验证集有 4 条，模型答对 3 条：
    accuracy = 3 / 4 = 0.75

    这个 MVP 先用 accuracy 就够了。
    后面做真实任务时，可以再加 F1、AUROC 等指标。
    """

    # eval() 会关闭 dropout 等训练专用行为。
    # 评估时我们希望模型稳定输出。
    model.eval()
    # 记录答对多少、总共多少
    correct = 0
    total = 0

    # no_grad 表示：这里只做评估，不需要计算梯度。
    # 好处：更快、更省内存。
    with torch.no_grad():
        for batch in dataloader:
            # 把数据搬到同一个设备上。
            # 如果模型在 mps，数据也要在 mps；如果模型在 cpu，数据也要在 cpu。
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播：拿到模型输出。
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # logits 是每个类别的原始分数。
            # argmax 取分数最高的类别，作为预测结果。
            predictions = outputs["logits"].argmax(dim=-1)

            # 统计答对了多少条。
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # 防御性写法：如果 dataloader 里没有数据，就返回 0。
    if total == 0:
        return 0.0

    return correct / total
