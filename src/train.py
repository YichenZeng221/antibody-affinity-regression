"""Training loop for the SeqProFT mini reproduction.

中文说明：
这个文件负责“真正训练模型”。它会读取数据、建立 DataLoader、创建模型、
跑 epoch、打印验证集 accuracy，最后保存一个 checkpoint。
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.dataset import ProteinSequenceDataset
from src.evaluate import evaluate_accuracy
from src.model import SeqProFTMiniClassifier
from src.utils import ensure_output_dirs, get_device, set_seed


def count_trainable_parameters(model) -> tuple[int, int]:
    """Return the number of trainable parameters and total parameters.

    中文说明：
    ESM-2 本身很大。用了 LoRA 以后，我们主要训练很少一部分 adapter 参数，
    再加上最后的分类头。这个函数帮我们确认：到底有多少参数正在训练。
    """

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def train(config: dict) -> Path:
    """Train the tiny SeqProFT model and save one checkpoint.

    中文说明：
    这是主训练函数。`run_train.py` 会读取 config.yaml，然后调用这里。
    """

    set_seed(42)
    ensure_output_dirs()

    # 自动选择设备。
    # 你的 MacBook Pro 如果 PyTorch MPS 可用，这里应该打印 mps。
    device = get_device()
    print(f"Using device: {device}")

    # tokenizer 负责把氨基酸字母变成 ESM-2 能理解的 token ids。
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # 训练集：模型用它来学习。
    train_dataset = ProteinSequenceDataset(
        csv_path="data/processed/train.csv",
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
    )

    # 验证集：每个 epoch 后用它检查模型表现。
    # 验证集不参与参数更新。
    val_dataset = ProteinSequenceDataset(
        csv_path="data/processed/val.csv",
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
    )

    # DataLoader 负责把 Dataset 里的单条样本组成 batch。
    # shuffle=True 表示每个 epoch 打乱训练集顺序。
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )

    # 验证集不用打乱，因为我们只是计算整体 accuracy。
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    # 创建模型，并搬到 device 上。
    model = SeqProFTMiniClassifier(config).to(device)

    # 打印可训练参数数量。
    # 你应该看到 trainable 远小于 total，这是 LoRA 的意义。
    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    # AdamW 是 transformer 微调里常用的优化器。
    # optimizer 负责根据 loss 计算出来的梯度更新参数。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
    )

    # 只在第一个 batch 打印一次 logits shape，帮助你确认维度。
    printed_logits_shape = False

    for epoch in range(int(config["epochs"])):
        # train() 表示进入训练模式。
        # dropout 等训练行为会打开。
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            # 从 batch 里取出 input_ids / attention_mask / labels。
            # 然后搬到和模型相同的 device。
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 清空上一轮残留的梯度。
            optimizer.zero_grad()

            # 前向传播：模型根据输入算出 logits 和 loss。
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]

            # Beginner sanity check:
            # logits are the raw class scores before softmax.  For this binary
            # toy task, the shape should be [batch_size, 2].
            #
            # 中文说明：
            # logits 是模型最后输出的“原始分数”，还不是概率。
            # 这里是二分类，所以每条序列会有 2 个分数。
            # 如果 batch_size=2，那么第一批的形状应该是 torch.Size([2, 2])。
            if not printed_logits_shape:
                print(f"First batch logits shape: {outputs['logits'].shape}")
                printed_logits_shape = True

            # 反向传播：根据 loss 计算每个可训练参数的梯度。
            loss.backward()

            # 参数更新：optimizer 根据梯度更新 LoRA 和分类头参数。
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # 一个 epoch 结束后，计算平均训练 loss。
        average_loss = total_loss / len(train_dataloader)

        # 在验证集上计算 accuracy。
        val_accuracy = evaluate_accuracy(model, val_dataloader, device)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={average_loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}"
        )

    # 训练完成后，保存 checkpoint。
    # checkpoint 里包含模型参数和 config。
    checkpoint_path = Path("outputs/checkpoints/seqproft_mvp.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path
