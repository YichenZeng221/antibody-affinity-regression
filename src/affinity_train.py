"""Training loop for affinity regression.

中文人话说明：
这个文件负责“怎么训练模型”，不负责定义模型结构。

一个训练 epoch 大概做这些事：
1. 从 DataLoader 取一个 batch。
2. 把 batch 放到 device，比如 mps/cpu/cuda。
3. forward：模型算 prediction 和 loss。
4. backward：PyTorch 计算梯度。
5. optimizer.step()：根据梯度更新可训练参数。
6. epoch 结束后，在 validation set 上看指标。
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.affinity_dataset import AffinityRegressionDataset
from src.affinity_evaluate import evaluate_affinity_model
from src.affinity_model import SeqProFTAffinityRegressor
from src.train import count_trainable_parameters
from src.utils import ensure_output_dirs, get_device, set_seed


def train_affinity(config: dict) -> Path:
    """Train the Stage 1 affinity regression model.

    返回 checkpoint path，方便入口脚本或后续流程知道模型存在哪里。
    """

    # 固定随机种子，让 DataLoader shuffle、dropout 等随机行为更可复现。
    set_seed(int(config["seed"]))

    # 确保 outputs/checkpoints 等目录存在，不然后面保存 checkpoint 会报错。
    ensure_output_dirs()

    # get_device 会自动选择 cuda -> mps -> cpu。
    # 你的 Apple Silicon Mac 正常情况下会使用 mps。
    device = get_device()
    print(f"Using device: {device}")

    # tokenizer 把蛋白质序列字符串变成 ESM-2 能读的 input_ids / attention_mask。
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Dataset 负责从 CSV 读样本，并 tokenize heavy/light/antigen 三条序列。
    train_dataset = AffinityRegressionDataset(
        csv_path=config["train_csv"],
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        target_column=config["target_column"],
    )
    val_dataset = AffinityRegressionDataset(
        csv_path=config["val_csv"],
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        target_column=config["target_column"],
    )

    # DataLoader 负责把一个个样本组成 batch。
    # train shuffle=True：训练时打乱样本顺序，减少模型记住固定顺序的机会。
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    # validation 不需要打乱，因为只是评估。
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    # 创建模型并移动到 device。
    model = SeqProFTAffinityRegressor(config).to(device)

    # 这里会显示：到底有多少参数真的在训练。
    # LoRA 的目的就是让 trainable parameters 远小于 total parameters。
    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    # AdamW 是深度学习里常用 optimizer。
    # optimizer.step() 会根据 loss.backward() 算出来的梯度更新参数。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
    )

    printed_prediction_shape = False
    final_val_metrics = None

    for epoch in range(int(config["epochs"])):
        # train mode 会启用 dropout 等训练行为。
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Affinity epoch {epoch + 1}")

        for batch in progress_bar:
            # labels 是真实 target：neg_log10_affinity，形状是 [batch_size]。
            labels = batch["labels"].to(device)

            # 清空上一轮的梯度。
            # PyTorch 默认会累积梯度，所以每个 batch 前都要 zero_grad。
            optimizer.zero_grad()

            outputs = model(
                heavy_input_ids=batch["heavy_input_ids"].to(device),
                heavy_attention_mask=batch["heavy_attention_mask"].to(device),
                light_input_ids=batch["light_input_ids"].to(device),
                light_attention_mask=batch["light_attention_mask"].to(device),
                antigen_input_ids=batch["antigen_input_ids"].to(device),
                antigen_attention_mask=batch["antigen_attention_mask"].to(device),
                labels=labels,
            )

            if not printed_prediction_shape:
                # 第一个 batch 打印 shape 是 sanity check：
                # regression 这里应该是 [batch_size]，不是 [batch_size, num_classes]。
                print(f"First batch prediction shape: {outputs['predictions'].shape}")
                printed_prediction_shape = True

            loss = outputs["loss"]

            # backward 计算每个可训练参数应该怎么改。
            loss.backward()

            # step 真正更新参数。
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = total_loss / len(train_dataloader)

        # 每个 epoch 后看 validation metrics，避免只盯着 train loss。
        # train loss 下降不代表模型一定能泛化到没见过的数据。
        val_metrics, _, _ = evaluate_affinity_model(model, val_dataloader, device)
        final_val_metrics = val_metrics

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={average_loss:.4f}, "
            f"val_MAE={val_metrics['mae']:.4f} log10 units, "
            f"val_MSE={val_metrics['mse']:.4f}, "
            f"val_RMSE={val_metrics['rmse']:.4f} log10 units, "
            f"val_Spearman={val_metrics['spearman']:.4f}, "
            f"val_MAE_fold_error~{val_metrics['approx_mae_fold_error']:.1f}x, "
            f"val_RMSE_fold_error~{val_metrics['approx_rmse_fold_error']:.1f}x"
        )

    checkpoint_path = Path(config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # checkpoint 是训练结果的存档。
    # 之后 evaluate / inference 可以加载 model_state_dict，不用重新训练。
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "final_val_metrics": final_val_metrics,
        },
        checkpoint_path,
    )

    print(f"Saved affinity checkpoint to {checkpoint_path}")
    return checkpoint_path
