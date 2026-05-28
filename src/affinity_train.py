"""Training loop for affinity regression.

:
,

 epoch :
1.  DataLoader  batch
2.  batch  device, mps/cpu/cuda
3. forward: prediction  loss
4. backward:PyTorch 
5. optimizer.step():
6. epoch , validation set 
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

     checkpoint path,
    """

    # , DataLoader shuffledropout 
    set_seed(int(config["seed"]))

    #  outputs/checkpoints , checkpoint 
    ensure_output_dirs()

    # get_device  cuda -> mps -> cpu
    #  Apple Silicon Mac  mps
    device = get_device()
    print(f"Using device: {device}")

    # tokenizer  ESM-2  input_ids / attention_mask
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Dataset  CSV , tokenize heavy/light/antigen 
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

    # DataLoader  batch
    # train shuffle=True:,
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    # validation ,
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    #  device
    model = SeqProFTAffinityRegressor(config).to(device)

    # :
    # LoRA  trainable parameters  total parameters
    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    # AdamW  optimizer
    # optimizer.step()  loss.backward() 
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
    )

    printed_prediction_shape = False
    final_val_metrics = None

    for epoch in range(int(config["epochs"])):
        # train mode  dropout 
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Affinity epoch {epoch + 1}")

        for batch in progress_bar:
            # labels  target:neg_log10_affinity, [batch_size]
            labels = batch["labels"].to(device)

            # 
            # PyTorch , batch  zero_grad
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
                #  batch  shape  sanity check:
                # regression  [batch_size], [batch_size, num_classes]
                print(f"First batch prediction shape: {outputs['predictions'].shape}")
                printed_prediction_shape = True

            loss = outputs["loss"]

            # backward 
            loss.backward()

            # step 
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = total_loss / len(train_dataloader)

        #  epoch  validation metrics, train loss
        # train loss 
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

    # checkpoint 
    #  evaluate / inference  model_state_dict,
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
