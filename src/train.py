"""Training loop for the SeqProFT mini reproduction.

:
 DataLoader
 epoch accuracy, checkpoint
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

    :
    ESM-2  LoRA , adapter ,
    :
    """

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def train(config: dict) -> Path:
    """Train the tiny SeqProFT model and save one checkpoint.

    :
    `run_train.py`  config.yaml,
    """

    set_seed(42)
    ensure_output_dirs()

    # 
    #  MacBook Pro  PyTorch MPS , mps
    device = get_device()
    print(f"Using device: {device}")

    # tokenizer  ESM-2  token ids
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # :
    train_dataset = ProteinSequenceDataset(
        csv_path="data/processed/train.csv",
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
    )

    # : epoch 
    # 
    val_dataset = ProteinSequenceDataset(
        csv_path="data/processed/val.csv",
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
    )

    # DataLoader  Dataset  batch
    # shuffle=True  epoch 
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )

    # , accuracy
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    # , device 
    model = SeqProFTMiniClassifier(config).to(device)

    # 
    #  trainable  total, LoRA 
    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    # AdamW  transformer 
    # optimizer  loss 
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
    )

    #  batch  logits shape,
    printed_logits_shape = False

    for epoch in range(int(config["epochs"])):
        # train() 
        # dropout 
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            #  batch  input_ids / attention_mask / labels
            #  device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 
            optimizer.zero_grad()

            # : logits  loss
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
            # :
            # logits ,
            # , 2 
            #  batch_size=2, torch.Size([2, 2])
            if not printed_logits_shape:
                print(f"First batch logits shape: {outputs['logits'].shape}")
                printed_logits_shape = True

            # : loss 
            loss.backward()

            # :optimizer  LoRA 
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        #  epoch , loss
        average_loss = total_loss / len(train_dataloader)

        #  accuracy
        val_accuracy = evaluate_accuracy(model, val_dataloader, device)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={average_loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}"
        )

    # , checkpoint
    # checkpoint  config
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
