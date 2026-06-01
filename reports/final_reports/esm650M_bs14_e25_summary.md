# ESM2 650M Unweighted bs14 25 Epoch Summary

## Setup

- Model: ESM2 650M + LoRA + all-six-CDR cross-attention
- Dataset: ANDD antibody v2 stratified split
- Loss: unweighted regression
- Seed: 2026
- Batch size: 14
- Learning rate: 1e-5
- Epochs: 25
- Device: CUDA, RTX 6000 Ada

## Training status

Training completed successfully for 25 epochs.

Checkpoint saved to:

outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_esm650M_unweighted_s2026_lr1e-5_e25_bs14/checkpoints/best_model.pt

## Validation trend

Early training improved clearly:

- Epoch 1: val_MAE = 2.3289, val_Spearman = -0.0054
- Epoch 5: val_MAE = 1.0490, val_Spearman = 0.2360
- Epoch 10: val_MAE = 1.0218, val_Spearman = 0.3468
- Epoch 15: val_MAE = 1.0098, val_Spearman = 0.3978
- Epoch 20: val_MAE = 0.9899, val_Spearman = 0.4501
- Epoch 25: val_MAE = 0.9910, val_Spearman = 0.4385

Best observed validation Spearman was around epoch 20:

- Epoch 20: val_MAE = 0.9899, val_RMSE = 1.4013, val_Spearman = 0.4501

Final epoch:

- Epoch 25: train_loss = 0.9897
- val_MAE = 0.9910
- val_RMSE = 1.3912
- val_Spearman = 0.4385
- val_MAE_fold_error ≈ 9.8x
- val_RMSE_fold_error ≈ 24.6x

## Interpretation

The 650M bs14 run learned steadily from epoch 1 to around epoch 20, but validation performance plateaued after that.

Compared with the previous 150M unweighted results:

- 150M seed42: MAE = 0.9047, Spearman = 0.5221, tail_MAE = 1.5150
- 150M seed123: MAE = 0.8815, Spearman = 0.5532, tail_MAE = 1.6043

The 650M bs14 25-epoch run did not outperform the 150M unweighted baseline. It reached about val_Spearman ≈ 0.45 and val_MAE ≈ 0.99, then plateaued.

Likely issue:

- batch_size = 14 gives only about 67 updates per epoch
- 25 epochs gives about 1675 updates total
- this may be too few/far too large-batch for this small dataset

Recommended next 650M setting if continuing later:

- batch_size = 8 or 4
- learning_rate = 1e-5
- epochs = 20–30
- compare validation-selected checkpoint against 150M seed42/seed123
