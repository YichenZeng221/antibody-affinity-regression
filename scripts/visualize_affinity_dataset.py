"""Visualize the Stage 1 affinity regression dataset.

:
,, CSV

:
1. train / val / test  target 
2. heavy / light / antigen sequence 
3. , 8.1 
4. fold error 

:
    ./.venv/bin/python scripts/visualize_affinity_dataset.py
"""

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Matplotlib 
#  Mac /  home ,
#  outputs/matplotlib_cache/ 
MPL_CACHE_DIR = PROJECT_ROOT / "outputs" / "matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))
os.environ.setdefault("HOME", str(MPL_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import numpy as np
import pandas as pd


# Agg  matplotlib 
# //IDE , GUI 
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(PROJECT_ROOT))

from src.utils import load_config


SPLIT_COLORS = {
    "train": "#4C78A8",
    "val": "#F58518",
    "test": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    """Read command line arguments.

     config_affinity.yaml
     --config  clean_v2 all_methods / spr_only 
    """

    parser = argparse.ArgumentParser(description="Visualize affinity regression dataset.")
    parser.add_argument("--config", default="config_affinity.yaml")
    return parser.parse_args()


def make_output_dir() -> Path:
    """Create outputs/figures/ if it does not exist yet."""

    output_dir = PROJECT_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_histogram_for_splits(
    split_data: dict[str, pd.DataFrame],
    value_getter,
    title: str,
    x_label: str,
    output_path: Path,
    bins: np.ndarray,
) -> None:
    """Draw one histogram that compares train / val / test.

    value_getter :
    -  dataframe
    - 

     target, sequence length

    histogram :
    ,
    train/val/test , split 

    :
     train/val/test , bin edges
     split ,,
    
    """

    plt.figure(figsize=(9, 6))

    for split_name, dataframe in split_data.items():
        values = value_getter(dataframe)
        plt.hist(
            values,
            bins=bins,
            alpha=0.55,
            label=split_name,
            color=SPLIT_COLORS[split_name],
        )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_shared_bins(split_data: dict[str, pd.DataFrame], value_getter, bin_count: int = 20) -> np.ndarray:
    """Create shared histogram bin edges from all train/val/test values."""

    all_values = []
    for dataframe in split_data.values():
        values = value_getter(dataframe)
        all_values.extend(pd.Series(values).astype(float).tolist())

    global_min = min(all_values)
    global_max = max(all_values)

    if global_min == global_max:
        # ,, linspace  bin edge
        global_min -= 0.5
        global_max += 0.5

    return np.linspace(global_min, global_max, bin_count)


def save_true_vs_predicted(predictions: pd.DataFrame, output_path: Path) -> None:
    """Draw true target vs model prediction scatter plot.

    :
     test sample:
    - x  target
    - y  target

    , y=x 
    , collapsed 

    :
     x/y ,scatter plot 
    equal axis  y=x
    """

    true_values = predictions["true_neg_log10_affinity"].astype(float)
    predicted_values = predictions["predicted_neg_log10_affinity"].astype(float)

    min_value = min(true_values.min(), predicted_values.min())
    max_value = max(true_values.max(), predicted_values.max())
    padding = (max_value - min_value) * 0.05
    min_value -= padding
    max_value += padding

    plt.figure(figsize=(7, 7))
    plt.scatter(true_values, predicted_values, alpha=0.75, color="#4C78A8", label="test samples")

    # y = x :,
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        color="#E45756",
        label="perfect prediction: y = x",
    )

    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("True vs Predicted Affinity (Test Set, Equal Axis)")
    plt.xlabel("True neg_log10_affinity")
    plt.ylabel("Predicted neg_log10_affinity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_residual_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    """Draw residual plot: true target vs prediction error.

    :
    residual = predicted - true

    :
    -  target ;
    -  target ;
    -  regression-to-mean,

     0  0 ,
    
    """

    true_values = predictions["true_neg_log10_affinity"].astype(float)
    errors = predictions["error"].astype(float)

    plt.figure(figsize=(8, 6))
    plt.scatter(true_values, errors, alpha=0.75, color="#4C78A8", label="test residuals")
    plt.axhline(0, color="#E45756", linestyle="--", label="zero error")
    plt.title("Residual Plot (Test Set)")
    plt.xlabel("True neg_log10_affinity")
    plt.ylabel("Prediction error = predicted - true")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_error_histogram(predictions: pd.DataFrame, output_path: Path) -> None:
    """Draw histogram of prediction error.

    error = predicted - true
     error  0, neg_log10_affinity 
     error  0, neg_log10_affinity 
    """

    errors = predictions["error"].astype(float)

    plt.figure(figsize=(9, 6))
    plt.hist(errors, bins=30, alpha=0.75, color="#4C78A8", label="test errors")
    plt.axvline(0, color="#E45756", linestyle="--", label="zero error")
    plt.title("Test Error Distribution")
    plt.xlabel("Prediction error: predicted - true")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_fold_error_histogram(predictions: pd.DataFrame, output_path: Path) -> None:
    """Draw histogram of fold error with a log-scaled x-axis.

     target  -log10(affinity), log10 :
        fold_error = 10 ** absolute_error

    fold_error , log scale
    """

    fold_errors = predictions["fold_error"].astype(float)

    plt.figure(figsize=(9, 6))
    plt.hist(fold_errors, bins=30, alpha=0.75, color="#F58518", label="test fold errors")
    plt.xscale("log")
    plt.title("Fold Error Distribution (log-scaled x-axis)")
    plt.xlabel("Fold error = 10 ** absolute_error")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_distribution_hint(train: pd.DataFrame, test: pd.DataFrame, target_column: str) -> None:
    """Print a small text summary to help interpret the plots."""

    train_target = train[target_column].astype(float)
    test_target = test[target_column].astype(float)

    print("Target quick check:")
    print(f"  train mean/std: {train_target.mean():.4f} / {train_target.std():.4f}")
    print(f"  test  mean/std: {test_target.mean():.4f} / {test_target.std():.4f}")
    print(f"  mean difference: {abs(train_target.mean() - test_target.mean()):.4f}")
    print()

    for column_name in ["heavy_sequence", "light_sequence", "antigen_sequence"]:
        train_lengths = train[column_name].astype(str).str.len()
        test_lengths = test[column_name].astype(str).str.len()
        print(f"{column_name} length quick check:")
        print(f"  train mean/max: {train_lengths.mean():.2f} / {int(train_lengths.max())}")
        print(f"  test  mean/max: {test_lengths.mean():.2f} / {int(test_lengths.max())}")
    print()


def print_prediction_hint(predictions: pd.DataFrame) -> None:
    """Print a tiny prediction summary after drawing prediction figures.

    prediction std :
    , prediction collapse
    """

    predicted = predictions["predicted_neg_log10_affinity"].astype(float)
    true_values = predictions["true_neg_log10_affinity"].astype(float)

    print("Prediction quick check:")
    print(f"  true min/max/mean/std: {true_values.min():.4f} / {true_values.max():.4f} / {true_values.mean():.4f} / {true_values.std():.4f}")
    print(f"  pred min/max/mean/std: {predicted.min():.4f} / {predicted.max():.4f} / {predicted.mean():.4f} / {predicted.std():.4f}")
    print("  If pred std is much smaller than true std, predictions are probably collapsed near one value.")
    print()


def main() -> None:
    """Create all affinity dataset visualization PNG files."""

    args = parse_args()
    config = load_config(args.config)
    target_column = config["target_column"]
    output_dir = make_output_dir()

    split_data = {
        "train": pd.read_csv(PROJECT_ROOT / config["train_csv"]),
        "val": pd.read_csv(PROJECT_ROOT / config["val_csv"]),
        "test": pd.read_csv(PROJECT_ROOT / config["test_csv"]),
    }

    generated_files = []

    target_path = output_dir / "target_distribution_train_val_test.png"
    target_bins = make_shared_bins(
        split_data,
        value_getter=lambda df: df[target_column].astype(float),
        bin_count=20,
    )
    save_histogram_for_splits(
        split_data,
        value_getter=lambda df: df[target_column].astype(float),
        title="Train/Val/Test Target Distribution",
        x_label="neg_log10_affinity",
        output_path=target_path,
        bins=target_bins,
    )
    generated_files.append(target_path)

    for sequence_column, output_name, title, x_label in [
        (
            "heavy_sequence",
            "heavy_length_distribution.png",
            "Heavy Sequence Length Distribution",
            "Heavy sequence length",
        ),
        (
            "light_sequence",
            "light_length_distribution.png",
            "Light Sequence Length Distribution",
            "Light sequence length",
        ),
        (
            "antigen_sequence",
            "antigen_length_distribution.png",
            "Antigen Sequence Length Distribution",
            "Antigen sequence length",
        ),
    ]:
        output_path = output_dir / output_name
        length_bins = make_shared_bins(
            split_data,
            value_getter=lambda df, column=sequence_column: df[column].astype(str).str.len(),
            bin_count=20,
        )
        save_histogram_for_splits(
            split_data,
            value_getter=lambda df, column=sequence_column: df[column].astype(str).str.len(),
            title=title,
            x_label=x_label,
            output_path=output_path,
            bins=length_bins,
        )
        generated_files.append(output_path)

    predictions_path = PROJECT_ROOT / config["predictions_path"]
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)

        true_vs_predicted_path = output_dir / "true_vs_predicted_affinity.png"
        save_true_vs_predicted(predictions, true_vs_predicted_path)
        generated_files.append(true_vs_predicted_path)

        residual_path = output_dir / "residual_plot.png"
        save_residual_plot(predictions, residual_path)
        generated_files.append(residual_path)

        error_path = output_dir / "test_error_distribution.png"
        save_error_histogram(predictions, error_path)
        generated_files.append(error_path)

        fold_error_path = output_dir / "fold_error_distribution.png"
        save_fold_error_histogram(predictions, fold_error_path)
        generated_files.append(fold_error_path)

        print_prediction_hint(predictions)
    else:
        print(f"No prediction file found at {predictions_path}.")
        print("Prediction figures were skipped.")
        print()

    print_distribution_hint(split_data["train"], split_data["test"], target_column)

    print("Generated figures:")
    for path in generated_files:
        print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
