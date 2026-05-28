"""Evaluate the trained SeqProFT mini model on the full test set.

:
 test set evaluation

,
:

1.  config.yaml
2.  data/processed/test.csv
3.  tokenizer
4. 
5.  outputs/checkpoints/seqproft_mvp.pt
6.  test set  inference
7.  accuracyconfusion matrix accuracy
8.  outputs/test_predictions.csv

:
    label 0 = light chain
    label 1 = heavy chain

:
    python scripts/evaluate_test_set.py
"""

from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

#  `python scripts/evaluate_test_set.py` ,
# Python  scripts/  import 
#  src/ , sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import ProteinSequenceDataset
from src.model import SeqProFTMiniClassifier
from src.utils import ensure_output_dirs, get_device, load_config


TEST_CSV_PATH = PROJECT_ROOT / "data/processed/test.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs/checkpoints/seqproft_mvp.pt"
PREDICTIONS_OUTPUT_PATH = PROJECT_ROOT / "outputs/test_predictions.csv"


def safe_class_accuracy(correct: int, total: int) -> float:
    """ accuracy

    :
     class  test set ,total  0
     0, 0.0
    """

    if total == 0:
        return 0.0
    return correct / total


def main() -> None:
    """"""

    if not TEST_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {TEST_CSV_PATH}. Please build the dataset first."
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {CHECKPOINT_PATH}. Please run python run_train.py first."
        )

    ensure_output_dirs()

    # test set ?
    #
    # :
    # train set 
    # test set 
    #  test set,,
    #  test accuracy 
    config = load_config("config.yaml")
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    test_dataset = ProteinSequenceDataset(
        csv_path=str(TEST_CSV_PATH),
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    model = SeqProFTMiniClassifier(config).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_true_labels = []
    all_predicted_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs["logits"], dim=-1)
            predicted_labels = probabilities.argmax(dim=-1)

            all_true_labels.extend(labels.cpu().tolist())
            all_predicted_labels.extend(predicted_labels.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    total_samples = len(all_true_labels)
    total_correct = sum(
        true_label == predicted_label
        for true_label, predicted_label in zip(all_true_labels, all_predicted_labels)
    )
    total_wrong = total_samples - total_correct

    # test accuracy ?
    #
    # :
    # test accuracy = test set 
    #  100  test samples  90 ,accuracy  0.90
    test_accuracy = safe_class_accuracy(total_correct, total_samples)

    # Confusion matrix ?
    #
    # ,:
    #
    # true 0 predicted 0:light chain  light
    # true 0 predicted 1:light chain  heavy
    # true 1 predicted 0:heavy chain  light
    # true 1 predicted 1:heavy chain  heavy
    confusion = {
        (0, 0): 0,
        (0, 1): 0,
        (1, 0): 0,
        (1, 1): 0,
    }

    for true_label, predicted_label in zip(all_true_labels, all_predicted_labels):
        confusion[(true_label, predicted_label)] += 1

    class_0_total = confusion[(0, 0)] + confusion[(0, 1)]
    class_1_total = confusion[(1, 0)] + confusion[(1, 1)]
    class_0_accuracy = safe_class_accuracy(confusion[(0, 0)], class_0_total)
    class_1_accuracy = safe_class_accuracy(confusion[(1, 1)], class_1_total)

    #  class  accuracy?
    #
    # :
    #  accuracy 
    #  test set  heavy light , heavy 
    #  light accuracy  heavy accuracy,
    print()
    print("Test set evaluation")
    print("Label meaning: 0 = light chain, 1 = heavy chain")
    print(f"Total test samples: {total_samples}")
    print(f"Wrong predictions: {total_wrong}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Class 0 accuracy (light chain): {class_0_accuracy:.4f}")
    print(f"Class 1 accuracy (heavy chain): {class_1_accuracy:.4f}")
    print()
    print("Confusion matrix")
    print("Rows = true label, columns = predicted label")
    print("              predicted 0    predicted 1")
    print(f"true 0        {confusion[(0, 0)]:11d}    {confusion[(0, 1)]:11d}")
    print(f"true 1        {confusion[(1, 0)]:11d}    {confusion[(1, 1)]:11d}")

    raw_test_dataframe = pd.read_csv(TEST_CSV_PATH)

    #  processed CSV  pdb/chain/chain_type
    #  CSV ,
    #  sequence,label;metadata 
    for metadata_column in ["pdb", "chain", "chain_type"]:
        if metadata_column not in raw_test_dataframe.columns:
            raw_test_dataframe[metadata_column] = ""

    prediction_rows = []

    for sequence, true_label, predicted_label, pdb_id, chain_id, chain_type, probs in zip(
        raw_test_dataframe["sequence"].astype(str).tolist(),
        all_true_labels,
        all_predicted_labels,
        raw_test_dataframe["pdb"].astype(str).tolist(),
        raw_test_dataframe["chain"].astype(str).tolist(),
        raw_test_dataframe["chain_type"].astype(str).tolist(),
        all_probabilities,
    ):
        class_0_probability = float(probs[0])
        class_1_probability = float(probs[1])
        correct = true_label == predicted_label

        prediction_rows.append(
            {
                "sequence": sequence,
                "label": true_label,
                "predicted_label": predicted_label,
                "pdb": pdb_id,
                "chain": chain_id,
                "chain_type": chain_type,
                "class_0_probability": class_0_probability,
                "class_1_probability": class_1_probability,
                "correct": correct,
            }
        )

    predictions_dataframe = pd.DataFrame(
        prediction_rows,
        columns=[
            "sequence",
            "label",
            "predicted_label",
            "pdb",
            "chain",
            "chain_type",
            "class_0_probability",
            "class_1_probability",
            "correct",
        ],
    )
    predictions_dataframe.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)

    #  predictions.csv?
    #
    # :
    #  accuracy 
    # ,:
    #  PDB chain ?
    #  light  heavy?
    print()
    print(f"Saved test predictions to {PREDICTIONS_OUTPUT_PATH}")

    wrong_predictions = predictions_dataframe[predictions_dataframe["correct"] == False]

    print()
    print("First 10 wrong predictions")
    if wrong_predictions.empty:
        print("No wrong predictions found.")
    else:
        for _, row in wrong_predictions.head(10).iterrows():
            print(f"sequence_first_40_aa: {row['sequence'][:40]}")
            print(f"pdb: {row['pdb']}")
            print(f"chain: {row['chain']}")
            print(f"chain_type: {row['chain_type']}")
            print(f"true_label: {int(row['label'])}")
            print(f"predicted_label: {int(row['predicted_label'])}")
            print(f"class_0_probability: {row['class_0_probability']:.4f}")
            print(f"class_1_probability: {row['class_1_probability']:.4f}")
            print()


if __name__ == "__main__":
    main()
