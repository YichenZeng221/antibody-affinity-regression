"""Evaluate the trained SeqProFT mini model on the full test set.

中文人话说明：
这个脚本做“正式 test set evaluation”。

它不会训练模型，也不会修改模型结构。
它只做这些事：

1. 读取 config.yaml
2. 读取 data/processed/test.csv
3. 加载 tokenizer
4. 创建和训练时相同的模型结构
5. 加载 outputs/checkpoints/seqproft_mvp.pt
6. 在整个 test set 上跑 inference
7. 计算 accuracy、confusion matrix、每个类别的 accuracy
8. 保存每条样本的预测结果到 outputs/test_predictions.csv

标签含义：
    label 0 = light chain
    label 1 = heavy chain

运行命令：
    python scripts/evaluate_test_set.py
"""

from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 当我们运行 `python scripts/evaluate_test_set.py` 时，
# Python 默认会把 scripts/ 当成 import 起点。
# 但 src/ 在项目根目录下，所以这里手动把项目根目录加入 sys.path。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import ProteinSequenceDataset
from src.model import SeqProFTMiniClassifier
from src.utils import ensure_output_dirs, get_device, load_config


TEST_CSV_PATH = PROJECT_ROOT / "data/processed/test.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs/checkpoints/seqproft_mvp.pt"
PREDICTIONS_OUTPUT_PATH = PROJECT_ROOT / "outputs/test_predictions.csv"


def safe_class_accuracy(correct: int, total: int) -> float:
    """安全计算某个类别的 accuracy。

    中文人话说明：
    如果某个 class 在 test set 里一条样本都没有，total 就是 0。
    这时候不能除以 0，所以返回 0.0。
    """

    if total == 0:
        return 0.0
    return correct / total


def main() -> None:
    """主评估流程。"""

    if not TEST_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {TEST_CSV_PATH}. Please build the dataset first."
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {CHECKPOINT_PATH}. Please run python run_train.py first."
        )

    ensure_output_dirs()

    # test set 为什么不能参与训练？
    #
    # 人话解释：
    # train set 是给模型学习用的。
    # test set 是最后考试用的。
    # 如果模型训练时看过 test set，就像考试前已经看过答案，
    # 那 test accuracy 就不再公平。
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

    # test accuracy 代表什么？
    #
    # 人话解释：
    # test accuracy = test set 上预测正确的比例。
    # 例如 100 条 test samples 里预测对 90 条，accuracy 就是 0.90。
    test_accuracy = safe_class_accuracy(total_correct, total_samples)

    # Confusion matrix 怎么看？
    #
    # 对二分类，这里用四个数字：
    #
    # true 0 predicted 0：light chain 被正确预测成 light
    # true 0 predicted 1：light chain 被错误预测成 heavy
    # true 1 predicted 0：heavy chain 被错误预测成 light
    # true 1 predicted 1：heavy chain 被正确预测成 heavy
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

    # 为什么要看每个 class 的 accuracy？
    #
    # 人话解释：
    # 总 accuracy 可能隐藏问题。
    # 如果 test set 里 heavy 很多、light 很少，模型全猜 heavy 也可能总分不低。
    # 分别看 light accuracy 和 heavy accuracy，可以知道模型是不是偏向某一类。
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

    # 新版 processed CSV 会包含 pdb/chain/chain_type。
    # 为了让脚本对旧 CSV 也不直接崩掉，如果缺列就填空字符串。
    # 训练和评估真正需要的是 sequence,label；metadata 主要用于分析。
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

    # 为什么要保存 predictions.csv？
    #
    # 人话解释：
    # 只看一个 accuracy 数字不够。
    # 保存每条样本的预测，可以让我们之后回头分析：
    # 哪些 PDB、哪些 chain 容易错？
    # 模型是不是总把 light 误判成 heavy？
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
