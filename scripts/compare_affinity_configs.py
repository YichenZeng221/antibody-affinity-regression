"""Compare TDC affinity configs before a fair dataset-version experiment.

中文人话说明：
这个脚本不训练模型。
它只检查两份 YAML config：

1. 原始 TDC v1 config。
2. TDC + SAbDab supplement v1 config。

我们允许 dataset path 和输出路径不同。
但是 model、LoRA、learning rate、batch size 这类训练条件不能悄悄变，
否则后面的 metric 差异就不是 apples-to-apples comparison。
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 这个脚本从 scripts/ 目录直接运行时，先把项目根目录加入 import path，
# Python 才能找到 src/utils.py 里的 load_config。
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_config


ORIGINAL_CONFIG_PATH = PROJECT_ROOT / "config_affinity_tdc_v1_antigen.yaml"
SUPPLEMENT_CONFIG_PATH = PROJECT_ROOT / "config_affinity_tdc_plus_sabdab_supplement_v1.yaml"
REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "tdc_plus_sabdab_supplement_v1"
    / "config_comparison_report.md"
)

# 这些字段本来就应该不同：
# - dataset CSV 指向不同 dataset version
# - checkpoint / predictions 不能覆盖旧实验输出
# - output/run/experiment 命名字段若未来加入，也允许不同
EXPECTED_DIFFERENCE_FIELDS = {
    "train_csv",
    "val_csv",
    "test_csv",
    "checkpoint_path",
    "predictions_path",
    "output_dir",
    "run_name",
    "experiment_name",
}

# 这些字段是公平训练比较里最需要盯住的字段。
# 当前 config 没有显式 pooling/loss/metrics，代码固定 mean pooling + MSE + regression metrics；
# 如果未来 config 里加入它们，这个脚本也会把差异抓出来。
UNEXPECTED_CONTROL_FIELDS = {
    "model_name",
    "task_type",
    "target_column",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "learning_rate",
    "batch_size",
    "epochs",
    "max_length",
    "pooling",
    "pooling_method",
    "loss",
    "loss_name",
    "seed",
    "metrics",
    "evaluation_metrics",
}


def value_for_display(config: dict, field_name: str):
    """Return config value or readable marker when field is absent."""

    return config[field_name] if field_name in config else "<missing>"


def compare_configs(original: dict, supplement: dict) -> list[dict]:
    """Return all fields whose values differ."""

    all_fields = sorted(set(original) | set(supplement))
    differences = []
    for field_name in all_fields:
        original_value = value_for_display(original, field_name)
        supplement_value = value_for_display(supplement, field_name)
        if original_value != supplement_value:
            differences.append(
                {
                    "field": field_name,
                    "original": original_value,
                    "supplement": supplement_value,
                }
            )
    return differences


def classify_differences(differences: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split diff rows into expected vs fairness-breaking differences."""

    expected = []
    unexpected = []
    for difference in differences:
        if difference["field"] in EXPECTED_DIFFERENCE_FIELDS:
            expected.append(difference)
        else:
            # A changed unknown field is also unexpected.
            # That keeps config drift visible instead of silently passing.
            unexpected.append(difference)
    return expected, unexpected


def invariant_status(original: dict, supplement: dict) -> dict:
    """Show watched training fields even when they are equal or both absent."""

    status = {}
    for field_name in sorted(UNEXPECTED_CONTROL_FIELDS):
        original_value = value_for_display(original, field_name)
        supplement_value = value_for_display(supplement, field_name)
        status[field_name] = {
            "original": original_value,
            "supplement": supplement_value,
            "same": original_value == supplement_value,
        }
    return status


def markdown_value(value) -> str:
    """Format one value compactly inside Markdown table cell."""

    return str(value).replace("|", "\\|")


def difference_table(differences: list[dict]) -> list[str]:
    """Render config diff rows as Markdown table lines."""

    if not differences:
        return ["No differences in this category."]

    lines = [
        "| field | original config | supplement config |",
        "|---|---|---|",
    ]
    for difference in differences:
        lines.append(
            f"| `{difference['field']}` | `{markdown_value(difference['original'])}` | "
            f"`{markdown_value(difference['supplement'])}` |"
        )
    return lines


def write_report(
    expected: list[dict],
    unexpected: list[dict],
    watched_status: dict,
    passed: bool,
) -> None:
    """Write Markdown report beside supplement dataset reports."""

    watched_lines = [
        "| field | same | original | supplement |",
        "|---|---|---|---|",
    ]
    for field_name, status in watched_status.items():
        watched_lines.append(
            f"| `{field_name}` | `{status['same']}` | `{markdown_value(status['original'])}` | "
            f"`{markdown_value(status['supplement'])}` |"
        )

    lines = [
        "# Affinity Config Comparison Report",
        "",
        "## Scope",
        "",
        f"- Original config: `{ORIGINAL_CONFIG_PATH.relative_to(PROJECT_ROOT)}`",
        f"- New config: `{SUPPLEMENT_CONFIG_PATH.relative_to(PROJECT_ROOT)}`",
        "- Goal: check apples-to-apples dataset-version comparison before any training.",
        "",
        "## Expected Differences",
        "",
        *difference_table(expected),
        "",
        "## Unexpected Differences",
        "",
        *difference_table(unexpected),
        "",
        "## Watched Training Fields",
        "",
        "If `pooling`, `loss`, or `metrics` are missing in both YAML files, current code behavior is still shared: "
        "mean pooling lives in the model, MSE loss lives in the regression model, and regression metrics live in evaluation code.",
        "",
        *watched_lines,
        "",
        "## Verdict",
        "",
    ]
    if passed:
        lines.append("**CONFIG CHECK PASSED: apples-to-apples dataset-version comparison.**")
    else:
        lines.append("**CONFIG CHECK FAILED: training comparison would not be fair.**")
    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "The supplement v1 dataset was re-split after merging supplement rows. "
            "Future metric differences must be interpreted as a dataset-version comparison, "
            "not as a pure `+7 rows` effect.",
            "",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def print_differences(title: str, differences: list[dict]) -> None:
    """Print one diff category in terminal."""

    print(title)
    if not differences:
        print("  None")
        return
    for difference in differences:
        print(
            f"  {difference['field']}: "
            f"{difference['original']} -> {difference['supplement']}"
        )


def main() -> None:
    """Load YAML configs, compare them, and write the Markdown report."""

    original = load_config(ORIGINAL_CONFIG_PATH)
    supplement = load_config(SUPPLEMENT_CONFIG_PATH)
    differences = compare_configs(original, supplement)
    expected, unexpected = classify_differences(differences)
    watched_status = invariant_status(original, supplement)
    passed = len(unexpected) == 0

    print(f"Original config: {ORIGINAL_CONFIG_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Supplement config: {SUPPLEMENT_CONFIG_PATH.relative_to(PROJECT_ROOT)}")
    print_differences("Expected differences:", expected)
    print_differences("Unexpected differences:", unexpected)
    if passed:
        print("CONFIG CHECK PASSED: apples-to-apples dataset-version comparison.")
    else:
        print("CONFIG CHECK FAILED: training comparison would not be fair.")
    write_report(expected, unexpected, watched_status, passed)
    print(f"Markdown report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
