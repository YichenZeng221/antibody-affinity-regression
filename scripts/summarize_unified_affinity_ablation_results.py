"""Summarize finished unified affinity ablation runs without training.

中文人话说明：
你可以先在本地 Terminal 手动训练、手动 evaluate、手动跑 baseline。
等 checkpoint 和 prediction CSV 都准备好后，再运行这个脚本生成：
    outputs/ablation/unified_affinity_dataset_v1/ablation_results.csv
    outputs/ablation/unified_affinity_dataset_v1/ablation_report.md

这个脚本只调用 runner 里已有的汇总函数，不会启动训练或 evaluation。
"""

from __future__ import annotations

import pandas as pd

from run_unified_affinity_ablation_experiments import (
    OUTPUT_DIR,
    REPORT_PATH,
    RESULTS_PATH,
    RUNS,
    collect_row,
    write_report,
)


def main() -> None:
    """Read existing ablation artifacts and write result tables."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame([collect_row(run) for run in RUNS])
    results.to_csv(RESULTS_PATH, index=False)
    write_report(results)
    print(results.to_string(index=False))
    print(f"Saved results: {RESULTS_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print("No training or evaluation was run by this summary script.")


if __name__ == "__main__":
    main()
