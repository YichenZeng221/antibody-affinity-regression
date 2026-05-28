#!/usr/bin/env bash

# 中文人话说明：
# 这个小脚本只负责在 macOS 上打开 ablation 汇报图。
# 它不会训练模型，也不会修改 dataset。

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIGURE_PATH="$PROJECT_ROOT/outputs/final_reports/figures/ablation_mae_spearman_baseline.png"
FIGURE_DIR="$PROJECT_ROOT/outputs/final_reports/figures"

if [[ ! -f "$FIGURE_PATH" ]]; then
  echo "找不到图片：$FIGURE_PATH"
  echo "请先运行："
  echo "  ./.venv/bin/python scripts/make_unified_ablation_report_figures.py"
  exit 1
fi

echo "正在打开图片："
echo "  $FIGURE_PATH"
open "$FIGURE_PATH"

echo "正在打开图片文件夹："
echo "  $FIGURE_DIR"
open "$FIGURE_DIR"
