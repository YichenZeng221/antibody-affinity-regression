"""Train the affinity regression MVP.

中文人话说明：
这是 affinity regression 的训练入口文件。
你运行这个文件，它会：
1. 读取 config_affinity.yaml。
2. 根据配置加载数据、tokenizer、模型。
3. 开始训练。
4. 保存 checkpoint。

它故意很短，因为真正的训练细节都放在 src/affinity_train.py。

运行：
    python run_train_affinity.py
"""

import argparse

from src.affinity_train import train_affinity
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    """Read command line arguments.

    默认仍然读取 config_affinity.yaml，所以旧命令行为不变。
    如果想训练 clean_v2，可以传入 --config 指向新的配置文件。
    """

    parser = argparse.ArgumentParser(description="Train affinity regression model.")
    parser.add_argument("--config", default="config_affinity.yaml")
    return parser.parse_args()


def main() -> None:
    """Load config_affinity.yaml and start training."""

    args = parse_args()

    # config_affinity.yaml 像“实验设置面板”：
    # 数据路径、batch_size、learning_rate、checkpoint 路径都在里面。
    config = load_config(args.config)
    train_affinity(config)


if __name__ == "__main__":
    main()
