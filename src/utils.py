"""项目工具函数：放一些训练和推理都会用到的小工具。

人话解释：
这个文件不负责模型本身，也不负责训练循环。
它只负责“杂活”，比如：

1. 读取 config.yaml 配置文件
2. 自动选择用 GPU / Apple Silicon MPS / CPU
3. 固定随机种子，让结果更容易复现
4. 创建 outputs/ 这些输出文件夹

把这些小功能放在 utils.py 里，可以让 train.py 和 inference.py 更干净。
"""

from pathlib import Path
import random

import numpy as np
import torch
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """读取 YAML 配置文件，并返回一个 Python 字典。

    人话解释：
    config.yaml 是项目的“设置面板”。
    例如 config.yaml 里有：

        batch_size: 2

    读进 Python 以后，就可以这样使用：

        config["batch_size"]

    这样我们不用把 batch_size、learning_rate 这些值写死在代码里。
    """

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_device() -> torch.device:
    """自动选择 PyTorch 可以使用的最好设备。

    人话解释：
    - cuda：NVIDIA 显卡，一般 Linux/Windows 工作站常见
    - mps：Apple Silicon GPU，比如 M1/M2/M3/M4/M5 Mac
    - cpu：普通 CPU，最慢但几乎一定能用

    你的 MacBook Pro M5 Pro 如果 PyTorch 安装正确，一般会选择 mps。
    """

    # 如果有 NVIDIA GPU，优先用 CUDA。
    if torch.cuda.is_available():
        return torch.device("cuda")

    # 如果是 Apple Silicon，并且当前 PyTorch 支持 MPS，就用 mps。
    if torch.backends.mps.is_available():
        return torch.device("mps")

    # 如果上面两个都没有，就退回 CPU。
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """固定随机种子，让实验更容易重复。

    人话解释：
    机器学习里很多地方都有随机性：
    - 模型参数初始化
    - DataLoader 打乱数据顺序
    - dropout 随机丢掉一部分神经元

    固定 seed 不能保证所有机器上结果完全一样，
    但可以让同一台机器上的调试更稳定。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dirs() -> None:
    """创建输出文件夹。

    人话解释：
    训练结束后，我们会把 checkpoint 存到 outputs/checkpoints/。
    如果这个文件夹不存在，保存模型时就会报错。
    所以训练前先确保这些文件夹都存在。
    """

    for folder in ["outputs", "outputs/checkpoints", "outputs/logs"]:
        Path(folder).mkdir(parents=True, exist_ok=True)
