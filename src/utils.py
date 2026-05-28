""":

:
,
,:

1.  config.yaml 
2.  GPU / Apple Silicon MPS / CPU
3. ,
4.  outputs/ 

 utils.py , train.py  inference.py 
"""

from pathlib import Path
import random

import numpy as np
import torch
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """ YAML , Python 

    :
    config.yaml 
     config.yaml :

        batch_size: 2

     Python ,:

        config["batch_size"]

     batch_sizelearning_rate 
    """

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_device() -> torch.device:
    """ PyTorch 

    :
    - cuda:NVIDIA , Linux/Windows 
    - mps:Apple Silicon GPU, M1/M2/M3/M4/M5 Mac
    - cpu: CPU,

     MacBook Pro M5 Pro  PyTorch , mps
    """

    #  NVIDIA GPU, CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")

    #  Apple Silicon, PyTorch  MPS, mps
    if torch.backends.mps.is_available():
        return torch.device("mps")

    # , CPU
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """,

    :
    :
    - 
    - DataLoader 
    - dropout 

     seed ,
    
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dirs() -> None:
    """

    :
    , checkpoint  outputs/checkpoints/
    ,
    
    """

    for folder in ["outputs", "outputs/checkpoints", "outputs/logs"]:
        Path(folder).mkdir(parents=True, exist_ok=True)
