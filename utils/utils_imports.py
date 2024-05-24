import yaml
import time
from typing import Tuple, List, Optional, Union, Any
from utils.utils import *
import torch
torch.manual_seed(0)
import random
random.seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


CONFIG_FILE_PATH = "config.yml"

def load_config(config_file: str) -> Dict[str, Any]:
    try:
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: Configuration file not found.")
        exit(1)

config = load_config(CONFIG_FILE_PATH)
