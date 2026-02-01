from dataclasses import dataclass
from typing import List

@dataclass
class TrainConfig:
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2  # fraction of train used for val
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: List[int] = None
    dropout: float = 0.2
    threshold: float = 0.5
    data_dir: str = "data"
