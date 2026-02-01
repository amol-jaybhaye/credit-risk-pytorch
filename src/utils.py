import json
from pathlib import Path
import numpy as np

def set_seed(seed: int):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_run_dir(base: str = "runs") -> Path:
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = Path(base) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=2))
