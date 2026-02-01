import argparse
from pathlib import Path
import numpy as np
import joblib

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from .config import TrainConfig
from .download_data import ensure_dataset
from .dataset import load_uci_credit_xls, TARGET_COL
from .model import MLP
from .evaluate import eval_and_plots
from .utils import set_seed, make_run_dir, save_json

from pathlib import Path
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def build_preprocess(df):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values.astype(int)

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return pre, X, y

def train_torch(model, train_loader, val_loader, device, epochs, lr, weight_decay, pos_weight):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:02d} | train_loss={np.mean(tr_losses):.4f} | val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model

def predict_proba_torch(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)
    return np.concatenate(probs, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="Download dataset from UCI")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    args = ap.parse_args()

    cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        threshold=args.threshold,
        hidden=args.hidden,
        data_dir=args.data_dir,
    )

    set_seed(cfg.seed)
    run_dir = make_run_dir("runs")

    # Data
    xls_path = None
    if args.download:
        xls_path = ensure_dataset(cfg.data_dir)
    else:
        # Try to find local file
        data_path = Path(cfg.data_dir)
        candidates = list(data_path.glob("*.xls*"))
        if not candidates:
            raise FileNotFoundError(
                f"No .xls/.xlsx found in {cfg.data_dir}. Run with --download or place dataset there."
            )
        xls_path = candidates[0]

    df = load_uci_credit_xls(str(xls_path))
    pre, X, y = build_preprocess(df)

    # Split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    # Split train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=cfg.val_size, stratify=y_trainval, random_state=cfg.seed
    )

    # Fit preprocess on train only
    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    # Baseline (logreg)
    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    base.fit(X_train_t, y_train)
    base_prob = base.predict_proba(X_test_t)[:, 1]

    # Torch data
    def to_tensor(m, y_arr):
        # sklearn transformers return sparse sometimes; make dense
        if hasattr(m, "toarray"):
            m = m.toarray()
        x = torch.tensor(m, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)
        return x, y_t

    Xtr, ytr = to_tensor(X_train_t, y_train)
    Xva, yva = to_tensor(X_val_t, y_val)
    Xte, yte = to_tensor(X_test_t, y_test)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=cfg.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = Xtr.shape[1]
    model = MLP(in_dim=in_dim, hidden=cfg.hidden, dropout=cfg.dropout).to(device)

    # pos_weight = (#neg / #pos) for BCEWithLogitsLoss
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)

    model = train_torch(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        pos_weight=pos_weight,
    )

    torch_prob = predict_proba_torch(model, test_loader, device)

    # Evaluate and save
    from sklearn.metrics import roc_auc_score, average_precision_score
    summary = {
        "baseline_logreg": {
            "roc_auc": float(roc_auc_score(y_test, base_prob)),
            "pr_auc": float(average_precision_score(y_test, base_prob)),
        }
    }

    torch_metrics = eval_and_plots(y_test, torch_prob, cfg.threshold, run_dir)
    # Evaluate and save
    from sklearn.metrics import roc_auc_score, average_precision_score

    summary = {
        "baseline_logreg": {
            "roc_auc": float(roc_auc_score(y_test, base_prob)),
            "pr_auc": float(average_precision_score(y_test, base_prob)),
        }
    }

    torch_metrics = eval_and_plots(y_test, torch_prob, cfg.threshold, run_dir)

    baseline_dir = run_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    base_metrics = eval_and_plots(y_test, base_prob, cfg.threshold, baseline_dir)

    summary["torch_mlp"] = torch_metrics
    summary["baseline_logreg_full"] = base_metrics

    save_json(summary, run_dir / "metrics.json")

    # Save model + preprocess
    torch.save(model.state_dict(), run_dir / "model.pt")
    joblib.dump(pre, run_dir / "preprocess.joblib")

    print(f"\nSaved run artifacts to: {run_dir}")
    print("Metrics:")
    print(summary)

if __name__ == "__main__":
    main()
