"""
LightGBM benchmark on UCI Cover Type (binary: class 1 vs class 2) —
mirrors examples/covertype.rs.

Run:  python examples/bench_covertype.py [--num_trees=100] [--num_leaves=31]
                                          [--logloss] [--importance]
Data: cargo run --example fetch_covertype   (first time only)
"""
import argparse
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--num_trees",  type=int,  default=100)
parser.add_argument("--num_leaves", type=int,  default=31)
parser.add_argument("--logloss",    action="store_true")
parser.add_argument("--importance", action="store_true")
args = parser.parse_args()

# ── Load CSV ───────────────────────────────────────────────────────────────────
path = Path("data/covtype_binary.csv")
if not path.exists():
    raise SystemExit("cannot open data/covtype_binary.csv — run `cargo run --example fetch_covertype` first")

df = pd.read_csv(path)
feature_cols = [c for c in df.columns if c != "label"]
X = df[feature_cols].values
y = df["label"].values.astype(float)

print(f"{len(df)} samples, {len(feature_cols)} features")

# ── Train / test split (80 / 20) — same as Rust ───────────────────────────────
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)

# ── Train ──────────────────────────────────────────────────────────────────────
params = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "num_leaves":       args.num_leaves,
    "min_data_in_leaf": 20,
    "learning_rate":    0.1,
    "lambda_l2":        1.0,
    "verbose":          -1,
}

logloss_log = []

def on_iteration(env):
    logloss_log.append(env.evaluation_result_list[0][2])

model = lgb.train(
    params,
    train_ds,
    num_boost_round=args.num_trees,
    valid_sets=[train_ds],
    callbacks=[lgb.record_evaluation({}), on_iteration],
)

# ── Evaluation ─────────────────────────────────────────────────────────────────
preds_prob  = model.predict(X_test)
train_logloss = logloss_log[-1] if logloss_log else 0.0

p = np.clip(preds_prob, 1e-15, 1 - 1e-15)
test_logloss = -np.mean(y_test * np.log(p) + (1 - y_test) * np.log(1 - p))
test_acc     = np.mean((preds_prob >= 0.5) == y_test) * 100

print(
    f"num_trees={args.num_trees}  num_leaves={args.num_leaves}"
    f"  train_logloss={train_logloss:.4f}  test_logloss={test_logloss:.4f}"
    f"  test_acc={test_acc:.2f}%"
)

# ── Per-iteration log-loss ─────────────────────────────────────────────────────
if args.logloss:
    print(f"\n{'iter':<6}  train_logloss")
    print("-" * 24)
    for i, v in enumerate(logloss_log, 1):
        print(f"{i:<6}  {v:.4f}")

# ── Feature importance (gain) ──────────────────────────────────────────────────
if args.importance:
    importance = model.feature_importance(importance_type="gain")
    ranked = sorted(zip(importance, feature_cols), reverse=True)
    print("\nFeature importance (gain):")
    print(f"{'rank':<6}  {'gain':<14}  feature")
    print("-" * 38)
    for rank, (gain, name) in enumerate(ranked, 1):
        print(f"{rank:<6}  {gain:<14.0f}  {name}")
