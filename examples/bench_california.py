"""
LightGBM benchmark on California Housing — mirrors examples/california.rs exactly.

Run:  python examples/bench_california.py [--num_trees=100] [--num_leaves=31] [--error] [--importance]
Data: python examples/fetch_california.py   (first time only)
"""
import argparse
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--num_trees",   type=int,  default=50)
parser.add_argument("--num_leaves",  type=int,  default=31)
parser.add_argument("--error",       action="store_true")
parser.add_argument("--importance",  action="store_true")
args = parser.parse_args()

# ── Load CSV ───────────────────────────────────────────────────────────────────
path = Path("data/california_housing.csv")
if not path.exists():
    raise SystemExit("cannot open data/california_housing.csv — run `python examples/fetch_california.py` first")

df = pd.read_csv(path)
print(f"{len(df)} samples, {len(df.columns) - 1} features")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ── Train / test split (80 / 20) — same as Rust ───────────────────────────────
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_ds = lgb.Dataset(X_train, label=y_train, feature_name=list(df.columns[:-1]), free_raw_data=False)

# ── Train ──────────────────────────────────────────────────────────────────────
params = {
    "objective":        "regression",
    "metric":           "rmse",
    "num_leaves":       args.num_leaves,
    "min_data_in_leaf": 20,
    "learning_rate":    0.1,
    "lambda_l2":        1.0,
    "verbose":          -1,
}

rmse_per_iter = []

def on_iteration(env):
    rmse = env.evaluation_result_list[0][2]
    rmse_per_iter.append(rmse)

model = lgb.train(
    params,
    train_ds,
    num_boost_round=args.num_trees,
    valid_sets=[train_ds],
    callbacks=[lgb.record_evaluation({}), on_iteration],
)

# ── Evaluation ─────────────────────────────────────────────────────────────────
preds = model.predict(X_test)
test_rmse = math.sqrt(np.mean((preds - y_test) ** 2))
train_rmse_final = rmse_per_iter[-1] if rmse_per_iter else 0.0

print(
    f"num_trees={args.num_trees}  num_leaves={args.num_leaves}"
    f"  train_rmse={train_rmse_final * 100_000:.0f}"
    f"  test_rmse={test_rmse * 100_000:.0f}"
)

# ── Per-iteration RMSE ─────────────────────────────────────────────────────────
if args.error:
    print(f"\n{'iter':<6}  {'rmse'}")
    print("-" * 18)
    for i, rmse in enumerate(rmse_per_iter, 1):
        print(f"{i:<6}  {rmse * 100_000:.0f}")

# ── Feature importance (gain) ──────────────────────────────────────────────────
if args.importance:
    importance = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    ranked = sorted(zip(importance, names), reverse=True)

    print("\nFeature importance (gain):")
    print(f"{'rank':<6}  {'gain':<14}  feature")
    print("-" * 38)
    for rank, (gain, name) in enumerate(ranked, 1):
        print(f"{rank:<6}  {gain:<14.0f}  {name}")
