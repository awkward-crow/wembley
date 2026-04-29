"""
LightGBM benchmark on California Housing — mirrors examples/california.rs exactly.

Run:  python examples/bench_california.py [--num_trees=100] [--num_leaves=31]
Data: python examples/fetch_california.py   (first time only)
"""
import argparse
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--num_trees",  type=int, default=50)
parser.add_argument("--num_leaves", type=int, default=31)
args = parser.parse_args()

print(f"num_trees={args.num_trees}  num_leaves={args.num_leaves}")

# ── Load CSV ───────────────────────────────────────────────────────────────────
path = Path("data/california_housing.csv")
if not path.exists():
    raise SystemExit("cannot open data/california_housing.csv — run `python examples/fetch_california.py` first")

df = pd.read_csv(path)
print(f"California Housing: {len(df)} samples, {len(df.columns) - 1} features")

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
    "learning_rate":    0.05,
    "lambda_l2":        1.0,
    "verbose":          -1,
}

print(f"\n{'iter':<6}  {'rmse'}")
print("-" * 18)

rmse_per_iter = []

def on_iteration(env):
    # eval_results: [(dataset_name, metric_name, value, is_higher_better), ...]
    rmse = env.evaluation_result_list[0][2]
    rmse_per_iter.append(rmse)
    print(f"{env.iteration:<6}  {rmse * 100_000:.0f}")

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
print(f"\nTest RMSE:  {test_rmse * 100_000:.0f}")

# ── Feature importance (gain) ──────────────────────────────────────────────────
importance = model.feature_importance(importance_type="gain")
names = model.feature_name()
ranked = sorted(zip(importance, names), reverse=True)

print("\nFeature importance (gain):")
print(f"{'rank':<6}  {'gain':<14}  feature")
print("-" * 38)
for rank, (gain, name) in enumerate(ranked, 1):
    print(f"{rank:<6}  {gain:<14.0f}  {name}")
