"""
LightGBM benchmark on UCI Bike Sharing (daily) — mirrors examples/bike.rs.
Runs quantile regression at α=0.9 / 0.5 (median) / 0.1.

Run:  python examples/bench_bike.py [--num_trees=200] [--num_leaves=15]
                                     [--shuffle] [--rmse] [--importance]
Data: cargo run --example fetch_bike   (first time only)
"""
import argparse
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--num_trees",  type=int,  default=200)
parser.add_argument("--num_leaves", type=int,  default=15)
parser.add_argument("--shuffle",    action="store_true")
parser.add_argument("--rmse",       action="store_true")
parser.add_argument("--importance", action="store_true")
args = parser.parse_args()

# ── Load CSV ───────────────────────────────────────────────────────────────────
path = Path("data/bike_sharing_day.csv")
if not path.exists():
    raise SystemExit("cannot open data/bike_sharing_day.csv — run `cargo run --example fetch_bike` first")

df = pd.read_csv(path)

TARGET = "cnt"
DROP   = ["instant", "dteday", "casual", "registered"]

feature_cols = [c for c in df.columns if c != TARGET and c not in DROP]
X = df[feature_cols].values
y = df[TARGET].values.astype(float)

# ── Shuffle (fixed seed matching --shuffle behaviour in bike.rs) ───────────────
if args.shuffle:
    rng = np.random.default_rng(0xDEADBEEF)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

split_label = "shuffled" if args.shuffle else "chronological"
print(f"{len(df)} samples, {len(feature_cols)} features  [{split_label}]")

# ── Train / test split (80 / 20) ──────────────────────────────────────────────
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

BASE_PARAMS = {
    "num_leaves":       args.num_leaves,
    "min_data_in_leaf": 10,
    "learning_rate":    0.05,
    "lambda_l2":        1.0,
    "verbose":          -1,
}

def make_callback(log):
    def cb(env):
        log.append(env.evaluation_result_list[0][2])
    return cb

# ── Quantile Regression ────────────────────────────────────────────────────────
for alpha in (0.9, 0.5, 0.1):
    q_log = []
    train_ds_q = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)
    q_model = lgb.train(
        {**BASE_PARAMS, "objective": "quantile", "metric": "quantile", "alpha": alpha},
        train_ds_q,
        num_boost_round=args.num_trees,
        valid_sets=[train_ds_q],
        callbacks=[lgb.record_evaluation({}), make_callback(q_log)],
    )
    preds_q  = q_model.predict(X_test)
    coverage = np.mean(y_test <= preds_q) * 100
    delta    = preds_q - y_test
    test_pinball = np.mean(np.where(delta >= 0, alpha * delta, (alpha - 1) * delta))
    print(
        f"Q(α={alpha:.1f})    test_pinball={test_pinball:.2f}"
        f"  coverage={coverage:.1f}%  (target {alpha*100:.0f}%)"
    )

    if args.rmse:
        print(f"\n{'iter':<6}  pinball(α={alpha:.1f})")
        print("-" * 26)
        for i, v in enumerate(q_log, 1):
            print(f"{i:<6}  {v:.4f}")

    if args.importance and alpha == 0.5:
        importance = q_model.feature_importance(importance_type="gain")
        ranked = sorted(zip(importance, feature_cols), reverse=True)
        print(f"\nFeature importance (gain) — Q(α={alpha:.1f}):")
        print(f"{'rank':<6}  {'gain':<14}  feature")
        print("-" * 38)
        for rank, (gain, name) in enumerate(ranked, 1):
            print(f"{rank:<6}  {gain:<14.0f}  {name}")
