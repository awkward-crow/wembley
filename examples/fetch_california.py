#! /bin/env python

"""
Exports sklearn's California Housing dataset to data/california_housing.csv.

Run with:  python examples/fetch_california.py [--shuffle [--seed=SEED]]
           (from the repo root, with the venv active or via examples/.venv/bin/python)
"""
import argparse
import os
from pathlib import Path

from sklearn.datasets import fetch_california_housing

parser = argparse.ArgumentParser()
parser.add_argument("--shuffle", action="store_true", help="randomly shuffle rows before writing")
parser.add_argument("--seed", type=int, default=42, help="random seed for shuffle (default: 42)")
args = parser.parse_args()

os.makedirs("data", exist_ok=True)
out = Path("data/california_housing.csv")

ds = fetch_california_housing(as_frame=True)
df = ds.frame  # 8 feature columns + MedHouseVal

if args.shuffle:
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

df.to_csv(out, index=False)
print(f"wrote {out}  ({len(df)} rows, {len(df.columns)-1} features + target)")
print(f"columns: {', '.join(df.columns)}")
if args.shuffle:
    print(f"shuffled  (seed={args.seed})")
