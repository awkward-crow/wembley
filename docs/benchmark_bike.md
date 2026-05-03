# Bike Sharing Benchmark

Rust (`examples/bike.rs`) vs LightGBM Python (`examples/bench_bike.py`)
across a grid of `num_trees` × `num_leaves`. Three quantile models are run:
α=0.9, α=0.5 (median), α=0.1. Chronological 80/20 split (no shuffle), so
both implementations see identical train and test sets.

Timing is wall-clock from process start to exit, including data loading.

Both implementations now use the same leaf-output formula (α-quantile of
per-leaf residuals, index `floor(α·n)`), and test pinball is computed on the
held-out set in both cases. Results are within ~3% across all configs.

## Hyperparameters

Both implementations use identical settings:

| parameter         | value |
|:------------------|------:|
| min_data_in_leaf  |    10 |
| learning_rate     |  0.05 |
| lambda_l2         |   1.0 |
| max_bin           |   255 |
| min_gain_to_split |   0.0 |

## Reproduce

Build the Rust binary once:

```sh
cargo build --example bike --release
```

Run Rust:

```sh
for t in 50 100; do for l in 31 63; do
  /usr/bin/time -f "real %e s" \
    ./target/release/examples/bike --num_trees=$t --num_leaves=$l 2>&1
done; done
```

Run Python:

```sh
for t in 50 100; do for l in 31 63; do
  /usr/bin/time -f "real %e s" \
    .venv/bin/python examples/bench_bike.py --num_trees=$t --num_leaves=$l 2>&1
done; done
```

## Results

### Q(α=0.9) — upper bound (target coverage: 90%)

| trees | leaves | rust pinball | rust cov% | py pinball | py cov% | rust time | py time |
|------:|-------:|-------------:|----------:|-----------:|--------:|----------:|--------:|
|    50 |     31 |       637.64 |     56.5% |     671.27 |   58.5% |     0.17s |   0.82s |
|    50 |     63 |       637.64 |     56.5% |     671.27 |   58.5% |     0.20s |   0.82s |
|   100 |     31 |       522.03 |     47.6% |     527.81 |   49.0% |     0.40s |   0.88s |
|   100 |     63 |       522.03 |     47.6% |     546.64 |   49.7% |     0.52s |   0.89s |

### Q(α=0.5) — median (target coverage: 50%)

| trees | leaves | rust pinball | rust cov% | py pinball | py cov% |
|------:|-------:|-------------:|----------:|-----------:|--------:|
|    50 |     31 |       499.29 |     21.8% |     517.06 |   23.1% |
|    50 |     63 |       496.76 |     19.7% |     534.39 |   22.4% |
|   100 |     31 |       419.42 |     23.8% |     436.25 |   24.5% |
|   100 |     63 |       426.64 |     23.1% |     459.39 |   23.1% |

### Q(α=0.1) — lower bound (target coverage: 10%)

| trees | leaves | rust pinball | rust cov% | py pinball | py cov% |
|------:|-------:|-------------:|----------:|-----------:|--------:|
|    50 |     31 |      2361.46 |      8.8% |    2304.31 |    8.8% |
|    50 |     63 |      2361.46 |      8.8% |    2304.31 |    8.8% |
|   100 |     31 |      1509.89 |     10.2% |    1477.31 |   11.6% |
|   100 |     63 |      1487.92 |     10.2% |    1465.32 |   11.6% |
