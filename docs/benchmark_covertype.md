# Cover Type Benchmark

Rust (`examples/covertype.rs`) vs LightGBM Python (`examples/bench_covertype.py`)
across a grid of `num_trees` × `num_leaves`. Binary classification: class 1
(Spruce/Fir) vs class 2 (Lodgepole Pine), 495 141 samples, 54 features.
Chronological 80/20 split. Timing is wall-clock from process start to exit,
including data loading.

## Hyperparameters

Both implementations use identical settings:

| parameter         | value |
|:------------------|------:|
| min_data_in_leaf  |    20 |
| learning_rate     |   0.1 |
| lambda_l2         |   1.0 |
| max_bin           |   255 |
| min_gain_to_split |   0.0 |

## Reproduce

Build the Rust binary once (data download on first run):

```sh
cargo run --example fetch_covertype
cargo build --example covertype --release
```

Run Rust:

```sh
for t in 50 100; do for l in 31 63; do
  /usr/bin/time -f "real %e s" \
    ./target/release/examples/covertype --num_trees=$t --num_leaves=$l 2>&1
done; done
```

Run Python:

```sh
for t in 50 100; do for l in 31 63; do
  /usr/bin/time -f "real %e s" \
    .venv/bin/python examples/bench_covertype.py --num_trees=$t --num_leaves=$l 2>&1
done; done
```

## Results

| trees | leaves | rust logloss | rust acc% | rust time | py logloss | py acc% | py time |
|------:|-------:|-------------:|----------:|----------:|-----------:|--------:|--------:|
|    50 |     31 |       0.7282 |    66.49% |     4.16s |     0.7263 |  66.80% |   2.28s |
|    50 |     63 |       0.7687 |    67.11% |     5.07s |     0.7501 |  67.43% |   2.47s |
|   100 |     31 |       0.7542 |    67.09% |     7.24s |     0.7480 |  67.10% |   2.73s |
|   100 |     63 |       0.7910 |    67.38% |     9.48s |     0.7629 |  67.60% |   3.02s |
