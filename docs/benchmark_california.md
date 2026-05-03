# California Housing Benchmark

Rust (`examples/california.rs`) vs LightGBM Python (`examples/bench_california.py`)
across a grid of `num_trees` × `num_leaves`. RMSE values are scaled ×100 000
(dollar units for median house value). Timing is wall-clock from process start
to exit, including data loading.

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

Build the Rust binary once:

```sh
cargo build --example california --release
```

Run Rust:

```sh
for t in 50 100; do for l in 31 63; do
  /usr/bin/time -f "real %e s" \
    ./target/release/examples/california --num_trees=$t --num_leaves=$l 2>&1
done; done
```

Run Python:

```sh
for t in 50 100; do for l in 31 63; do
  /usr/bin/time -f "real %e s" \
    .venv/bin/python examples/bench_california.py --num_trees=$t --num_leaves=$l 2>&1
done; done
```

## Results

| trees | leaves | rust train | rust test | rust time | py train | py test | py time |
|------:|-------:|-----------:|----------:|----------:|---------:|--------:|--------:|
|    50 |     31 |      45091 |     50215 |     0.21s |    44804 |   49828 |   0.84s |
|    50 |     63 |      40349 |     48444 |     0.49s |    39797 |   48337 |   0.83s |
|   100 |     31 |      40120 |     47908 |     0.39s |    39649 |   47227 |   0.84s |
|   100 |     63 |      34249 |     46646 |     0.94s |    33685 |   46573 |   0.89s |

### end
