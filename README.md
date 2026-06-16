# wembley

[![CI](https://github.com/awkward-crow/wembley/actions/workflows/ci.yml/badge.svg)](https://github.com/awkward-crow/wembley/actions/workflows/ci.yml)

A Rust implementation of the core [LightGBM](https://github.com/microsoft/LightGBM) training algorithm. Replicates the three key speed optimisations — histogram-based split finding, the histogram subtraction trick, and leaf-wise (best-first) tree growth — and supports L2 regression, binary classification, and quantile regression. Benchmarked against the Python LightGBM library across three datasets.

## Performance

Wall-clock time from process start to exit (100 trees, 63 leaves):

| dataset | samples | features | task | Rust | LightGBM |
|:--------|--------:|---------:|:-----|-----:|---------:|
| Bike Sharing | 731 | 11 | quantile regression | **0.52 s** | 0.89 s |
| California Housing | 20,640 | 8 | L2 regression | 0.94 s | 0.89 s |
| Cover Type | 495,141 | 54 | binary classification | 9.48 s | 3.02 s |

Rust leads on small datasets where LightGBM's startup cost dominates. The gap on Cover Type is a single algorithmic difference: the current implementation parallelises histogram construction over *features*, reading the data `num_features` times per leaf; LightGBM parallelises over *rows* and reads each sample exactly once. Model quality (RMSE / log-loss / pinball) is within 1–4% of LightGBM across all configurations. See [docs/performance.md](docs/performance.md) for a full analysis and the two targeted changes that would close the gap.

## What's implemented

- Histogram-based split finding (continuous features → u8 bin indices, up to 255 bins)
- Histogram subtraction trick (larger child = parent − smaller child; O(bins) not O(data))
- Leaf-wise (best-first) tree growth
- Rayon parallelism across features during histogram build and across leaves during split search
- L2 regression, binary logistic regression, quantile (pinball) regression
- Feature importance by gain (sum of split gains) and split count

## Setup

Requires Python 3 for the data-fetch scripts. Create a virtual environment from the repo root:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Examples

### California Housing — L2 regression

20,640 samples, 8 features. Predicts median house value (dollars).

**Fetch data** (first time only):

```sh
python examples/fetch_california.py --shuffle --seed=10331
```

**Run:**

```sh
cargo run --example california --release
```

| flag | effect |
|:-----|:-------|
| `--num_trees=N` | boosting rounds (default 50) |
| `--num_leaves=N` | max leaves per tree (default 31) |
| `--error` | print train RMSE after each tree |
| `--importance` | print feature importance table (gain) |
| `--importance-by-tree` | print per-tree feature importance |

Example with all output enabled (100 trees, 63 leaves):

```sh
cargo run --example california --release -- --num_trees=100 --num_leaves=63 --error --importance
```

At these settings: train RMSE ≈ 34 250, test RMSE ≈ 46 650 (dollar units). `MedInc` accounts for the majority of total split gain, followed by `Longitude` and `Latitude`.

### Bike Sharing — quantile regression

731 daily observations, 11 features. Fits three quantile models (α = 0.9, 0.5, 0.1) to produce upper bound, median, and lower bound predictions simultaneously.

**Fetch data** (first time only):

```sh
cargo run --example fetch_bike
```

**Run:**

```sh
cargo run --example bike --release
```

| flag | effect |
|:-----|:-------|
| `--num_trees=N` | boosting rounds (default 200) |
| `--num_leaves=N` | max leaves per tree (default 15) |
| `--shuffle` | random 80/20 split instead of chronological |
| `--error` | print per-iteration train metric |
| `--importance` | print feature importance for the median model |

### Cover Type — binary classification

495,141 samples, 54 features. Classifies Spruce/Fir (class 1) vs Lodgepole Pine (class 2).

**Fetch data** (first time only):

```sh
cargo run --example fetch_covertype
```

**Run:**

```sh
cargo run --example covertype --release
```

| flag | effect |
|:-----|:-------|
| `--num_trees=N` | boosting rounds (default 100) |
| `--num_leaves=N` | max leaves per tree (default 31) |
| `--logloss` | print train log-loss after each tree |
| `--importance` | print feature importance table (gain) |

At default settings: test log-loss ≈ 0.748, test accuracy ≈ 67.1%.

## Documentation

- [Architecture](docs/architecture.md) — module map and algorithm walkthrough (histogram build/subtract, leaf-wise growth, data partition, quantile leaf renew, borrow-checker friction)
- [Performance](docs/performance.md) — histogram scatter-accumulate, SIMD, feature-parallel vs row-parallel histogram build, and what would close the LightGBM gap
- [California Housing benchmark](docs/benchmark_california.md)
- [Bike Sharing benchmark](docs/benchmark_bike.md)
- [Cover Type benchmark](docs/benchmark_covertype.md)
