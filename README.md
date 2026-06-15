# wembley

A rust port of the gradient boosting machine library LightGBM. The goal is to replicate the key speed
optimisations — histogram-based split finding, the histogram subtraction trick and leaf-wise
tree growth. It supports regression, binary classification and quantile regression.

## An example -- California Housing 

Gradient-boosted regression on the sklearn California Housing dataset (20,640 samples, 8 features).
Prints RMSE and a feature importance table at the end.

**set up a virtual environment**

Requires Python 3. From the project root:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**fetch the data**

```sh
python examples/fetch_california.py --shuffle --seed=10331
```

This writes `data/california_housing.csv`.

**Run the example**

```sh
cargo run --example california --release
```

By default prints a single summary line. Optional flags:

| flag | effect |
|---|---|
| `--num_trees=N` | number of boosting rounds (default 50) |
| `--num_leaves=N` | max leaves per tree (default 31) |
| `--error` | print train error after each tree |
| `--importance` | print feature importance table (gain, summed over all trees) |
| `--importance-by-tree` | print a feature importance table for each tree individually |

Example with all output:

```sh
cargo run --example california --release -- --num_trees=100 --error --importance
```

## Benchmarks

Benchmarks of this implementation against LightGBM can be found in dir. `docs`.

## Architecture and performance

see `architecture.md` and `performance.md` in dir. `docs`.


### end
