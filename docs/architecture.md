# Architecture

## Module map

```
src/
  config.rs          hyperparameters (Config struct)
  bin_mapper.rs      continuous → u8 bin index (quantile cut-points)
  dataset.rs         column-major binned dataset
  histogram.rs       histogram build / subtract / split-finding; HistogramPool
  data_partition.rs  tracks which samples belong to each leaf
  tree.rs            decision tree with leaf-wise promotion map
  objective/
    mod.rs           Objective trait
    regression.rs    L2 regression
    binary.rs        binary logistic regression
    quantile.rs      quantile (pinball) regression
  boosting.rs        GBDT training loop + SerialTreeLearner
```


## Key algorithms and design decisions

### 1. Histogram-based split finding

The central reason LightGBM is faster than earlier GBDT implementations (e.g. the pre-sort
algorithm in early XGBoost) is that it replaces an O(#data × #features) scan at every split
with a two-phase approach:

**Phase 1 — Build** (O(#data_in_leaf) per feature):
For each sample in the current leaf, look up its pre-computed bin index for the feature being
considered and accumulate its gradient and hessian into that bin's running totals. This is a
simple scatter operation over a compact array.

**Phase 2 — Scan** (O(#bins) per feature):
Walk the histogram left to right, maintaining a prefix sum of (gradient, hessian). At each
bin boundary compute the split gain using the standard XGBoost gain formula:

```
gain = G_L² / (H_L + λ) + G_R² / (H_R + λ) − G_P² / (H_P + λ)
```

where G is the sum of gradients, H is the sum of hessians, and λ is L2 regularisation.
Because #bins is at most 255 (u8 storage), this scan is extremely fast and cache-friendly
regardless of how large the dataset is.

The histogram layout mirrors LightGBM's `GET_GRAD` / `GET_HESS` macros: gradient and hessian
for bin `b` are interleaved at positions `[b*2]` and `[b*2+1]` in a flat `Vec<f64>`. This
keeps the two values that are always read together on the same cache line.

### 2. Feature binning (`bin_mapper.rs`)

Before any training, every continuous feature value is mapped to a `u8` bin index (0–254) via
quantile-based cut-points. The `BinMapper` sorts the distinct values, picks up to `max_bin`
evenly-spaced quantile boundaries, and stores the upper bound of each bin. At runtime,
`map_value` does a binary search over the boundary array to find the bin.

The `u8` representation is load-bearing for performance. Storing 20,000 samples × 8 features
as bytes rather than f64s reduces the dataset footprint by 8× and makes the histogram build
loop cache-efficient: the entire feature column fits in L1/L2 cache.

### 3. The histogram subtraction trick

After a leaf is split into two children, LightGBM only builds histograms for the *smaller*
child from scratch. The larger child's histograms are obtained for free by subtracting the
smaller child's histograms from the parent's:

```
hist[larger] = hist[parent] − hist[smaller]
```

This subtraction is O(#bins) per feature, which is negligible. Building from scratch would
be O(#data_in_larger_leaf), which on average is more than half the parent's data. The
subtraction trick therefore roughly halves histogram construction cost at every level of
the tree.

The parent histograms are preserved in the `HistogramPool` until both children have been
processed. Pool slots are indexed by leaf ID, which are stable integers that grow monotonically
as the tree is built.

### 4. Leaf-wise (best-first) tree growth

Standard GBDT implementations grow trees level by level. LightGBM instead always splits
whichever *leaf* currently has the highest gain, regardless of which level it is on. This
produces asymmetric trees that concentrate capacity where the data is most complex, achieving
lower loss for the same number of leaves compared to a symmetric tree of the same depth.

In `boosting.rs`, the active leaf set is maintained as a `Vec<usize>`. After each split,
both child leaves are added to the set. At the start of each round, split candidates are
computed for all active leaves in parallel, then the leaf with the globally highest gain is
selected for splitting. The optional `max_depth` parameter can cap tree depth, but growth
within that cap remains leaf-wise.

### 5. Tree representation and traversal (`tree.rs`)

The tree uses a split node array (parallel `Vec`s for feature, threshold, gain, left_child,
right_child) alongside a flat `leaf_output` array. Child pointers encode leaves as negative
integers: leaf `k` → `-(k+1)`.

The key complication with leaf-wise growth is that a leaf referenced by an existing child
pointer may later be *promoted* to an internal node when it is itself split. A companion
`leaf_to_node: Vec<Option<usize>>` array tracks these promotions. During traversal, whenever
a child pointer resolves to a leaf, the promotion map is consulted before returning the leaf
output. This avoids having to update all existing child pointers when a leaf is split, which
would require a parent-pointer scan.

The root is always the first split (node 0) because at the start of training there is only one
leaf, so it must be split first. All subsequent traversals begin at node 0.

### 6. Data partition (`data_partition.rs`)

Sample indices are held in a single contiguous `Vec<u32>` with per-leaf start and count
metadata. Splitting a leaf is an in-place two-pointer partition of its slice: left pointer
advances while the bin is ≤ threshold, right pointer retreats while it is >, and the two swap
when they meet. This is O(#data_in_leaf) with zero allocation and good cache behaviour because
the indices being swapped are adjacent in memory.

Using `u32` rather than `usize` for indices halves their memory footprint (assuming datasets
under 4 billion rows, which is a reasonable assumption for single-machine training).

### 7. Objective functions and the GBDT loop

The `Objective` trait has four required methods:

- `gradients_hessians` — compute first and second derivatives of the loss
- `init_score` — the initial prediction before any trees are added
- `eval_metric` — the scalar training metric reported after each round
- `metric_name` — short string for display ("rmse", "binary_logloss", "pinball")

And two optional overrides:

- `needs_renew_leaf_output` — whether leaf outputs should be replaced after the tree
  structure is fixed (used only by quantile regression, described below)
- `alpha` — returns `Some(α)` for quantile objectives, `None` for all others; used by
  `renew_leaf_outputs_quantile` to avoid inferring α from the gradient values

**L2 regression**: gradients are residuals (`score − label`), hessians are 1. Initial score
is the mean label. This is the simplest case and the one used to validate the histogram
machinery.

**Binary logistic regression**: `p = sigmoid(score)`, gradient = `p − label`, hessian =
`p(1−p)`. Initial score is the log-odds of the base rate. Hessians are clamped to 1e-16 to
avoid degenerate splits when predictions saturate.

**Quantile regression**: The gradient is a step function of the sign of the residual:
`+(1−α)` if `score ≥ label`, `−α` if `score < label`. Hessians are 1 throughout. The tree
*structure* is therefore learned using these constant-magnitude gradient proxies, which is
valid because the gain formula only needs gradients to be informative about which direction
to split, not to encode the exact magnitude of the error.

The leaf *outputs*, however, must be the α-quantile of the residuals in each leaf, not the
standard `−G/(H+λ)` formula (which would produce the mean). After each tree is grown,
`renew_leaf_outputs_quantile` traverses the dataset, assigns each sample to its leaf, collects
the residuals `(label − score)` for that leaf, sorts them, and takes the value at position
`floor(n × α)` (clamped to `n−1`). This matches LightGBM's `RenewTreeOutput` /
`IsRenewTreeOutput` mechanism in `regression_objective.hpp`.

The initial score for quantile regression is the α-quantile of the training labels, consistent
with the same logic.

### 8. Feature importance

Two importance measures are provided on `GBDT`:

- **Gain importance** (`feature_importance_gain`): the sum of split gains across all nodes in
  all trees where a feature was used. This is the default in LightGBM and captures how much
  each feature actually reduced the loss. Stored as `split_gain: Vec<f64>` on each `Tree` at
  the time the split is made.

- **Split count** (`feature_importance_split`): the number of times a feature appears as a
  split across all trees. Cheaper to compute and less noisy, but blind to gain magnitude.

Both return a `Vec` indexed by feature ID. The examples sort by gain descending and display
the feature name alongside the score.

### 9. Parallelism

Rayon is used in two places:

- **Histogram construction**: feature histograms for a leaf are built in parallel across
  features (`par_iter_mut` over the histogram array). Each feature's histogram is independent.

- **Split candidate search**: active leaves are processed in parallel (`par_iter`). Within
  each leaf, features are scanned serially to find the best split, yielding one
  `Option<(feature, SplitInfo)>` per leaf. The results are written into the `best_per_leaf`
  table (indexed by leaf ID) in a serial post-pass.

These two parallel regions cover the two innermost loops of the training algorithm and
represent the bulk of wall-clock time.

The number of threads is controlled by `Config::num_threads` (0 = use all available cores).

Note: the current parallelism strategy — splitting work across *features* rather than *rows*
— is the primary reason the implementation is slower than LightGBM on large datasets.
LightGBM partitions rows across threads and merges thread-local partial histograms, reading
each sample once regardless of the number of features. See `docs/performance.md` for a full
discussion.


## Out of scope

The following features are present in LightGBM but not implemented here. They are noted
for completeness and as a roadmap for future work.

**Distributed and parallel tree learners.** LightGBM supports feature-parallel, data-parallel,
and voting-parallel training across multiple machines. Feature-parallel partitions features
across workers and communicates best splits; data-parallel partitions rows and uses Reduce
Scatter to merge histograms. These require network communication (MPI or socket) and are
orthogonal to the single-machine optimisations implemented here.

**GPU acceleration.** LightGBM has a GPU tree learner (and a separate CUDA learner) that
offloads histogram construction to the GPU. The GPU kernel maps naturally to the scatter
pattern of histogram building. Not implemented here.

**Quantized gradients.** LightGBM optionally discretises gradients and hessians to int8 or
int16 before building histograms. Packed int16 values (gradient in the high byte, hessian in
the low byte) allow histograms to be accumulated as integer sums and the communication cost
in distributed training to be halved. The histogram arithmetic is more complex (packing,
unpacking, fixed-point scaling) and is not implemented here.

**Categorical feature splits.** LightGBM handles categorical features by sorting histogram
bins by `sum_gradient / sum_hessian` and finding the best contiguous subset, achieving
O(k log k) split finding for k categories rather than the naive 2^(k−1) enumeration. The
current implementation assumes all features are continuous.

**L1 regularisation.** The gain formula is extended with a soft-threshold on the gradient
sum: `G_effective = sign(G) × max(0, |G| − λ_l1)`. Not implemented; only L2 is present.

**DART (Dropout Additive Regression Trees).** Trees are randomly dropped during training
to reduce overfitting. Requires tracking which trees are active in each round and rescaling
the remaining trees' contributions. Not implemented.

**Bagging and feature subsampling.** LightGBM supports training each tree on a random row
subsample (`bagging_fraction`) and a random column subsample (`feature_fraction`). The column
sampler infrastructure exists in LightGBM (`col_sampler.hpp`) but is not implemented here.

**Monotone constraints.** Constraints of the form "feature f's effect must be non-decreasing"
are enforced by propagating bounds through the tree and clipping split candidates that would
violate them. Not implemented.

**Path smoothing.** Leaf outputs can be smoothed toward the parent node's output, controlled
by `path_smooth`. This reduces variance in leaves with few samples. Not implemented.

**Multi-class classification.** Softmax objective with one tree per class per round. Not
implemented; only binary classification is supported.

**LambdaRank / NDCG.** Learning-to-rank objective. Not implemented.

**Early stopping.** The GBDT loop does not yet monitor a validation metric and stop when it
stops improving. The per-iteration callback provides all the data needed to implement this
externally, but there is no built-in mechanism.



### end
