# Performance: Rust vs LightGBM

## Summary

There is no fundamental language-level barrier to matching LightGBM's speed in
Rust. Both compile through LLVM and produce native code of comparable quality.
The performance gap in the current benchmarks is partly algorithmic — a design
choice that can be changed — and partly a consequence of Rust's safety model
pushing the most impactful low-level optimisations into `unsafe` code.

## Where the gap stands

Across the three benchmark datasets:

| dataset        | samples  | rust time | py/lgbm time | ratio |
|:---------------|---------:|----------:|-------------:|------:|
| california     |   20 640 |     0.94s |        0.89s |  1.1× |
| bike           |      731 |     0.52s |        0.89s |  0.6× |
| covertype      |  495 141 |     9.48s |        3.02s |  3.1× |

_(t=100, l=63 in all cases)_

Rust wins on small datasets where LightGBM's Python/C++ startup overhead
dominates. The gap inverts and widens as dataset size grows. This is the
diagnostic: the bottleneck is **throughput inside the histogram builder**, not
latency or startup cost.

## The histogram scatter-accumulate loop

The single hottest path in the algorithm is `build_histogram`
(`src/histogram.rs:36–49`):

```rust
for &idx in leaf_indices {
    let i = idx as usize;
    let b = feature_bins[i] as usize;
    hist.data[b * 2]     += gradients[i] as f64;
    hist.data[b * 2 + 1] += hessians[i] as f64;
}
```

This is a **scatter-accumulate**: each iteration reads a bin index from
`feature_bins`, then writes to an arbitrary location in `hist.data`. The write
destinations are data-dependent and may repeat (multiple samples fall in the
same bin), so they are not independent.

Safe Rust cannot prove the writes are non-aliasing, which blocks
auto-vectorisation. The compiler emits a scalar loop. LightGBM's equivalent
loop in C++ is vectorised with explicit AVX2 intrinsics, processing 4–8 samples
per cycle. Writing the same in Rust requires `unsafe` with `std::arch`
intrinsics, or waiting for the portable SIMD API (`std::simd`) to stabilise.

## SIMD

`std::arch` (stable) exposes x86 AVX2 intrinsics but every call is `unsafe` and
the code is platform-specific. The portable SIMD API (`std::simd`) is cleaner
but has been nightly-only since its introduction. LightGBM has years of tuned
AVX2 paths for histogram accumulation, bin mapping, and leaf output computation.
Matching them in Rust is entirely possible but requires committing to `unsafe`
SIMD code, the same trade-off LightGBM's C++ authors accepted.

## Data-parallel histogram building

This is the biggest single gap, and it is **algorithmic**, not linguistic.

The current implementation (`src/boosting.rs`, `build_all_hists`) parallelises
over **features**: each Rayon thread builds the histogram for one feature
independently. This is clean and borrow-checker-friendly, but it means each
thread scans the full `leaf_indices` array once per feature — the data is read
`num_features` times.

LightGBM partitions **rows** across threads instead. Each thread receives a
contiguous slice of the leaf's sample indices and accumulates into a
thread-local partial histogram. The partial histograms are merged at the end.
Each sample is read exactly once regardless of the number of features, and the
merge is a cheap O(features × bins) addition.

The performance difference compounds with dataset size. At 20k samples and 8
features (california) it is negligible. At 495k samples and 54 features
(covertype) it is the dominant cost.

This pattern is expressible in Rust — thread-local partial histograms are
straightforward with `thread_local!` or explicit per-thread allocation — but
the merge step requires mutable access to multiple histogram slots
simultaneously, which pushes it into `unsafe` for the same aliasing reason as
`subtract_slots`.

## Borrow checker friction

Throughout the codebase, patterns that are trivial in C++ require either
restructuring or `unsafe` because the borrow checker cannot reason about
non-overlapping indices into a shared data structure at runtime.

The clearest example is `HistogramPool::subtract_slots`
(`src/histogram.rs:207–224`): computing `larger = parent - smaller` requires
simultaneous read access to two slots and write access to a third. The borrow
checker sees three mutable borrows of `self.slots` and rejects it. The fix is
unsafe raw pointer arithmetic, which is sound because the three indices are
asserted distinct, but the burden of proof falls on the programmer.

This pattern recurs whenever the hot path needs to alias into the same pool,
tree, or partition structure from multiple angles at once. It does not prevent
correct or fast code, but it raises the engineering cost of each such
optimisation.

## What would close the gap

Two changes would bring the Rust implementation within ~10–20% of LightGBM on
large datasets:

1. **Data-parallel histogram building.** Partition `leaf_indices` across Rayon
   threads, accumulate per-thread partial histograms, merge with unsafe
   pointer-based summation. This removes the `num_features` multiplier on data
   reads and is the single highest-leverage change.

2. **SIMD scatter-accumulate.** Replace the scalar loop in `build_histogram`
   with an AVX2 path using `std::arch`. A 4× throughput improvement on the
   inner loop is realistic. This requires `unsafe` but is isolated to one
   function of ~15 lines.

Both are engineering work, not language limitations. The `unsafe` boundary is
the cost of operating at this level in Rust — the same trade-off exists in any
systems language that gives the programmer control over memory layout and
vectorisation.
