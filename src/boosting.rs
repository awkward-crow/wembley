use rayon::prelude::*;

use crate::{
    config::Config,
    data_partition::DataPartition,
    dataset::Dataset,
    histogram::{build_histogram, find_best_split, Histogram, HistogramPool},
    objective::Objective,
    tree::Tree,
};

// ── Per-iteration callback ─────────────────────────────────────────────────

/// Called after each boosting round: `fn(iteration_index, metric_value)`.
pub type IterationCallback = Box<dyn Fn(usize, f64) + Send + Sync>;

// ── GBDT ──────────────────────────────────────────────────────────────────

pub struct GBDT {
    pub config: Config,
    pub trees: Vec<Tree>,
    /// Initial prediction (stored to allow correct external prediction).
    pub init_score: f64,
    /// Optional callback fired after each tree. Receives (0-based iter, metric).
    pub on_iteration: Option<IterationCallback>,
}

impl GBDT {
    pub fn new(config: Config) -> Self {
        Self { config, trees: Vec::new(), init_score: 0.0, on_iteration: None }
    }

    /// Train on `dataset` using `objective`.
    pub fn train(&mut self, dataset: &Dataset, objective: &dyn Objective) {
        if self.config.num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.config.num_threads)
                .build_global()
                .ok();
        }

        let n = dataset.num_data;
        self.init_score = objective.init_score(&dataset.labels);
        let mut scores = vec![self.init_score; n];
        let mut grads = vec![0.0f32; n];
        let mut hess  = vec![0.0f32; n];
        let lr = self.config.learning_rate;

        for iter in 0..self.config.num_trees {
            objective.gradients_hessians(&scores, &dataset.labels, &mut grads, &mut hess);

            let mut tree = train_one_tree(dataset, &grads, &hess, &self.config);

            if objective.needs_renew_leaf_output() {
                renew_leaf_outputs_quantile(&mut tree, dataset, &scores, objective);
            }

            // Accumulate lr * tree_prediction into scores
            let mut delta = vec![0.0f64; n];
            tree.predict_into(dataset, &mut delta);
            for i in 0..n {
                scores[i] += lr * delta[i];
            }

            self.trees.push(tree);

            if let Some(cb) = &self.on_iteration {
                let metric = objective.eval_metric(&scores, &dataset.labels);
                cb(iter, metric);
            }
        }
    }

    /// Predict for a dataset using all trained trees, starting from `init_score`.
    pub fn predict(&self, dataset: &Dataset) -> Vec<f64> {
        let lr = self.config.learning_rate;
        let mut scores = vec![self.init_score; dataset.num_data];
        let mut delta = vec![0.0f64; dataset.num_data];
        for tree in &self.trees {
            delta.fill(0.0);
            tree.predict_into(dataset, &mut delta);
            for (s, d) in scores.iter_mut().zip(delta.iter()) {
                *s += lr * d;
            }
        }
        scores
    }

    /// Gain-based feature importance per tree: one Vec<f64> per tree, indexed by feature.
    pub fn feature_importance_gain_per_tree(&self, num_features: usize) -> Vec<Vec<f64>> {
        self.trees.iter().map(|tree| {
            let mut importance = vec![0.0f64; num_features];
            for (node, &feat) in tree.split_feature.iter().enumerate() {
                importance[feat] += tree.split_gain[node];
            }
            importance
        }).collect()
    }

    /// Gain-based feature importance: sum of split gains over all trees.
    pub fn feature_importance_gain(&self, num_features: usize) -> Vec<f64> {
        let mut importance = vec![0.0f64; num_features];
        for tree in &self.trees {
            for (node, &feat) in tree.split_feature.iter().enumerate() {
                importance[feat] += tree.split_gain[node];
            }
        }
        importance
    }

    /// Split-count feature importance: how many times each feature was used.
    pub fn feature_importance_split(&self, num_features: usize) -> Vec<u32> {
        let mut counts = vec![0u32; num_features];
        for tree in &self.trees {
            for &feat in &tree.split_feature {
                counts[feat] += 1;
            }
        }
        counts
    }
}

// ── Serial tree learner ────────────────────────────────────────────────────

#[derive(Clone)]
struct LeafState {
    sum_grad: f64,
    sum_hess: f64,
    count: usize,
    hist_slot: usize,
    depth: usize,
}

fn train_one_tree(
    dataset: &Dataset,
    gradients: &[f32],
    hessians: &[f32],
    config: &Config,
) -> Tree {
    let nf = dataset.num_features;
    let n  = dataset.num_data;

    // Pool needs one slot per potential leaf index.
    // With num_leaves cap, max leaf index = 2*(num_leaves-1).
    let pool_size = config.num_leaves * 2;
    let num_bins: Vec<usize> = (0..nf).map(|f| dataset.num_bins(f)).collect();
    let mut pool = HistogramPool::new(nf, num_bins);
    pool.resize(pool_size);

    let mut partition = DataPartition::new(n, pool_size);

    // Root leaf stats
    let (root_grad, root_hess) = sum_grad_hess(partition.leaf_indices(0), gradients, hessians);
    let root_out = calc_leaf_output(root_grad, root_hess, config.lambda_l2);

    let mut tree = Tree::new_leaf(root_out);

    // Leaf state table
    let mut states: Vec<Option<LeafState>> = vec![None; pool_size];
    states[0] = Some(LeafState {
        sum_grad: root_grad,
        sum_hess: root_hess,
        count: n,
        hist_slot: 0,
        depth: 0,
    });

    // Build root histograms
    build_all_hists(dataset, partition.leaf_indices(0), gradients, hessians, pool.get_mut(0));

    let mut active: Vec<usize> = vec![0];

    for _ in 0..(config.num_leaves - 1) {
        if active.is_empty() { break; }

        // Find best split per leaf (parallel over leaves, serial over features within each)
        let mut best_per_leaf: Vec<Option<(usize, crate::histogram::SplitInfo)>> =
            vec![None; pool_size];
        let leaf_bests: Vec<(usize, Option<(usize, crate::histogram::SplitInfo)>)> = active
            .par_iter()
            .map(|&leaf| {
                let st = states[leaf].as_ref().unwrap();
                let hists = pool.get(st.hist_slot);
                let best = (0..nf)
                    .map(|f| (f, find_best_split(&hists[f], st.sum_grad, st.sum_hess, st.count, config)))
                    .filter(|(_, sp)| sp.gain.is_finite() && sp.gain > 0.0)
                    .max_by(|(_, a), (_, b)| a.gain.partial_cmp(&b.gain).unwrap_or(std::cmp::Ordering::Equal));
                (leaf, best)
            })
            .collect();
        for (leaf, best) in leaf_bests {
            best_per_leaf[leaf] = best;
        }

        // Leaf-wise: pick leaf with highest gain
        let best_leaf = match active.iter().copied().max_by(|&a, &b| {
            let ga = best_per_leaf[a].as_ref().map_or(f64::NEG_INFINITY, |(_, s)| s.gain);
            let gb = best_per_leaf[b].as_ref().map_or(f64::NEG_INFINITY, |(_, s)| s.gain);
            ga.partial_cmp(&gb).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Some(l) if best_per_leaf[l].is_some() => l,
            _ => break,
        };

        let (best_feat, best_split) = best_per_leaf[best_leaf].take().unwrap();
        if best_split.gain <= 0.0 { break; }

        // Depth guard
        let parent_depth = states[best_leaf].as_ref().unwrap().depth;
        if config.max_depth.map_or(false, |md| parent_depth >= md) {
            active.retain(|&l| l != best_leaf);
            continue;
        }

        let left_leaf  = tree.num_leaves;
        let right_leaf = tree.num_leaves + 1;

        let left_out  = calc_leaf_output(best_split.left_sum_grad,  best_split.left_sum_hess,  config.lambda_l2);
        let right_out = calc_leaf_output(best_split.right_sum_grad, best_split.right_sum_hess, config.lambda_l2);

        tree.split_leaf(best_leaf, best_feat, best_split.threshold, best_split.gain, left_out, right_out);

        // Partition data
        partition.split_leaf(
            best_leaf, left_leaf, right_leaf,
            &dataset.bins[best_feat],
            best_split.threshold,
        );

        let lc = partition.leaf_count(left_leaf);
        let rc = partition.leaf_count(right_leaf);
        let parent_slot = states[best_leaf].as_ref().unwrap().hist_slot;

        let (sm_leaf, lg_leaf, sm_grad, sm_hess, sm_cnt, lg_grad, lg_hess, lg_cnt) =
            if lc <= rc {
                (left_leaf,  right_leaf,
                 best_split.left_sum_grad,  best_split.left_sum_hess,  lc,
                 best_split.right_sum_grad, best_split.right_sum_hess, rc)
            } else {
                (right_leaf, left_leaf,
                 best_split.right_sum_grad, best_split.right_sum_hess, rc,
                 best_split.left_sum_grad,  best_split.left_sum_hess,  lc)
            };

        let sm_slot = sm_leaf;
        let lg_slot = lg_leaf;

        // Build smaller leaf's histograms
        build_all_hists(dataset, partition.leaf_indices(sm_leaf), gradients, hessians, pool.get_mut(sm_slot));

        // Larger leaf = parent − smaller  (O(#bins), histogram subtraction trick)
        pool.subtract_slots(parent_slot, sm_slot, lg_slot);

        states[sm_leaf] = Some(LeafState { sum_grad: sm_grad, sum_hess: sm_hess, count: sm_cnt, hist_slot: sm_slot, depth: parent_depth + 1 });
        states[lg_leaf] = Some(LeafState { sum_grad: lg_grad, sum_hess: lg_hess, count: lg_cnt, hist_slot: lg_slot, depth: parent_depth + 1 });
        states[best_leaf] = None;

        active.retain(|&l| l != best_leaf);
        active.push(sm_leaf);
        active.push(lg_leaf);
    }

    tree
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn build_all_hists(
    dataset: &Dataset,
    indices: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    hists: &mut Vec<Histogram>,
) {
    hists.par_iter_mut().enumerate().for_each(|(f, hist)| {
        build_histogram(&dataset.bins[f], indices, gradients, hessians, hist);
    });
}

fn sum_grad_hess(indices: &[u32], gradients: &[f32], hessians: &[f32]) -> (f64, f64) {
    indices.iter().fold((0.0f64, 0.0f64), |(g, h), &i| {
        let i = i as usize;
        (g + gradients[i] as f64, h + hessians[i] as f64)
    })
}

#[inline]
fn calc_leaf_output(sum_grad: f64, sum_hess: f64, lambda: f64) -> f64 {
    -sum_grad / (sum_hess + lambda)
}

/// Renew leaf outputs for quantile regression.
///
/// Each leaf's output is replaced by the alpha-quantile of (label - score)
/// for samples in that leaf, inferred via tree traversal.
fn renew_leaf_outputs_quantile(
    tree: &mut Tree,
    dataset: &Dataset,
    scores: &[f64],
    objective: &dyn Objective,
) {
    let alpha = objective.alpha().expect("renew_leaf_outputs_quantile called on non-quantile objective");
    let n = dataset.num_data;

    // Assign samples to leaves via traversal, collect residuals per leaf.
    let nf = dataset.num_features;
    let mut leaf_residuals: Vec<Vec<f64>> = vec![Vec::new(); tree.num_leaves];
    let mut bins = vec![0u8; nf];
    for i in 0..n {
        for f in 0..nf { bins[f] = dataset.bins[f][i]; }
        let leaf = find_leaf(tree, &bins);
        leaf_residuals[leaf].push(dataset.labels[i] as f64 - scores[i]);
    }

    for (leaf, residuals) in leaf_residuals.iter_mut().enumerate() {
        if residuals.is_empty() { continue; }
        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Match LightGBM: floor(alpha * n), clamped to valid range.
        let pos = ((alpha * residuals.len() as f64) as usize).min(residuals.len() - 1);
        tree.set_leaf_value(leaf, residuals[pos]);
    }
}

/// Traverse tree to find the leaf index for a sample.
fn find_leaf(tree: &Tree, bins: &[u8]) -> usize {
    if tree.split_feature.is_empty() { return 0; }
    let mut node = 0usize;
    loop {
        let bin = bins[tree.split_feature[node]];
        let child = if bin <= tree.split_threshold[node] {
            tree.left_child[node]
        } else {
            tree.right_child[node]
        };
        if child < 0 {
            let leaf = (-child - 1) as usize;
            if let Some(promoted) = tree.leaf_to_node[leaf] {
                node = promoted;
            } else {
                return leaf;
            }
        } else {
            node = child as usize;
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, dataset::Dataset, objective::RegressionL2};

    fn linear_dataset(n: usize) -> Dataset {
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        Dataset::from_columns(&[&x], &y, &Config::default(), None)
    }

    #[test]
    fn rmse_decreases() {
        let ds = linear_dataset(200);
        let obj = RegressionL2;
        let mut gbdt = GBDT::new(Config {
            num_trees: 30,
            num_leaves: 8,
            min_data_in_leaf: 5,
            learning_rate: 0.1,
            ..Config::default()
        });
        gbdt.train(&ds, &obj);
        let scores = gbdt.predict(&ds);
        let rmse = obj.eval_metric(&scores, &ds.labels);
        assert!(rmse < 0.05, "expected rmse < 0.05, got {:.4}", rmse);
    }
}
