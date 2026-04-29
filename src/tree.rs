use crate::dataset::Dataset;

/// A single decision tree built by the GBDT learner.
///
/// Nodes and leaves are managed separately. Each leaf may later be promoted to
/// an internal split node; `leaf_to_node[leaf]` records this promotion so
/// traversal can follow the chain correctly.
///
/// Leaf encoding in child arrays: leaf k → -(k as i32 + 1).
pub struct Tree {
    // ── Internal nodes ────────────────────────────────────────────────────
    /// Feature index at each node.
    pub split_feature: Vec<usize>,
    /// Bin threshold: samples with bin <= threshold go left.
    pub split_threshold: Vec<u8>,
    /// Information gain stored at each node (for feature importance).
    pub split_gain: Vec<f64>,
    /// Left child: >= 0 → node index, < 0 → leaf (decode: -v - 1).
    pub left_child: Vec<i32>,
    /// Right child: same encoding.
    pub right_child: Vec<i32>,

    // ── Leaves ────────────────────────────────────────────────────────────
    /// Prediction value for each leaf.
    pub leaf_output: Vec<f64>,

    // ── Promotion map ────────────────────────────────────────────────────
    /// `leaf_to_node[leaf]` = Some(node_idx) once that leaf has been split.
    pub leaf_to_node: Vec<Option<usize>>,

    pub num_leaves: usize,
}

impl Tree {
    /// Create a tree containing only the root leaf with the given output.
    pub fn new_leaf(output: f64) -> Self {
        Self {
            split_feature: Vec::new(),
            split_threshold: Vec::new(),
            split_gain: Vec::new(),
            left_child: Vec::new(),
            right_child: Vec::new(),
            leaf_output: vec![output],
            leaf_to_node: vec![None],
            num_leaves: 1,
        }
    }

    /// Split `leaf_idx` into an internal node.
    ///
    /// Allocates two new leaves. Returns (left_leaf, right_leaf).
    pub fn split_leaf(
        &mut self,
        leaf_idx: usize,
        feature: usize,
        threshold: u8,
        gain: f64,
        left_output: f64,
        right_output: f64,
    ) -> (usize, usize) {
        let node_idx = self.split_feature.len();
        let left_leaf = self.num_leaves;
        let right_leaf = self.num_leaves + 1;

        // Add the new internal node
        self.split_feature.push(feature);
        self.split_threshold.push(threshold);
        self.split_gain.push(gain);
        self.left_child.push(leaf_encode(left_leaf));
        self.right_child.push(leaf_encode(right_leaf));

        // Extend leaves
        self.leaf_output.push(left_output);
        self.leaf_output.push(right_output);
        self.leaf_to_node.push(None);
        self.leaf_to_node.push(None);

        // Mark old leaf as promoted to this node
        self.leaf_to_node[leaf_idx] = Some(node_idx);

        self.num_leaves += 2;
        (left_leaf, right_leaf)
    }

    /// Predict for a single sample given its per-feature bin indices.
    pub fn predict_one(&self, sample_bins: &[u8]) -> f64 {
        if self.split_feature.is_empty() {
            return self.leaf_output[0];
        }
        // Start at the root = node 0 (always the first split performed)
        // — the root leaf (leaf 0) must always be split first because it's the
        // only leaf at the start.
        self.traverse_node(0, sample_bins)
    }

    fn traverse_node(&self, node: usize, bins: &[u8]) -> f64 {
        let bin = bins[self.split_feature[node]];
        let child = if bin <= self.split_threshold[node] {
            self.left_child[node]
        } else {
            self.right_child[node]
        };

        if child >= 0 {
            // Another internal node
            self.traverse_node(child as usize, bins)
        } else {
            // A leaf — but it may have been promoted to a node later
            let leaf = leaf_decode(child);
            if let Some(promoted_node) = self.leaf_to_node[leaf] {
                self.traverse_node(promoted_node, bins)
            } else {
                self.leaf_output[leaf]
            }
        }
    }

    /// Add tree predictions to `out` for all samples in `dataset`.
    pub fn predict_into(&self, dataset: &Dataset, out: &mut [f64]) {
        let nf = dataset.num_features;
        let mut sample_bins = vec![0u8; nf];
        for i in 0..dataset.num_data {
            for f in 0..nf {
                sample_bins[f] = dataset.bins[f][i];
            }
            out[i] += self.predict_one(&sample_bins);
        }
    }

    /// Set a leaf's output value (used for quantile leaf renewal).
    pub fn set_leaf_value(&mut self, leaf: usize, value: f64) {
        self.leaf_output[leaf] = value;
    }
}

#[inline]
fn leaf_encode(leaf: usize) -> i32 {
    -(leaf as i32 + 1)
}

#[inline]
fn leaf_decode(encoded: i32) -> usize {
    (-encoded - 1) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stump_predict() {
        let t = Tree::new_leaf(3.14);
        assert_eq!(t.predict_one(&[0, 1, 2]), 3.14);
    }

    #[test]
    fn one_split() {
        // Feature 0, threshold bin 1: left → 1.0, right → 2.0
        let mut t = Tree::new_leaf(0.0);
        t.split_leaf(0, 0, 1, 10.0, 1.0, 2.0);
        // bin 0 → left (1.0)
        assert_eq!(t.predict_one(&[0]), 1.0);
        // bin 1 → left (threshold = 1 means <=1 goes left)
        assert_eq!(t.predict_one(&[1]), 1.0);
        // bin 2 → right (2.0)
        assert_eq!(t.predict_one(&[2]), 2.0);
    }

    #[test]
    fn two_splits_leaf_wise() {
        // Split root (leaf 0) → leaves 1, 2.
        // Then split leaf 2 → leaves 3, 4.
        let mut t = Tree::new_leaf(0.0);
        t.split_leaf(0, 0, 1, 10.0, 1.0, 0.0); // leaf 0 → node 0, leaves 1,2
        t.split_leaf(2, 0, 3, 5.0, 3.0, 4.0);  // leaf 2 → node 1, leaves 3,4

        // bin 0 → node0.left → leaf 1 (not promoted) → 1.0
        assert_eq!(t.predict_one(&[0]), 1.0);
        // bin 2 → node0.right → leaf 2 (promoted to node 1)
        //   → node1: bin 2 <= 3 → leaf 3 → 3.0
        assert_eq!(t.predict_one(&[2]), 3.0);
        // bin 4 → node0.right → leaf 2 (promoted to node 1)
        //   → node1: bin 4 > 3 → leaf 4 → 4.0
        assert_eq!(t.predict_one(&[4]), 4.0);
    }
}
