/// Hyperparameters for GBDT training.
#[derive(Debug, Clone)]
pub struct Config {
    /// Maximum number of leaves per tree.
    pub num_leaves: usize,
    /// Optional hard cap on tree depth (leaf-wise growth still applies within this).
    pub max_depth: Option<usize>,
    /// Minimum number of samples required in a leaf for a split to be accepted.
    pub min_data_in_leaf: usize,
    /// Number of boosting rounds (trees).
    pub num_trees: usize,
    /// Learning rate / shrinkage applied to each tree's output.
    pub learning_rate: f64,
    /// Maximum number of histogram bins per feature.
    pub max_bin: usize,
    /// L2 regularisation on leaf weights.
    pub lambda_l2: f64,
    /// Minimum gain required to accept a split.
    pub min_gain_to_split: f64,
    /// Number of Rayon threads (0 = use all available).
    pub num_threads: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_leaves: 31,
            max_depth: None,
            min_data_in_leaf: 20,
            num_trees: 100,
            learning_rate: 0.1,
            max_bin: 255,
            lambda_l2: 1e-2,
            min_gain_to_split: 0.0,
            num_threads: 0,
        }
    }
}
