use crate::config::Config;

/// A flat histogram for one feature.
///
/// Layout (mirrors LightGBM's GET_GRAD / GET_HESS macros):
///   data[b*2]   = sum of gradients  for bin b
///   data[b*2+1] = sum of hessians   for bin b
pub struct Histogram {
    pub data: Vec<f64>,
    pub num_bins: usize,
}

impl Histogram {
    pub fn new(num_bins: usize) -> Self {
        Self { data: vec![0.0; num_bins * 2], num_bins }
    }

    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    #[inline]
    pub fn grad(&self, b: usize) -> f64 { self.data[b * 2] }
    #[inline]
    pub fn hess(&self, b: usize) -> f64 { self.data[b * 2 + 1] }
    #[inline]
    pub fn grad_mut(&mut self, b: usize) -> &mut f64 { &mut self.data[b * 2] }
    #[inline]
    pub fn hess_mut(&mut self, b: usize) -> &mut f64 { &mut self.data[b * 2 + 1] }
}

/// Accumulate (gradient, hessian) into bins for all samples in `leaf_indices`.
///
/// `feature_bins`: full column for this feature (length = num_data).
/// `leaf_indices`: indices of samples currently in this leaf.
pub fn build_histogram(
    feature_bins: &[u8],
    leaf_indices: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    hist: &mut Histogram,
) {
    hist.clear();
    for &idx in leaf_indices {
        let i = idx as usize;
        let b = feature_bins[i] as usize;
        hist.data[b * 2]     += gradients[i] as f64;
        hist.data[b * 2 + 1] += hessians[i] as f64;
    }
}

/// Compute `larger = parent - smaller` in O(#bins).
/// All three histograms must have the same `num_bins`.
pub fn subtract_histogram(parent: &Histogram, smaller: &Histogram, larger: &mut Histogram) {
    debug_assert_eq!(parent.num_bins, smaller.num_bins);
    debug_assert_eq!(parent.num_bins, larger.num_bins);
    for i in 0..parent.data.len() {
        larger.data[i] = parent.data[i] - smaller.data[i];
    }
}

/// The result of evaluating one potential split point.
#[derive(Debug, Clone)]
pub struct SplitInfo {
    /// Bin threshold: samples with bin <= threshold go left.
    pub threshold: u8,
    pub gain: f64,
    pub left_sum_grad: f64,
    pub left_sum_hess: f64,
    pub left_count: usize,
    pub right_sum_grad: f64,
    pub right_sum_hess: f64,
    pub right_count: usize,
}

impl SplitInfo {
    pub fn invalid() -> Self {
        Self {
            threshold: 0,
            gain: f64::NEG_INFINITY,
            left_sum_grad: 0.0,
            left_sum_hess: 0.0,
            left_count: 0,
            right_sum_grad: 0.0,
            right_sum_hess: 0.0,
            right_count: 0,
        }
    }
}

/// Compute the leaf gain: G² / (H + λ).
#[inline]
fn leaf_gain(sum_grad: f64, sum_hess: f64, lambda: f64) -> f64 {
    sum_grad * sum_grad / (sum_hess + lambda)
}

/// Scan the histogram left-to-right to find the best split threshold.
///
/// `sum_grad` / `sum_hess` / `num_data` are the totals for this leaf.
/// Returns `SplitInfo::invalid()` if no valid split exists.
pub fn find_best_split(
    hist: &Histogram,
    sum_grad: f64,
    sum_hess: f64,
    num_data: usize,
    config: &Config,
) -> SplitInfo {
    let lambda = config.lambda_l2;
    let parent_gain = leaf_gain(sum_grad, sum_hess, lambda);
    let min_hess = lambda; // proxy for min data: skip bins with negligible hessian

    let mut best = SplitInfo::invalid();

    let mut left_grad = 0.0f64;
    let mut left_hess = 0.0f64;
    let mut left_cnt = 0usize;

    // We need a count-per-bin estimate. For constant hessian (=1.0 per sample)
    // hess == count. For other objectives, approximate count = round(hess / mean_hess).
    // We'll track count separately using the hessian ratio approach from LightGBM.
    let mean_hess = if num_data > 0 { sum_hess / num_data as f64 } else { 1.0 };

    // Don't split on the last bin — right side would be empty.
    for b in 0..(hist.num_bins - 1) {
        left_grad += hist.grad(b);
        left_hess += hist.hess(b);
        let bin_cnt = (hist.hess(b) / mean_hess).round() as usize;
        left_cnt += bin_cnt;

        // Skip bins with too little hessian (proxy for min_data_in_leaf)
        if left_hess < min_hess {
            continue;
        }

        let right_grad = sum_grad - left_grad;
        let right_hess = sum_hess - left_hess;
        let right_cnt = num_data.saturating_sub(left_cnt);

        if right_hess < min_hess {
            continue;
        }

        // Guard min_data_in_leaf
        if left_cnt < config.min_data_in_leaf || right_cnt < config.min_data_in_leaf {
            continue;
        }

        let gain = leaf_gain(left_grad, left_hess, lambda)
            + leaf_gain(right_grad, right_hess, lambda)
            - parent_gain;

        if gain > config.min_gain_to_split && gain > best.gain {
            best = SplitInfo {
                threshold: b as u8,
                gain,
                left_sum_grad: left_grad,
                left_sum_hess: left_hess,
                left_count: left_cnt,
                right_sum_grad: right_grad,
                right_sum_hess: right_hess,
                right_count: right_cnt,
            };
        }
    }

    best
}

/// Pool of pre-allocated histograms, one slot per leaf, reused across iterations.
///
/// Indexed by leaf id. Grows on demand.
pub struct HistogramPool {
    /// `slots[leaf][feature]`
    slots: Vec<Vec<Histogram>>,
    num_features: usize,
    num_bins_per_feature: Vec<usize>,
}

impl HistogramPool {
    pub fn new(num_features: usize, num_bins_per_feature: Vec<usize>) -> Self {
        Self { slots: Vec::new(), num_features, num_bins_per_feature }
    }

    /// Ensure the pool has at least `num_leaves` slots.
    pub fn resize(&mut self, num_leaves: usize) {
        while self.slots.len() < num_leaves {
            let hists: Vec<Histogram> = (0..self.num_features)
                .map(|f| Histogram::new(self.num_bins_per_feature[f]))
                .collect();
            self.slots.push(hists);
        }
    }

    /// Get mutable reference to the histogram array for a slot.
    pub fn get_mut(&mut self, slot: usize) -> &mut Vec<Histogram> {
        &mut self.slots[slot]
    }

    /// Get shared reference to the histogram array for a slot.
    pub fn get(&self, slot: usize) -> &Vec<Histogram> {
        &self.slots[slot]
    }

    /// Compute `slots[larger] = slots[parent] - slots[smaller]` for all features.
    ///
    /// All three slot indices must be distinct.
    pub fn subtract_slots(&mut self, parent: usize, smaller: usize, larger: usize) {
        debug_assert!(parent != smaller && parent != larger && smaller != larger);
        let ptr = self.slots.as_mut_ptr();
        // SAFETY: all three indices are distinct (asserted above) and in-bounds.
        unsafe {
            let parent_hists  = &*ptr.add(parent);
            let smaller_hists = &*ptr.add(smaller);
            let larger_hists  = &mut *ptr.add(larger);
            for f in 0..larger_hists.len() {
                let ld = &mut larger_hists[f].data;
                let pd = &parent_hists[f].data;
                let sd = &smaller_hists[f].data;
                for i in 0..ld.len() {
                    ld[i] = pd[i] - sd[i];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> Config {
        Config { min_data_in_leaf: 1, lambda_l2: 0.0, min_gain_to_split: 0.0, ..Config::default() }
    }

    #[test]
    fn build_and_subtract() {
        // 4 samples, 1 feature, 3 bins: [0,0,1,2]
        let bins = vec![0u8, 0, 1, 2];
        let grads = vec![1.0f32, 2.0, 3.0, 4.0];
        let hess  = vec![1.0f32; 4];
        let indices = vec![0u32, 1, 2, 3];

        let mut parent = Histogram::new(3);
        build_histogram(&bins, &indices, &grads, &hess, &mut parent);

        // bin 0: grad=3, hess=2; bin 1: grad=3, hess=1; bin 2: grad=4, hess=1
        assert!((parent.grad(0) - 3.0).abs() < 1e-9);
        assert!((parent.grad(1) - 3.0).abs() < 1e-9);
        assert!((parent.grad(2) - 4.0).abs() < 1e-9);

        // Smaller leaf = samples 0,1 (bin 0)
        let small_indices = vec![0u32, 1];
        let mut smaller = Histogram::new(3);
        build_histogram(&bins, &small_indices, &grads, &hess, &mut smaller);

        let mut larger = Histogram::new(3);
        subtract_histogram(&parent, &smaller, &mut larger);

        // larger should equal histogram of samples 2,3
        assert!((larger.grad(1) - 3.0).abs() < 1e-9);
        assert!((larger.grad(2) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn find_split_basic() {
        // 6 samples, 2 bins. Samples 0-2 in bin 0 (label -1), samples 3-5 in bin 1 (label +1).
        // Perfect split at threshold 0.
        let cfg = make_config();
        let mut hist = Histogram::new(2);
        // bin 0: grad = -3 (scores 0, labels 1 → grad = -1 each under L2)
        hist.data[0] = -3.0; hist.data[1] = 3.0; // bin 0: grad, hess
        hist.data[2] =  3.0; hist.data[3] = 3.0; // bin 1: grad, hess
        let split = find_best_split(&hist, 0.0, 6.0, 6, &cfg);
        assert!(split.gain > 0.0);
        assert_eq!(split.threshold, 0);
    }
}
