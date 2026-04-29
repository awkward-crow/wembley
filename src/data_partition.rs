/// Tracks which samples belong to each leaf, stored in a single contiguous buffer.
///
/// At initialisation all samples are in leaf 0. Splitting a leaf partitions its
/// slice in-place and records the two child ranges.
pub struct DataPartition {
    /// All sample indices, contiguously arranged by leaf.
    data: Vec<u32>,
    /// Start index in `data` for each leaf.
    leaf_start: Vec<usize>,
    /// Number of samples in each leaf.
    leaf_count: Vec<usize>,
}

impl DataPartition {
    /// Create a new partition with all `num_data` samples in leaf 0.
    pub fn new(num_data: usize, max_leaves: usize) -> Self {
        let data: Vec<u32> = (0..num_data as u32).collect();
        let mut leaf_start = vec![0usize; max_leaves];
        let mut leaf_count = vec![0usize; max_leaves];
        leaf_start[0] = 0;
        leaf_count[0] = num_data;
        Self { data, leaf_start, leaf_count }
    }

    /// Reset so all samples are back in leaf 0 (for the next tree).
    pub fn reset(&mut self, num_data: usize) {
        // Rebuild the index array
        for (i, v) in self.data.iter_mut().enumerate() {
            *v = i as u32;
        }
        self.leaf_start.fill(0);
        self.leaf_count.fill(0);
        self.leaf_count[0] = num_data;
    }

    /// Slice of sample indices for `leaf`.
    pub fn leaf_indices(&self, leaf: usize) -> &[u32] {
        let start = self.leaf_start[leaf];
        let count = self.leaf_count[leaf];
        &self.data[start..start + count]
    }

    /// Number of samples in `leaf`.
    pub fn leaf_count(&self, leaf: usize) -> usize {
        self.leaf_count[leaf]
    }

    /// Split `leaf` into two children (`left_leaf` and `right_leaf`) based on
    /// whether the sample's bin for `feature_bins` is `<= threshold`.
    ///
    /// The left child reuses the same start position; the right child begins
    /// immediately after. Returns (left_count, right_count).
    pub fn split_leaf(
        &mut self,
        leaf: usize,
        left_leaf: usize,
        right_leaf: usize,
        feature_bins: &[u8],
        threshold: u8,
    ) -> (usize, usize) {
        let start = self.leaf_start[leaf];
        let count = self.leaf_count[leaf];
        let slice = &mut self.data[start..start + count];

        // Two-pointer in-place partition: left ← bin<=threshold, right ← bin>threshold
        let mut lo = 0;
        let mut hi = count;
        while lo < hi {
            if feature_bins[slice[lo] as usize] <= threshold {
                lo += 1;
            } else {
                hi -= 1;
                slice.swap(lo, hi);
            }
        }
        let left_count = lo;
        let right_count = count - lo;

        self.leaf_start[left_leaf] = start;
        self.leaf_count[left_leaf] = left_count;
        self.leaf_start[right_leaf] = start + left_count;
        self.leaf_count[right_leaf] = right_count;

        (left_count, right_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_leaf_correct_counts() {
        // 6 samples, bins [0,0,1,1,2,2]
        let bins = vec![0u8, 0, 1, 1, 2, 2];
        let mut dp = DataPartition::new(6, 4);

        // Split leaf 0 → threshold 1: left={0,1,2,3}, right={4,5}
        let (l, r) = dp.split_leaf(0, 1, 2, &bins, 1);
        assert_eq!(l, 4);
        assert_eq!(r, 2);

        let left_idxs = dp.leaf_indices(1);
        assert!(left_idxs.iter().all(|&i| bins[i as usize] <= 1));

        let right_idxs = dp.leaf_indices(2);
        assert!(right_idxs.iter().all(|&i| bins[i as usize] > 1));
    }
}
