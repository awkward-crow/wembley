/// Maps continuous f64 feature values to discrete u8 bin indices using
/// quantile-based cut-points. Mirrors LightGBM's BinMapper in bin.h.
#[derive(Debug, Clone)]
pub struct BinMapper {
    /// Upper bound of each bin. `bin_upper_bound[i]` is the maximum value
    /// that maps to bin `i`. The last entry is +infinity.
    pub bin_upper_bound: Vec<f64>,
}

impl BinMapper {
    /// Build a BinMapper from a slice of (possibly unsorted, possibly NaN-free) values.
    /// At most `max_bin` distinct bins are created using quantile cut-points.
    pub fn from_values(values: &[f64], max_bin: usize) -> Self {
        assert!(max_bin >= 2, "max_bin must be at least 2");
        assert!(!values.is_empty(), "cannot build BinMapper from empty values");

        // Sort finite values
        let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted.dedup();

        if sorted.is_empty() {
            // All values were non-finite; single bin
            return Self { bin_upper_bound: vec![f64::INFINITY] };
        }

        let n_distinct = sorted.len();
        let n_bins = n_distinct.min(max_bin);

        let mut upper_bounds: Vec<f64> = Vec::with_capacity(n_bins);

        if n_distinct <= max_bin {
            // Fewer distinct values than bins: one bin per distinct value.
            // Each bin upper bound is the midpoint to the next value (last is +inf).
            for i in 0..n_distinct - 1 {
                upper_bounds.push((sorted[i] + sorted[i + 1]) / 2.0);
            }
        } else {
            // Choose `max_bin` quantile cut-points.
            for i in 1..n_bins {
                let pos = (i * n_distinct) / n_bins;
                let pos = pos.min(n_distinct - 1);
                // midpoint between sorted[pos-1] and sorted[pos]
                let mid = (sorted[pos - 1] + sorted[pos]) / 2.0;
                if upper_bounds.last().copied() != Some(mid) {
                    upper_bounds.push(mid);
                }
            }
        }
        upper_bounds.push(f64::INFINITY);
        upper_bounds.dedup();

        Self { bin_upper_bound: upper_bounds }
    }

    /// Map a single value to its bin index (0-based).
    /// Returns the index of the first bin whose upper bound >= value.
    #[inline]
    pub fn map_value(&self, v: f64) -> u8 {
        // Binary search for the first upper bound >= v
        let idx = self.bin_upper_bound
            .partition_point(|&ub| ub < v);
        idx.min(self.num_bins() - 1) as u8
    }

    /// Number of bins.
    pub fn num_bins(&self) -> usize {
        self.bin_upper_bound.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_stays_in_bounds() {
        let values: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
        let bm = BinMapper::from_values(&values, 255);
        for &v in &values {
            let b = bm.map_value(v) as usize;
            assert!(b < bm.num_bins());
        }
    }

    #[test]
    fn map_is_monotone() {
        let values: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let bm = BinMapper::from_values(&values, 64);
        let mut last = 0u8;
        for i in 0..500 {
            let b = bm.map_value(i as f64);
            assert!(b >= last);
            last = b;
        }
    }

    #[test]
    fn fewer_distinct_than_max_bin() {
        let values = vec![1.0, 2.0, 3.0];
        let bm = BinMapper::from_values(&values, 255);
        assert_eq!(bm.map_value(1.0), 0);
        assert_eq!(bm.map_value(2.0), 1);
        assert_eq!(bm.map_value(3.0), 2);
    }
}
