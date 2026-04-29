use crate::{bin_mapper::BinMapper, config::Config};

/// Column-major binned training dataset.
///
/// Each feature column is stored as a contiguous `Vec<u8>` of bin indices,
/// so the inner loop over samples in histogram construction is cache-friendly.
pub struct Dataset {
    pub num_data: usize,
    pub num_features: usize,
    /// `bins[f][i]` = bin index of sample `i` for feature `f`.
    pub bins: Vec<Vec<u8>>,
    pub bin_mappers: Vec<BinMapper>,
    pub labels: Vec<f32>,
    /// Optional feature names for display.
    pub feature_names: Option<Vec<String>>,
}

impl Dataset {
    /// Build a Dataset from a dense matrix stored as a slice of column slices.
    ///
    /// `columns[f]` must have length `num_data` for all `f`.
    pub fn from_columns(
        columns: &[&[f64]],
        labels: &[f32],
        config: &Config,
        feature_names: Option<Vec<String>>,
    ) -> Self {
        let num_features = columns.len();
        let num_data = labels.len();
        assert!(
            columns.iter().all(|c| c.len() == num_data),
            "all columns must have the same length as labels"
        );

        let mut bins = Vec::with_capacity(num_features);
        let mut bin_mappers = Vec::with_capacity(num_features);

        for col in columns {
            let bm = BinMapper::from_values(col, config.max_bin);
            let col_bins: Vec<u8> = col.iter().map(|&v| bm.map_value(v)).collect();
            bin_mappers.push(bm);
            bins.push(col_bins);
        }

        Self {
            num_data,
            num_features,
            bins,
            bin_mappers,
            labels: labels.to_vec(),
            feature_names,
        }
    }

    /// Build from a row-major dense matrix (`rows[i][f]`).
    pub fn from_rows(
        rows: &[Vec<f64>],
        labels: &[f32],
        config: &Config,
        feature_names: Option<Vec<String>>,
    ) -> Self {
        let num_data = rows.len();
        let num_features = if num_data > 0 { rows[0].len() } else { 0 };

        // Transpose to column-major
        let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(num_data); num_features];
        for row in rows {
            assert_eq!(row.len(), num_features, "all rows must have the same number of features");
            for (f, &v) in row.iter().enumerate() {
                columns[f].push(v);
            }
        }
        let col_refs: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();
        Self::from_columns(&col_refs, labels, config, feature_names)
    }

    /// Create a dataset for evaluation / prediction using bin mappers from a
    /// training dataset.  This ensures that the bin indices assigned to test
    /// samples are comparable to the thresholds stored in the trained trees.
    pub fn from_rows_with_mappers(
        rows: &[Vec<f64>],
        labels: &[f32],
        bin_mappers: &[BinMapper],
        feature_names: Option<Vec<String>>,
    ) -> Self {
        let num_data = rows.len();
        let num_features = bin_mappers.len();

        // Transpose to column-major
        let mut bins: Vec<Vec<u8>> = vec![Vec::with_capacity(num_data); num_features];
        for row in rows {
            for (f, &v) in row.iter().enumerate() {
                bins[f].push(bin_mappers[f].map_value(v));
            }
        }

        Self {
            num_data,
            num_features,
            bins,
            bin_mappers: bin_mappers.to_vec(),
            labels: labels.to_vec(),
            feature_names,
        }
    }

    /// Number of bins for feature `f`.
    pub fn num_bins(&self, f: usize) -> usize {
        self.bin_mappers[f].num_bins()
    }

    /// Return the feature name for display, falling back to "f{index}".
    pub fn feature_name(&self, f: usize) -> String {
        self.feature_names
            .as_ref()
            .and_then(|names| names.get(f))
            .cloned()
            .unwrap_or_else(|| format!("f{f}"))
    }
}
