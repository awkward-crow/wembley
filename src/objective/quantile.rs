use super::Objective;

/// Quantile (pinball) regression loss at quantile `alpha ∈ (0, 1)`.
///
/// grad[i] = (1 - alpha) if score[i] >= label[i]
///         = -alpha      if score[i] <  label[i]
/// hess[i] = 1.0  (constant, same as L2 for histogram structure finding)
///
/// After the tree structure is fixed the leaf outputs are replaced by the
/// alpha-quantile of residuals in each leaf. See `boosting.rs`.
pub struct QuantileRegression {
    pub alpha: f64,
}

impl QuantileRegression {
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0, 1)");
        Self { alpha }
    }
}

impl Objective for QuantileRegression {
    fn gradients_hessians(
        &self,
        scores: &[f64],
        labels: &[f32],
        grads: &mut [f32],
        hess: &mut [f32],
    ) {
        let alpha = self.alpha as f32;
        for i in 0..scores.len() {
            let delta = (scores[i] - labels[i] as f64) as f32;
            grads[i] = if delta >= 0.0 { 1.0 - alpha } else { -alpha };
            hess[i] = 1.0;
        }
    }

    fn init_score(&self, labels: &[f32]) -> f64 {
        // alpha-quantile of the label distribution
        let mut sorted: Vec<f32> = labels.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pos = ((sorted.len() - 1) as f64 * self.alpha) as usize;
        sorted[pos] as f64
    }

    fn eval_metric(&self, scores: &[f64], labels: &[f32]) -> f64 {
        // Mean pinball loss
        let alpha = self.alpha;
        scores
            .iter()
            .zip(labels)
            .map(|(&s, &l)| {
                let delta = s - l as f64;
                if delta >= 0.0 { alpha * delta } else { (alpha - 1.0) * delta }
            })
            .sum::<f64>()
            / scores.len() as f64
    }

    fn metric_name(&self) -> &'static str {
        "pinball"
    }

    fn needs_renew_leaf_output(&self) -> bool {
        true
    }
}
