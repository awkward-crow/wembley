use super::Objective;

/// Binary logistic regression loss (log-loss).
///
/// p[i]     = sigmoid(score[i])
/// grad[i]  = p[i] - label[i]
/// hess[i]  = p[i] * (1 - p[i])
pub struct BinaryLogistic;

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl Objective for BinaryLogistic {
    fn gradients_hessians(
        &self,
        scores: &[f64],
        labels: &[f32],
        grads: &mut [f32],
        hess: &mut [f32],
    ) {
        for i in 0..scores.len() {
            let p = sigmoid(scores[i]);
            grads[i] = (p - labels[i] as f64) as f32;
            // Clamp hessian away from zero to avoid degenerate splits
            hess[i] = (p * (1.0 - p)).max(1e-16) as f32;
        }
    }

    fn init_score(&self, labels: &[f32]) -> f64 {
        let mean: f64 = labels.iter().map(|&l| l as f64).sum::<f64>() / labels.len() as f64;
        let mean = mean.clamp(1e-6, 1.0 - 1e-6);
        // Log-odds
        (mean / (1.0 - mean)).ln()
    }

    fn eval_metric(&self, scores: &[f64], labels: &[f32]) -> f64 {
        let logloss: f64 = scores
            .iter()
            .zip(labels)
            .map(|(&s, &l)| {
                let p = sigmoid(s).clamp(1e-15, 1.0 - 1e-15);
                let y = l as f64;
                -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
            })
            .sum::<f64>()
            / scores.len() as f64;
        logloss
    }

    fn metric_name(&self) -> &'static str {
        "binary_logloss"
    }
}
