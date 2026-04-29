use super::Objective;

/// Squared-error (L2) regression loss.
///
/// grad[i] = score[i] - label[i]
/// hess[i] = 1.0
pub struct RegressionL2;

impl Objective for RegressionL2 {
    fn gradients_hessians(
        &self,
        scores: &[f64],
        labels: &[f32],
        grads: &mut [f32],
        hess: &mut [f32],
    ) {
        for i in 0..scores.len() {
            grads[i] = (scores[i] - labels[i] as f64) as f32;
            hess[i] = 1.0;
        }
    }

    fn init_score(&self, labels: &[f32]) -> f64 {
        let sum: f64 = labels.iter().map(|&l| l as f64).sum();
        sum / labels.len() as f64
    }

    fn eval_metric(&self, scores: &[f64], labels: &[f32]) -> f64 {
        let mse: f64 = scores
            .iter()
            .zip(labels)
            .map(|(&s, &l)| (s - l as f64).powi(2))
            .sum::<f64>()
            / scores.len() as f64;
        mse.sqrt()
    }

    fn metric_name(&self) -> &'static str {
        "rmse"
    }
}
