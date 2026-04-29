mod regression;
mod binary;
mod quantile;

pub use regression::RegressionL2;
pub use binary::BinaryLogistic;
pub use quantile::QuantileRegression;

/// Common interface for all loss functions.
pub trait Objective: Send + Sync {
    /// Compute gradients and hessians of the loss w.r.t. current predictions.
    fn gradients_hessians(
        &self,
        scores: &[f64],
        labels: &[f32],
        grads: &mut [f32],
        hess: &mut [f32],
    );

    /// Initial prediction score before any trees are added.
    fn init_score(&self, labels: &[f32]) -> f64;

    /// Scalar training metric for one iteration (RMSE, log-loss, pinball).
    fn eval_metric(&self, scores: &[f64], labels: &[f32]) -> f64;

    /// Short name used in per-iteration printouts, e.g. "rmse".
    fn metric_name(&self) -> &'static str;

    /// Whether leaf outputs should be re-computed from raw residuals after the
    /// tree structure is fixed. True only for quantile regression.
    fn needs_renew_leaf_output(&self) -> bool {
        false
    }
}
