/// UCI Bike Sharing (daily) — L2 regression and quantile regression (α=0.9),
/// with per-iteration error printed to show the effect of adding more trees.
///
/// Run:  cargo run --example bike --release
///       cargo run --example bike --release -- --shuffle
///
/// --shuffle  Randomly permutes the data before the 80/20 split so the test
///            set is a representative sample rather than the final time window.
///            This removes the distribution-shift effect and gives honest
///            coverage numbers for the quantile models.
///
/// Data: cargo run --example fetch_bike   (first time only)
use csv::ReaderBuilder;
use lgbm::{
    boosting::GBDT,
    config::Config,
    dataset::Dataset,
    objective::{Objective, QuantileRegression, RegressionL2},
};

/// Column names in the UCI day.csv file and which ones to use as features.
/// We drop: instant (row id), dteday (date string), casual, registered
/// (they sum to cnt, the target).
const TARGET: &str = "cnt";
const DROP: &[&str] = &["instant", "dteday", "casual", "registered"];

fn load_bike(path: &str) -> (Vec<Vec<f64>>, Vec<f32>, Vec<String>) {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .unwrap_or_else(|_| panic!("cannot open {path} — run `cargo run --example fetch_bike` first"));

    let all_headers: Vec<String> = rdr.headers().unwrap().iter().map(String::from).collect();

    let use_cols: Vec<usize> = all_headers
        .iter()
        .enumerate()
        .filter(|(_, h)| h.as_str() != TARGET && !DROP.contains(&h.as_str()))
        .map(|(i, _)| i)
        .collect();
    let feature_names: Vec<String> = use_cols.iter().map(|&i| all_headers[i].clone()).collect();
    let target_idx = all_headers.iter().position(|h| h == TARGET).expect("cnt column not found");

    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();

    for record in rdr.records() {
        let record = record.unwrap();
        let vals: Vec<f64> = record.iter().map(|s| s.trim().parse::<f64>().unwrap_or(0.0)).collect();
        labels.push(vals[target_idx] as f32);
        rows.push(use_cols.iter().map(|&i| vals[i]).collect());
    }
    (rows, labels, feature_names)
}

/// Fisher-Yates shuffle with a fixed seed for reproducibility.
fn shuffle(rows: &mut Vec<Vec<f64>>, labels: &mut Vec<f32>) {
    let n = rows.len();
    // LCG — good enough for a deterministic permutation, no extra deps needed.
    let mut rng: u64 = 0xdeadbeef_cafebabe;
    for i in (1..n).rev() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let j = (rng as usize) % (i + 1);
        rows.swap(i, j);
        labels.swap(i, j);
    }
}

fn run_regression(
    train_rows: &[Vec<f64>],
    train_labels: &[f32],
    test_rows: &[Vec<f64>],
    test_labels: &[f32],
    feature_names: &[String],
    config: &Config,
) {
    let train_ds = Dataset::from_rows(train_rows, train_labels, config, Some(feature_names.to_vec()));
    let test_ds  = Dataset::from_rows_with_mappers(test_rows, test_labels, &train_ds.bin_mappers, Some(feature_names.to_vec()));

    let obj = RegressionL2;
    let mut gbdt = GBDT::new(config.clone());

    gbdt.on_iteration = Some(Box::new(move |iter, train_rmse| {
        if (iter + 1) % 25 == 0 || iter == 0 {
            println!("  [iter {:>3}]  train_rmse = {:.1}", iter + 1, train_rmse);
        }
    }));

    println!("\n── L2 Regression ────────────────────────────────────────");
    gbdt.train(&train_ds, &obj);

    let test_scores = gbdt.predict(&test_ds);
    let test_rmse = obj.eval_metric(&test_scores, test_labels);
    println!("  Test RMSE: {:.1} bikes/day", test_rmse);
}

fn run_quantile(
    train_rows: &[Vec<f64>],
    train_labels: &[f32],
    test_rows: &[Vec<f64>],
    test_labels: &[f32],
    feature_names: &[String],
    config: &Config,
    alpha: f64,
) {
    let train_ds = Dataset::from_rows(train_rows, train_labels, config, Some(feature_names.to_vec()));
    let test_ds  = Dataset::from_rows_with_mappers(test_rows, test_labels, &train_ds.bin_mappers, Some(feature_names.to_vec()));

    let obj = QuantileRegression::new(alpha);
    let mut gbdt = GBDT::new(config.clone());

    gbdt.on_iteration = Some(Box::new(move |iter, pinball| {
        if (iter + 1) % 25 == 0 || iter == 0 {
            println!("  [iter {:>3}]  pinball(α={:.1}) = {:.2}", iter + 1, alpha, pinball);
        }
    }));

    println!("\n── Quantile Regression  α = {:.1} ────────────────────────", alpha);
    gbdt.train(&train_ds, &obj);

    let test_scores = gbdt.predict(&test_ds);

    // Coverage: fraction of test samples where actual <= predicted
    let coverage = test_scores
        .iter()
        .zip(test_labels)
        .filter(|(&pred, &actual)| actual as f64 <= pred)
        .count() as f64
        / test_labels.len() as f64;

    let pinball = obj.eval_metric(&test_scores, test_labels);
    println!("  Test pinball loss:  {:.2}", pinball);
    println!("  Coverage (actual ≤ pred): {:.1}%  (target: {:.0}%)", coverage * 100.0, alpha * 100.0);
}

fn main() {
    let shuffle_flag = std::env::args().any(|a| a == "--shuffle");

    let path = "data/bike_sharing_day.csv";
    let (mut rows, mut labels, feature_names) = load_bike(path);

    if shuffle_flag {
        shuffle(&mut rows, &mut labels);
        println!("UCI Bike Sharing (daily): {} samples, {} features  [shuffled]",
            rows.len(), feature_names.len());
    } else {
        println!("UCI Bike Sharing (daily): {} samples, {} features  [chronological split]",
            rows.len(), feature_names.len());
    }
    println!("Features: {}", feature_names.join(", "));

    let split = (rows.len() as f64 * 0.8) as usize;
    let (train_rows, test_rows)     = rows.split_at(split);
    let (train_labels, test_labels) = labels.split_at(split);

    let config = Config {
        num_trees: 200,
        num_leaves: 15,
        min_data_in_leaf: 10,
        learning_rate: 0.05,
        lambda_l2: 1.0,
        ..Config::default()
    };

    run_regression(train_rows, train_labels, test_rows, test_labels, &feature_names, &config);
    run_quantile(  train_rows, train_labels, test_rows, test_labels, &feature_names, &config, 0.9);
    run_quantile(  train_rows, train_labels, test_rows, test_labels, &feature_names, &config, 0.1);
}
