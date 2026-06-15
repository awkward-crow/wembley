/// California Housing — L2 regression with per-iteration RMSE and feature importance.
///
/// Run:  cargo run --example california --release [-- --num_trees=50 --num_leaves=31 --error --importance]
/// Data: cargo run --example fetch_california   (first time only)
use csv::ReaderBuilder;
use lgbm::{
    boosting::GBDT,
    config::Config,
    dataset::Dataset,
    objective::{Objective, RegressionL2},
};

fn parse_arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    let prefix = format!("--{}=", name);
    args.iter()
        .find_map(|a| a.strip_prefix(&prefix))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_trees:  usize = parse_arg(&args, "num_trees",  50);
    let num_leaves: usize = parse_arg(&args, "num_leaves", 31);
    let show_error:            bool = args.iter().any(|a| a == "--error");
    let show_importance:       bool = args.iter().any(|a| a == "--importance");
    let show_importance_trees: bool = args.iter().any(|a| a == "--importance-by-tree");

    // ── Load CSV ───────────────────────────────────────────────────────────
    let path = "data/california_housing.csv";
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .unwrap_or_else(|_| {
            panic!("cannot open {path} — run `cargo run --example fetch_california` first")
        });

    let headers: Vec<String> = rdr.headers().unwrap().iter().map(String::from).collect();
    let target_col = headers.len() - 1; // median_house_value is the last column
    let feature_names: Vec<String> = headers[..target_col].to_vec();

    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();
    for record in rdr.records() {
        let record = record.unwrap();
        let vals: Vec<f64> = record.iter().map(|s| s.parse::<f64>().unwrap()).collect();
        labels.push(vals[target_col] as f32);
        rows.push(vals[..target_col].to_vec());
    }

    println!("{} samples, {} features", rows.len(), feature_names.len());

    // ── Train / test split (80 / 20) ──────────────────────────────────────
    let split = (rows.len() as f64 * 0.8) as usize;
    let (train_rows, test_rows) = rows.split_at(split);
    let (train_labels, test_labels) = labels.split_at(split);

    let config = Config {
        num_trees,
        num_leaves,
        min_data_in_leaf: 20,
        learning_rate: 0.1,
        lambda_l2: 1.0,
        ..Config::default()
    };

    let train_ds = Dataset::from_rows(train_rows, train_labels, &config, Some(feature_names.clone()));
    let test_ds  = Dataset::from_rows_with_mappers(test_rows, test_labels, &train_ds.bin_mappers, Some(feature_names.clone()));

    // ── Train ──────────────────────────────────────────────────────────────
    let obj = RegressionL2;
    let mut gbdt = GBDT::new(config.clone());

    let rmse_log: std::sync::Arc<std::sync::Mutex<Vec<f64>>> = Default::default();
    let rmse_log_capture = rmse_log.clone();
    gbdt.on_iteration = Some(Box::new(move |_iter, train_rmse| {
        rmse_log_capture.lock().unwrap().push(train_rmse);
    }));

    gbdt.train(&train_ds, &obj);
    let rmse_log = rmse_log.lock().unwrap();
    let train_rmse_final = rmse_log.last().copied().unwrap_or(0.0);

    // ── Evaluation ────────────────────────────────────────────────────────
    let test_scores = gbdt.predict(&test_ds);
    let test_rmse = obj.eval_metric(&test_scores, test_labels);

    println!(
        "num_trees={num_trees}  num_leaves={num_leaves}  train_rmse={:.0}  test_rmse={:.0}",
        train_rmse_final * 100_000.0,
        test_rmse * 100_000.0,
    );

    if show_error {
        println!("\n{:<6}  {}", "iter", "rmse");
        println!("{}", "-".repeat(18));
        for (iter, rmse) in rmse_log.iter().enumerate() {
            println!("{:<6}  {:.0}", iter + 1, rmse * 100_000.0);
        }
    }

    let nf = train_ds.num_features;

    // ── Feature importance (gain) ──────────────────────────────────────────
    if show_importance {
        let mut ranked: Vec<(usize, f64)> = gbdt
            .feature_importance_gain(nf)
            .into_iter()
            .enumerate()
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\nFeature importance (gain):");
        println!("{:<6}  {:<14}  {}", "rank", "gain", "feature");
        println!("{}", "-".repeat(38));
        for (rank, (fi, gain)) in ranked.iter().enumerate() {
            println!("{:<6}  {:<14.0}  {}", rank + 1, gain, train_ds.feature_name(*fi));
        }
    }

    // ── Feature importance per tree ────────────────────────────────────────
    if show_importance_trees {
        for (tree_idx, importance) in gbdt.feature_importance_gain_per_tree(nf).iter().enumerate() {
            let mut ranked: Vec<(usize, f64)> = importance.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("\ntree {}", tree_idx + 1);
            println!("{:<6}  {:<14}  {}", "rank", "gain", "feature");
            println!("{}", "-".repeat(38));
            for (rank, (fi, gain)) in ranked.iter().enumerate() {
                println!("{:<6}  {:<14.0}  {}", rank + 1, gain, train_ds.feature_name(*fi));
            }
        }
    }
}
