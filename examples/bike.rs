/// UCI Bike Sharing (daily) — quantile regression at α=0.9 / 0.5 (median) / 0.1.
///
/// Run:  cargo run --example bike --release [-- --shuffle --error --importance --importance-by-tree]
///
/// --shuffle            Randomly permutes the data before the 80/20 split.
/// --error              Print per-iteration train metric for each model.
/// --importance         Print feature importance (gain) after the median model.
/// --importance-by-tree Print per-tree feature importance after the median model.
///
/// Data: cargo run --example fetch_bike   (first time only)
use csv::ReaderBuilder;
use lgbm::{
    boosting::GBDT,
    config::Config,
    dataset::Dataset,
    objective::{Objective, QuantileRegression},
};

/// Columns to drop: row id, date string, and the two sub-totals that sum to cnt.
const TARGET: &str = "cnt";
const DROP:   &[&str] = &["instant", "dteday", "casual", "registered"];

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

    let mut rows:   Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f32>      = Vec::new();
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

fn run_quantile(
    train_rows:    &[Vec<f64>],
    train_labels:  &[f32],
    test_rows:     &[Vec<f64>],
    test_labels:   &[f32],
    feature_names: &[String],
    config:        &Config,
    alpha:         f64,
    show_error:         bool,
    show_importance:    bool,
    show_importance_trees: bool,
) {
    let train_ds = Dataset::from_rows(train_rows, train_labels, config, Some(feature_names.to_vec()));
    let test_ds  = Dataset::from_rows_with_mappers(test_rows, test_labels, &train_ds.bin_mappers, Some(feature_names.to_vec()));

    let obj = QuantileRegression::new(alpha);
    let mut gbdt = GBDT::new(config.clone());

    let pinball_log: std::sync::Arc<std::sync::Mutex<Vec<f64>>> = Default::default();
    let pinball_log_capture = pinball_log.clone();
    gbdt.on_iteration = Some(Box::new(move |_iter, pinball| {
        pinball_log_capture.lock().unwrap().push(pinball);
    }));

    gbdt.train(&train_ds, &obj);
    let pinball_log = pinball_log.lock().unwrap();

    let test_scores = gbdt.predict(&test_ds);
    let coverage = test_scores
        .iter()
        .zip(test_labels)
        .filter(|(&pred, &actual)| actual as f64 <= pred)
        .count() as f64
        / test_labels.len() as f64;
    let test_pinball = obj.eval_metric(&test_scores, test_labels);

    println!(
        "Q(α={:.1})    test_pinball={:.2}  coverage={:.1}%  (target {:.0}%)",
        alpha, test_pinball, coverage * 100.0, alpha * 100.0,
    );

    if show_error {
        println!("\n{:<6}  {}", "iter", format!("pinball(α={:.1})", alpha));
        println!("{}", "-".repeat(26));
        for (iter, pinball) in pinball_log.iter().enumerate() {
            println!("{:<6}  {:.4}", iter + 1, pinball);
        }
    }

    let nf = train_ds.num_features;

    if show_importance {
        let mut ranked: Vec<(usize, f64)> = gbdt
            .feature_importance_gain(nf)
            .into_iter()
            .enumerate()
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\nFeature importance (gain) — Q(α={:.1}):", alpha);
        println!("{:<6}  {:<14}  {}", "rank", "gain", "feature");
        println!("{}", "-".repeat(38));
        for (rank, (fi, gain)) in ranked.iter().enumerate() {
            println!("{:<6}  {:<14.0}  {}", rank + 1, gain, train_ds.feature_name(*fi));
        }
    }

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

fn parse_arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    let prefix = format!("--{}=", name);
    args.iter()
        .find_map(|a| a.strip_prefix(&prefix))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_trees:  usize = parse_arg(&args, "num_trees",  200);
    let num_leaves: usize = parse_arg(&args, "num_leaves",  15);
    let shuffle_flag          = args.iter().any(|a| a == "--shuffle");
    let show_error            = args.iter().any(|a| a == "--error");
    let show_importance       = args.iter().any(|a| a == "--importance");
    let show_importance_trees = args.iter().any(|a| a == "--importance-by-tree");

    let path = "data/bike_sharing_day.csv";
    let (mut rows, mut labels, feature_names) = load_bike(path);

    if shuffle_flag {
        shuffle(&mut rows, &mut labels);
    }

    println!(
        "{} samples, {} features  [{}]",
        rows.len(), feature_names.len(),
        if shuffle_flag { "shuffled" } else { "chronological" },
    );

    let split = (rows.len() as f64 * 0.8) as usize;
    let (train_rows,   test_rows)   = rows.split_at(split);
    let (train_labels, test_labels) = labels.split_at(split);

    let config = Config {
        num_trees,
        num_leaves,
        min_data_in_leaf: 10,
        learning_rate:    0.05,
        lambda_l2:        1.0,
        ..Config::default()
    };

    run_quantile(train_rows, train_labels, test_rows, test_labels, &feature_names, &config, 0.9, show_error, false, false);
    run_quantile(train_rows, train_labels, test_rows, test_labels, &feature_names, &config, 0.5, show_error, show_importance, show_importance_trees);
    run_quantile(train_rows, train_labels, test_rows, test_labels, &feature_names, &config, 0.1, show_error, false, false);
}
