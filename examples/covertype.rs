/// UCI Cover Type — binary classification (class 1 Spruce/Fir vs class 2 Lodgepole Pine).
///
/// Run:  cargo run --example covertype --release [-- --num_trees=100 --num_leaves=31
///                                                   --logloss --importance --importance-by-tree]
/// Data: cargo run --example fetch_covertype   (first time only)
use csv::ReaderBuilder;
use lgbm::{
    boosting::GBDT,
    config::Config,
    dataset::Dataset,
    objective::{BinaryLogistic, Objective},
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
    let num_trees:  usize = parse_arg(&args, "num_trees",  100);
    let num_leaves: usize = parse_arg(&args, "num_leaves",  31);
    let show_logloss:         bool = args.iter().any(|a| a == "--logloss");
    let show_importance:      bool = args.iter().any(|a| a == "--importance");
    let show_importance_trees: bool = args.iter().any(|a| a == "--importance-by-tree");

    // ── Load CSV ───────────────────────────────────────────────────────────
    let path = "data/covtype_binary.csv";
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .unwrap_or_else(|_| {
            panic!("cannot open {path} — run `cargo run --example fetch_covertype` first")
        });

    let headers: Vec<String> = rdr.headers().unwrap().iter().map(String::from).collect();
    let target_col = headers.len() - 1;
    let feature_names: Vec<String> = headers[..target_col].to_vec();

    let mut rows:   Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f32>      = Vec::new();
    for record in rdr.records() {
        let record = record.unwrap();
        let vals: Vec<f64> = record.iter().map(|s| s.parse::<f64>().unwrap()).collect();
        labels.push(vals[target_col] as f32);
        rows.push(vals[..target_col].to_vec());
    }

    println!("{} samples, {} features", rows.len(), feature_names.len());

    // ── Train / test split (80 / 20) ──────────────────────────────────────
    let split = (rows.len() as f64 * 0.8) as usize;
    let (train_rows,   test_rows)   = rows.split_at(split);
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
    let obj = BinaryLogistic;
    let mut gbdt = GBDT::new(config.clone());

    let log: std::sync::Arc<std::sync::Mutex<Vec<f64>>> = Default::default();
    let log_capture = log.clone();
    gbdt.on_iteration = Some(Box::new(move |_iter, logloss| {
        log_capture.lock().unwrap().push(logloss);
    }));

    gbdt.train(&train_ds, &obj);
    let log = log.lock().unwrap();
    let train_logloss = log.last().copied().unwrap_or(0.0);

    // ── Evaluation ─────────────────────────────────────────────────────────
    let test_scores = gbdt.predict(&test_ds);
    let test_logloss = obj.eval_metric(&test_scores, test_labels);

    let test_acc = test_scores
        .iter()
        .zip(test_labels)
        .filter(|(&s, &l)| {
            let pred = if 1.0 / (1.0 + (-s).exp()) >= 0.5 { 1.0 } else { 0.0 };
            (pred - l as f64).abs() < 0.5
        })
        .count() as f64
        / test_labels.len() as f64;

    println!(
        "num_trees={num_trees}  num_leaves={num_leaves}  \
         train_logloss={train_logloss:.4}  test_logloss={test_logloss:.4}  \
         test_acc={:.2}%",
        test_acc * 100.0,
    );

    // ── Per-iteration log-loss ─────────────────────────────────────────────
    if show_logloss {
        println!("\n{:<6}  {}", "iter", "train_logloss");
        println!("{}", "-".repeat(24));
        for (iter, ll) in log.iter().enumerate() {
            println!("{:<6}  {:.4}", iter + 1, ll);
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
