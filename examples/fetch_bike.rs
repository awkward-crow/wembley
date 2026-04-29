/// Downloads and extracts the UCI Bike Sharing dataset (daily file) into data/.
///
/// Run with:  cargo run --example fetch_bike
use std::path::Path;
use std::process::Command;

fn main() {
    std::fs::create_dir_all("data").expect("could not create data/");

    let day_csv = "data/bike_sharing_day.csv";
    if Path::new(day_csv).exists() {
        println!("already exists: {day_csv}");
        return;
    }

    let zip_path = "data/bike_sharing.zip";
    if !Path::new(zip_path).exists() {
        println!("downloading {zip_path} ...");
        let status = Command::new("curl")
            .args([
                "-fsSL",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
                "-o",
                zip_path,
            ])
            .status()
            .expect("failed to run curl — is it installed?");

        if !status.success() {
            eprintln!("curl failed (exit {})", status);
            std::process::exit(1);
        }
    }

    println!("extracting day.csv → {day_csv} ...");
    let out_file = std::fs::File::create(day_csv).expect("cannot create output file");
    let status = Command::new("unzip")
        .args(["-p", zip_path, "day.csv"])
        .stdout(out_file)
        .status()
        .expect("failed to run unzip — is it installed?");

    if status.success() {
        println!("done → {day_csv}");
    } else {
        eprintln!("unzip failed — try extracting day.csv manually from {zip_path}");
        std::process::exit(1);
    }
}
