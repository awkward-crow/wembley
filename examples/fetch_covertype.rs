/// Downloads the UCI Cover Type dataset, decompresses it, filters to the
/// binary problem (classes 1 and 2 only) and writes data/covtype_binary.csv.
///
/// Labels are recoded: class 1 (Spruce/Fir) → 0, class 2 (Lodgepole Pine) → 1.
///
/// Run with:  cargo run --example fetch_covertype
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Command, Stdio};

const URL:      &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz";
const GZ_PATH:  &str = "data/covtype.data.gz";
const OUT_PATH: &str = "data/covtype_binary.csv";

const FEATURE_NAMES: &[&str] = &[
    "Elevation", "Aspect", "Slope",
    "H_Dist_Hydrology", "V_Dist_Hydrology", "H_Dist_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "H_Dist_Fire_Points",
    "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4",
    "Soil_Type1",  "Soil_Type2",  "Soil_Type3",  "Soil_Type4",  "Soil_Type5",
    "Soil_Type6",  "Soil_Type7",  "Soil_Type8",  "Soil_Type9",  "Soil_Type10",
    "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15",
    "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20",
    "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25",
    "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30",
    "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35",
    "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40",
];

fn main() {
    std::fs::create_dir_all("data").expect("could not create data/");

    if Path::new(OUT_PATH).exists() {
        println!("already exists: {OUT_PATH}");
        return;
    }

    // ── Download ───────────────────────────────────────────────────────────
    if !Path::new(GZ_PATH).exists() {
        println!("downloading {GZ_PATH} ...");
        let status = Command::new("curl")
            .args(["-fsSL", URL, "-o", GZ_PATH])
            .status()
            .expect("failed to run curl — is it installed?");
        if !status.success() {
            eprintln!("curl failed");
            std::process::exit(1);
        }
    }

    // ── Decompress into a pipe and filter ──────────────────────────────────
    println!("filtering classes 1 & 2 → {OUT_PATH} ...");

    let mut gunzip = Command::new("gunzip")
        .args(["-c", GZ_PATH])
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to run gunzip — is it installed?");

    let reader = BufReader::new(gunzip.stdout.take().unwrap());
    let out_file = std::fs::File::create(OUT_PATH).expect("cannot create output file");
    let mut writer = BufWriter::new(out_file);

    // Header
    write!(writer, "{}", FEATURE_NAMES.join(",")).unwrap();
    writeln!(writer, ",label").unwrap();

    let mut kept = 0usize;
    let mut total = 0usize;

    for line in reader.lines() {
        let line = line.unwrap();
        total += 1;
        let last_comma = line.rfind(',').expect("malformed row");
        let label_str = &line[last_comma + 1..];
        let label: u8 = label_str.trim().parse().expect("non-integer label");
        if label == 1 || label == 2 {
            let features = &line[..last_comma];
            // Recode: class 1 → 0, class 2 → 1
            let recoded = if label == 1 { 0 } else { 1 };
            writeln!(writer, "{},{}", features, recoded).unwrap();
            kept += 1;
        }
    }

    gunzip.wait().expect("gunzip did not finish");
    println!("done — kept {kept} / {total} rows → {OUT_PATH}");
}
