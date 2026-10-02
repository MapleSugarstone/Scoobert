//! Makes a model lab variant outside the app, for testing.
//! Usage: cargo run --release --example lab_make -- <model.gguf> <models folder> <tools folder> <name> steer "<toward>" "<away>" <strength> <first> <last>
//!        ... <name> convert <format>

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use scoobert::llama::LocalModel;
use scoobert::llama::lab::{self, Job, Progress};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(&a[1]);
    let size = std::fs::metadata(&path)?.len();
    let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
    let model = LocalModel { name: stem.clone(), path, mmproj: None, size, args: Vec::new(), family: stem, variant: None };
    let job = match a[5].as_str() {
        "steer" => Job::Steer { toward: a[6].clone(), away: a[7].clone(), strength: a[8].parse()?, first: a[9].parse()?, last: a[10].parse()? },
        "convert" => Job::Convert { target: lab::CONVERSIONS.iter().find(|c| c.0 == a[6]).map(|c| c.0).unwrap_or("Q4_K_M") },
        other => anyhow::bail!("unknown job {other}"),
    };
    let start = std::time::Instant::now();
    let last = Arc::new(std::sync::Mutex::new(String::new()));
    let result = lab::make(reqwest::Client::new(), PathBuf::from(&a[3]), PathBuf::from(&a[2]), model, a[4].clone(), job, Arc::new(AtomicBool::new(false)), move |p: Progress| {
        let line = format!("{} {}/{}", p.status, p.done, p.total);
        let mut l = last.lock().unwrap();
        if *l != line {
            println!("{line}");
            *l = line;
        }
    })
    .await;
    println!("{result:?} after {:.0?}", start.elapsed());
    Ok(())
}
