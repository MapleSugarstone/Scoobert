//! Downloads catalog models by name, or other models by "owner/repository:size", into a models folder the way setup
//! does, printing progress every 5%.
//! Usage: cargo run --release --example fetch_catalog -- <models folder> <catalog name or address>...

use scoobert::llama::catalog::CATALOG;
use scoobert::llama::download::download;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("a models folder"));
    let http = reqwest::Client::new();
    for name in args {
        let (spec, projector, revision) = match CATALOG.iter().find(|m| m.name == name) {
            Some(m) => (m.spec, m.projector, m.revision),
            None => (name.as_str(), false, "main"),
        };
        let last = std::sync::atomic::AtomicU64::new(u64::MAX);
        let result = download(&http, &dir, spec, projector, revision, &CancellationToken::new(), |p| {
            let step = (p.done * 20).checked_div(p.total).unwrap_or(0);
            if last.swap(step, std::sync::atomic::Ordering::SeqCst) != step {
                println!("{name}: {}%", step * 5);
            }
        })
        .await;
        match result {
            Ok(path) => println!("{name}: done in {}", path.display()),
            Err(e) => println!("{name}: error: {e:#}"),
        }
    }
}
