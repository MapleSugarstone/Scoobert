//! Packs a project folder and a conversation file into a review package, for checking the zip with other tools.
//! Usage: cargo run --release --example export_check -- <conversation.jsonl> <project folder> <out.zip>

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let packed = scoobert::export::review_package(a[1].as_ref(), a[2].as_ref(), a[3].as_ref(), "export_check").expect("the package saves");
    println!("{} files, {} bytes", packed.files, packed.bytes);
}
