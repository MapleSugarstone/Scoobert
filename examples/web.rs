//! Tries the web tools: one search, then reads the first result looking for a phrase, or reads one address.
//! Usage: cargo run --example web -- "<query>" "<words to find>"
//!        cargo run --example web -- https://example.com/

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args[1].starts_with("http") {
        let page = scoobert::agent::web::fetch(&args[1]).await.expect("the page loads");
        println!("{} characters of text: {}", page.text.chars().count(), scoobert::util::clip(&page.text.replace('\n', " "), 300));
        return;
    }
    let results = scoobert::agent::web::search(&args[1]).await.expect("the search works");
    println!("{}", scoobert::agent::web::format_results(&args[1], &results));
    let page = scoobert::agent::web::fetch(&results[0].url).await.expect("the page loads");
    println!("\n---- page: {} characters of text, {} links", page.text.chars().count(), page.links.len());
    println!("{}", scoobert::agent::web::format_page(&page, 0, args.get(2).map(String::as_str), 1500));
}
