fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        res.set("ProductName", "Scoobert");
        res.set("FileDescription", "Scoobert");
        res.compile().expect("the Windows icon resource compiles");
    }
}
