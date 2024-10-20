use std::env;

fn main() {
    println!("cargo:rerun-if-changed=./src/lib.rs");

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_include_version(true)
        .with_documentation(false)
        .with_parse_deps(true)
        .with_cpp_compat(true)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("libpyannote.h");
}
