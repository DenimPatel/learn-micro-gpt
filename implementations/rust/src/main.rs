//! Binary entry point. The model lives in the library so that `cargo test` can
//! drive it directly; this file only parses arguments and prints.

fn main() {
    microgpt::run();
}
