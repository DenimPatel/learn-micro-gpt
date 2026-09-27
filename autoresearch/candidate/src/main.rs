//! Binary entry point. The model lives in the library so that
//! `tests/gradient_check.rs` can drive it directly; this file only calls it.

fn main() {
    microgpt_tuned::run();
}
