//! A finite-difference check on this crate's autograd tape.
//!
//! Written by this repository, not by the research loop. Its sha256 is pinned in
//! `tools/autoresearch.py`, and the harness refuses to run an experiment whose
//! copy of this file has moved: a probe the candidate can weaken is not a probe.
//!
//! # Why it exists
//!
//! `docs/KNOWN-ISSUES.md` issue 1. The C port's hand-written backward pass has a
//! directional derivative that is `-0.12x` the true value, and its loss curve
//! *still* tracks the reference to within 7% and still trains. A loss number
//! cannot tell a correct gradient from a wrong one. Only a finite difference
//! can, so the research loop runs this on every candidate before the candidate
//! is allowed to compete on loss.
//!
//! # What it measures
//!
//! The directional derivative of the loss along one fixed direction `d`, in all
//! parameters at once, compared against a central difference. For a correct
//! gradient the ratio is `1.0`.
//!
//! Three models are built from the same seed, so they hold identical parameters:
//! one for the analytic pass, and two per step size, for `w + h*d` and `w - h*d`.
//! A fresh model per evaluation is not tidiness — `Model::forward` appends to the
//! KV cache and `reset_cache` is private, so a second forward sequence on the
//! same model would attend to the previous one's keys and quietly measure
//! something else.
//!
//! # Update, 2026-09-27: this candidate no longer measures 1.063
//!
//! The argument below is that the Rust *frozen* tape omits `rmsnorm`, so the
//! honest answer for a faithful copy of it is 1.063 and the band is therefore
//! off-centre. That was true when this probe was written. The loop then put
//! `rmsnorm` on the tape in the candidate -- experiment 0001, kept and then
//! superseded -- and this candidate now measures **1.0003**, with the
//! implementation in the probe unchanged. So the band is currently centred, and
//! the frozen track at `implementations/rust` still measures 1.063.
//!
//! The Go and TypeScript candidates carry the same omission and no fix, at 1.127
//! and 0.722, which is `docs/KNOWN-ISSUES.md` issue 6: the band is written for
//! 1.0 and none of the three frozen tapes is at 1.0.
//!
//! Nothing below changed. The band stays [0.5, 2.0] for the reason the last
//! paragraph of this module gives, which is that it is a tripwire against a
//! collapsed or inverted gradient rather than a precision measurement.
//!
//! # The ratio is 1.063 on a correct tape, and that is not a bug in this probe
//!
//! This is the second thing `docs/KNOWN-ISSUES.md` has to say about gradients,
//! and it is the reason the band below is not centred on 1.0.
//!
//! `reference/microgpt.py:102-105` differentiates through `rmsnorm`, because
//! Python hands it the tape for free: `xi * xi` is a `Value.__mul__`,
//! `sum(...)` is `Value.__add__`, `/ len(x)` is `Value.__truediv__`, and
//! `** -0.5` is `Value.__pow__`. Every one of those records a node. A reader
//! transliterating the function to a language without operator overloading has
//! to notice that, and this one did not: `rmsnorm` computes `ms` from
//! `Tensor::data(*v)` as a plain `f32`, so the whole normalisation sits
//! *outside* the tape.
//!
//! So the analytic derivative is the gradient of a slightly different function
//! than the one the loss evaluates, and the two disagree by a fixed amount.
//! Measured here, and flat across four orders of magnitude of `h` — which is
//! what distinguishes it from a truncation or roundoff artifact:
//!
//! ```text
//! h=1e-1  ratio 1.0685      h=1e-3  ratio 1.0632
//! h=3e-2  ratio 1.0641      h=3e-4  ratio 1.0746
//! h=1e-2  ratio 1.0632      h=1e-4  ratio 1.0829
//! h=3e-3  ratio 1.0640      h=1e-5  ratio 0.9313   <- roundoff takes over
//! ```
//!
//! Past `h=1e-4` the central difference is differencing two `f32` losses of
//! magnitude ~2.08 to extract a signal of ~1e-5, and f32 spacing at 2.08 is
//! 2.4e-7. That is the noise floor, and it is why the two step sizes used here
//! are not smaller.
//!
//! The band is therefore wide on purpose. It is a tripwire, not a precision
//! measurement: the C port's broken backward pass measures `-0.12`, a zero
//! gradient measures `0`, and any sign flip is negative. All of those are caught
//! by a factor of four or more. The stability check is what does the real work —
//! it says the number is a property of the tape rather than of the step size.
//!
//! Closing the gap is not this file's business. It is `KNOWN-ISSUES.md` issue 5,
//! and it is squarely in scope for the research loop.
//!
//! Run as:
//!
//! ```text
//! cargo test --release --test gradient_check -- --nocapture
//! ```
//!
//! and it prints one machine-readable line the harness parses, recording the
//! ratio at every step size so drift is visible on the site even when the band
//! passes:
//!
//! ```text
//! GRADCHECK analytic=0.055512 params=960 ratios=h=1e-2 ratio=1.063160 | h=3e-3 ratio=1.063970
//! ```

use microgpt_tuned::{Config, Model, Rng, Tensor, TensorHandle};

/// Same seed discipline as the model, so the parameters are identical on every
/// evaluation and the direction is fixed across runs.
const SEED: u64 = 0x5EED_1234_ABCD_0001;
const DIRECTION_SEED: u64 = 0x5EED_1234_ABCD_0002;

/// Fixed step sizes for the central difference. Two, not one, and that is the
/// whole reason the measurement is trustworthy: the analytic derivative is
/// exact, so any error here is the difference operator's, and it splits into a
/// truncation term that falls as `h^2` and a roundoff term that grows as `1/h`.
/// One step size cannot tell the two apart. Two can -- if the ratio moved much
/// when `h` moved, the number would be an artifact of the step size instead of a
/// property of the tape. `implementations/c/test_gradients.c` leans on exactly
/// this, and reports that its `-0.12` is stable across `1e-2` and `3e-3`, which
/// is what makes that finding a fact rather than a guess.
const STEPS: [f32; 2] = [1e-2, 3e-3];

/// How far the analytic and numeric derivatives may disagree.
///
/// The band is deliberately wide and deliberately *not* centred on 1.0, for the
/// reason in the module docs: this crate's tape omits the `rmsnorm` path, so the
/// honest answer for a correct tape is 1.063. What this catches is a tape that
/// is broken, and the known ways to be broken are not close calls — the C port's
/// hand-written backward pass measures -0.12, a discarded gradient measures 0, a
/// sign slip measures negative. A band of 0.5 to 2.0 fails every one of those by
/// a factor of four or more while leaving the faithful-to-the-port-but-wrong
/// baseline, and a genuinely correct rewrite, comfortably inside.
///
/// If you are tempted to narrow this, read `KNOWN-ISSUES.md` issue 5 first: the
/// right way to tighten it is to put `rmsnorm` on the tape, not to shrink the
/// number.
const RATIO_MIN: f64 = 0.5;
const RATIO_MAX: f64 = 2.0;

/// The ratio must not move when the step size moves. This is the check that
/// makes the number a measurement: truncation error falls as `h^2` and roundoff
/// grows as `1/h`, so a ratio that is stable across a 3.3x change in `h` is
/// neither. Anything wrong with the tape is a fixed offset and survives, and
/// this is what separates "fixed offset" from "step size artifact".
const STEP_STABILITY: f64 = 0.05;

/// A deliberately tiny configuration. The check is about the tape's arithmetic,
/// not about the model, and this runs on every experiment and in CI.
fn probe_config() -> Config {
    Config {
        n_embd: 8,
        n_head: 2,
        n_layer: 1,
        block_size: 8,
        head_dim: 4,
        vocab_size: 8,
        num_steps: 1,
    }
}

/// A fixed token sequence, so the check needs no dataset file and cannot be
/// affected by one going missing.
const TOKENS: [usize; 6] = [0, 1, 2, 3, 4, 5];

/// Softmax over `logits`, written here rather than taken from the crate because
/// `Model::softmax` is deliberately private. Only public tape operations are
/// used, which is the point: the probe must not need anything the candidate
/// could make convenient for itself.
fn softmax(logits: &[TensorHandle]) -> Vec<TensorHandle> {
    let max_value = logits
        .iter()
        .map(|l| Tensor::data(*l))
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<TensorHandle> = logits
        .iter()
        .map(|l| Tensor::exp(Tensor::sub_scalar(*l, max_value)))
        .collect();
    let mut total = Tensor::leaf(0.0);
    for e in &exps {
        total = Tensor::add(total, *e);
    }
    exps.iter().map(|e| Tensor::div(*e, total)).collect()
}

/// The mean cross-entropy over the fixed sequence, with `offset` applied to
/// every parameter first. Building the model is the caller's job so that the
/// caller can perturb before any forward pass runs.
fn loss_of(model: &mut Model) -> f32 {
    let n = TOKENS.len() - 1;
    let mut total = Tensor::leaf(0.0);
    for pos in 0..n {
        let logits = model.forward(TOKENS[pos], pos);
        let probs = softmax(&logits);
        total = Tensor::add(total, Tensor::neg(Tensor::log(probs[TOKENS[pos + 1]])));
    }
    Tensor::data(Tensor::div_scalar(total, n as f32))
}

fn fresh() -> Model {
    let mut rng = Rng::new(SEED);
    Model::new(probe_config(), &mut rng)
}

#[test]
fn directional_derivative_matches_the_tape() {
    // The direction: one pseudo-random unit-ish vector over all parameters,
    // from its own seed so it is identical on every machine and every run.
    let mut direction_rng = Rng::new(DIRECTION_SEED);
    let probe = fresh();
    let params = probe.params();
    let direction: Vec<f32> = (0..params.len())
        .map(|_| (direction_rng.uniform() as f32) * 2.0 - 1.0)
        .collect();
    let direction_norm: f32 = direction.iter().map(|d| d * d).sum::<f32>().sqrt();
    let direction: Vec<f32> = direction
        .iter()
        .map(|d| d / direction_norm)
        .collect();

    // Analytic: one backward pass on a fresh model, then the dot product.
    let mut analytic_model = fresh();
    let analytic_params = analytic_model.params();
    let loss = {
        // Re-walk the same sequence so the tape is rooted at the loss.
        let n = TOKENS.len() - 1;
        let mut total = Tensor::leaf(0.0);
        for pos in 0..n {
            let logits = analytic_model.forward(TOKENS[pos], pos);
            let probs = softmax(&logits);
            total = Tensor::add(total, Tensor::neg(Tensor::log(probs[TOKENS[pos + 1]])));
        }
        Tensor::div_scalar(total, n as f32)
    };
    Tensor::backward(loss);
    let analytic: f64 = analytic_params
        .iter()
        .zip(direction.iter())
        .map(|(p, d)| Tensor::grad(*p) as f64 * *d as f64)
        .sum();

    // Numeric: for each step size, two more fresh models perturbed in
    // opposite directions. Fresh per evaluation, because `Model::forward`
    // appends to the KV cache and `reset_cache` is private.
    let mut reported: Vec<String> = Vec::new();
    let mut ratios: Vec<f64> = Vec::new();
    for h in STEPS {
        let mut plus = fresh();
        for (p, d) in plus.params().iter().zip(direction.iter()) {
            Tensor::set_data(*p, Tensor::data(*p) + h * *d);
        }
        let loss_plus = loss_of(&mut plus) as f64;

        let mut minus = fresh();
        for (p, d) in minus.params().iter().zip(direction.iter()) {
            Tensor::set_data(*p, Tensor::data(*p) - h * *d);
        }
        let loss_minus = loss_of(&mut minus) as f64;

        let numeric = (loss_plus - loss_minus) / (2.0 * h as f64);
        let ratio = if numeric == 0.0 { f64::INFINITY } else { analytic / numeric };
        ratios.push(ratio);
        reported.push(format!("h={h:e} ratio={ratio:.6} numeric={numeric:.6}"));
    }

    println!(
        "GRADCHECK analytic={analytic:.6} params={} ratios={}",
        analytic_params.len(),
        reported.join(" | ")
    );

    for (h, ratio) in STEPS.iter().zip(ratios.iter()) {
        assert!(
            ratio.is_finite(),
            "the ratio at h={h:e} is not finite; the loss is diverging"
        );
        assert!(
            (RATIO_MIN..=RATIO_MAX).contains(ratio),
            "directional derivative ratio is {ratio:.4} at h={h:e}, outside \
             [{RATIO_MIN}, {RATIO_MAX}]. Analytic {analytic:.6}. Compare \
             docs/KNOWN-ISSUES.md issue 1, where the same measurement is -0.12 for \
             a backward pass that is wrong and whose loss curve still looks fine."
        );
    }

    let spread = (ratios[0] - ratios[1]).abs() / ratios[1].abs();
    assert!(
        spread <= STEP_STABILITY,
        "the ratio moved {spread:.1}% when the step size moved from {:e} to {:e} \
         ({:.4} then {:.4}), so it is an artifact of the difference operator rather \
         than a property of the tape. Widen the step sizes, or the loss is not \
         smooth enough at this scale for a central difference to mean anything.",
        STEPS[0], STEPS[1], ratios[0], ratios[1]
    );
}
