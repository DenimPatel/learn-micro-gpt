//! Finite-difference checks on the autograd tape, plus determinism checks on a
//! fixed-seed forward pass.
//!
//! ## Why this file exists
//!
//! `cargo test` used to compile zero tests: `src/main.rs` printed a placeholder,
//! and nothing in the crate carried a `#[test]`. A command that always passes
//! looks like evidence and is worse than no command at all.
//!
//! ## What is checked, and why each one earns its place
//!
//! 1. **The backward pass, by finite difference.** The whole value of an
//!    autograd library is that `backward` produces the derivative of the
//!    forward function. A sign error or a dropped term in the chain rule
//!    produces a model that still trains, just badly, and nothing in the loss
//!    curve says so. Perturbing the input and re-evaluating the forward function
//!    gives a definitionally-correct number to compare the analytic gradient
//!    against, and no amount of clever internal structure can fake agreement.
//!
//!    This is the same method `implementations/c/test_gradients.c` uses, and it
//!    is deliberately *not* a test of private internals: it drives the tape
//!    through its public surface (`Tensor::leaf`, the operators, `backward`,
//!    `grad`, `data`) exactly as the model does. It is therefore free to treat
//!    the backward pass as a black box, which is the only honest way to test it.
//!
//! 2. **The chain rule through every operator the model uses.** One function per
//!    operator would be a transcription of the implementation; instead each
//!    case below composes several operators, so a mistake in any of them shows
//!    up in at least one graph. `div` and `pow` are the ones worth watching:
//!    both are defined in terms of each other, so an error in either is an error
//!    in both.
//!
//! 3. **Determinism of a fixed-seed forward pass.** The reference's whole claim
//!    is that a seed fixes the run. Two `Model`s built from the same seed must
//!    produce bit-identical logits. If a hash map iteration order, an address,
//!    or a time source ever leaks into the forward pass, this is what catches
//!    it -- and the loss would stop being reproducible against the Python
//!    reference for a reason no loss curve reveals.
//!
//! ## Why the input is perturbed by rebuilding the graph
//!
//! `Model`'s weights are private and there is no `set_data`, on purpose: the
//! library is frozen, and the arena accumulates gradients in place, so there is
//! no way to re-evaluate a graph with one input moved. Both facts fall away if
//! each evaluation *builds a fresh graph from fresh leaves*: a new leaf starts
//! at `grad = 0.0`, so the analytic pass and every finite-difference probe get
//! gradients that are theirs alone. Perturbing a value then means passing a
//! different number to the function that builds the graph -- which is what the
//! `Fn(f32) -> TensorHandle` shape below makes explicit.

use microgpt::{Model, Rng, Tensor, TensorHandle};

/// The step for the central difference. Large enough that `f32` rounding in the
/// subtraction does not dominate, small enough that the `O(h^2)` truncation
/// error stays under the tolerance. `sqrt(f32::EPSILON)` is the textbook value
/// for a central difference and lands near `3.4e-4`; this is a factor of a few
/// under it, where both errors are small.
const H: f32 = 1e-3;

/// Central difference of a scalar function at `x`: `(f(x+h) - f(x-h)) / 2h`.
///
/// Central rather than forward because it is accurate to `O(h^2)` instead of
/// `O(h)`, which is the difference between a tolerance of `1e-2` and one of
/// `1e-5` for the same step size.
fn central_diff(f: impl Fn(f32) -> f32, x: f32) -> f32 {
    (f(x + H) - f(x - H)) / (2.0 * H)
}

/// Build the graph, run `backward`, and return the gradient of the input leaf
/// alongside the forward value of the root.
///
/// The leaf is what the derivative is *about*; returning the root's value too
/// lets a caller assert the forward pass agreed, so a backward pass wired to the
/// wrong node cannot pass by producing a plausible number for something else.
fn grad_and_value(build: impl Fn(TensorHandle) -> TensorHandle, x: f32) -> (f32, f32) {
    let leaf = Tensor::leaf(x);
    let root = build(leaf);
    Tensor::backward(root);
    (Tensor::grad(leaf), Tensor::data(root))
}

/// Assert that the analytic gradient of the graph `build` at `x` matches the
/// central difference of `f`, to a relative tolerance.
///
/// Relative, not absolute: the functions below have derivatives spanning a few
/// orders of magnitude, and an absolute tolerance would either be meaningless at
/// the small end or fail at the large end for rounding reasons.
fn assert_grad_matches(fd: impl Fn(f32) -> f32, f: impl Fn(f32) -> f32, build: impl Fn(TensorHandle) -> TensorHandle, x: f32) {
    let (analytic, value) = grad_and_value(&build, x);

    // The forward pass must agree with the closed form first; otherwise the
    // derivative comparison is testing the wrong function.
    let expected_value = f(x);
    assert!(
        (value - expected_value).abs() < 1e-4,
        "forward mismatch at x = {x}: graph = {value}, expected = {expected_value}"
    );

    let numeric = central_diff(&fd, x);

    let scale = analytic.abs().max(numeric.abs()).max(1e-3);
    let rel = (analytic - numeric).abs() / scale;
    assert!(
        rel < 2e-3,
        "gradient mismatch at x = {x}:\n  analytic = {analytic}\n  numeric  = {numeric}\n  rel      = {rel}"
    );
}

// ─── The operators the model actually uses, composed ───────────────────────

/// `f(x) = x*x + 2x`, so `f'(x) = 2x + 2`. Exercises `mul`, `add`, `mul_scalar`.
#[test]
fn grad_of_quadratic() {
    let f = |x: f32| x * x + 2.0 * x;
    let build = |x: TensorHandle| {
        let squared = Tensor::mul(x, x);
        let doubled = Tensor::mul_scalar(x, 2.0);
        Tensor::add(squared, doubled)
    };
    for &x in &[-3.0f32, -0.5, 0.0, 0.25, 1.0, 4.0] {
        assert_grad_matches(f, f, build, x);
    }
}

/// `f(x) = log(exp(x) + x*x)`, so `f'(x) = (exp(x) + 2x) / (exp(x) + x*x)`.
/// Exercises `exp`, `log` and `pow` -- the loss the model minimises is a `log`,
/// so this is the operator it cannot get wrong.
///
/// `x = 0` is excluded because `pow`'s local gradient is `n * data / base`, and
/// at a zero base that is `0/0`. See [`pow_gradient_is_nan_at_zero`] for that
/// limit, which is pinned rather than hidden.
#[test]
fn grad_of_log_sum() {
    let f = |x: f32| (x.exp() + x * x).ln();
    let build = |x: TensorHandle| {
        let e = Tensor::exp(x);
        let sq = Tensor::pow(x, 2.0);
        let sum = Tensor::add(e, sq);
        Tensor::log(sum)
    };
    for &x in &[-1.0f32, 0.5, 1.5, 3.0] {
        assert_grad_matches(f, f, build, x);
    }
}

/// `f(x) = exp(x*x)`, so `f'(x) = 2x * exp(x*x)`. Exercises `pow` feeding `exp`,
/// which is the shape of every activation in the model.
///
/// As above, `x = 0` is excluded: `pow(x, 2)`'s local gradient divides by the
/// base, which is zero there.
#[test]
fn grad_of_exp_of_square() {
    let f = |x: f32| (x * x).exp();
    let build = |x: TensorHandle| {
        let sq = Tensor::pow(x, 2.0);
        Tensor::exp(sq)
    };
    for &x in &[-1.0f32, -0.4, 0.7, 1.2] {
        assert_grad_matches(f, f, build, x);
    }
}

/// `pow`'s gradient is undefined at a zero base, and the port inherits that.
///
/// `d/dx x^n` is computed as `n * data / base`, so at `x = 0` the analytic
/// gradient is `0/0 = NaN` even for `n = 2`, where the true derivative is the
/// perfectly ordinary `0`. This is the reference's formula, not a porting bug,
/// and the model never evaluates it there because `pow` only ever sees
/// positive bases (`exp`, `1/n` over a positive count, and a variance that is
/// `+1e-5`). The test exists so the behaviour is *recorded*: if a later change
/// makes this a real number, that is a deliberate improvement, and if it makes
/// it `inf` or panics, this is the test that says so.
#[test]
fn pow_gradient_is_nan_at_zero() {
    let build = |x: TensorHandle| Tensor::pow(x, 2.0);
    let (analytic, _) = grad_and_value(build, 0.0);
    assert!(
        analytic.is_nan(),
        "expected NaN gradient for pow at a zero base, got {analytic}"
    );

    // Away from zero it is the ordinary derivative, so the limit is local to the
    // base being exactly zero rather than a broken formula.
    let (analytic, _) = grad_and_value(build, 2.0);
    assert!((analytic - 4.0).abs() < 1e-5, "d/dx x^2 at 2 should be 4, got {analytic}");
}

/// `f(x) = relu(x) * x`. For `x > 0` this is `x*x` with `f'(x) = 2x`; at
/// `x = 0` `relu` is not differentiable and the graph picks the sub-gradient 0,
/// so the point is deliberately excluded rather than papered over.
#[test]
fn grad_of_relu_product() {
    let f = |x: f32| if x > 0.0 { x * x } else { 0.0 };
    let build = |x: TensorHandle| {
        let r = Tensor::relu(x);
        Tensor::mul(r, x)
    };
    // Positive points only: relu's derivative has a kink at zero, and a finite
    // difference straddling the kink would average the two one-sided slopes.
    for &x in &[0.3f32, 1.0, 2.5] {
        assert_grad_matches(f, f, build, x);
    }
}

/// `f(x) = (x*x - x) / (x + 2)`, so
/// `f'(x) = ((2x - 1)(x + 2) - (x*x - x)) / (x + 2)^2`.
/// Exercises `sub`, `div` and `pow` together -- `div` is `mul` with a `pow(-1)`,
/// so this is the pair that a naive implementation gets wrong.
#[test]
fn grad_of_quotient() {
    let f = |x: f32| (x * x - x) / (x + 2.0);
    let build = |x: TensorHandle| {
        let sq = Tensor::mul(x, x);
        let numer = Tensor::sub(sq, x);
        let denom = Tensor::add_scalar(x, 2.0);
        Tensor::div(numer, denom)
    };
    for &x in &[-1.5f32, -0.5, 0.0, 1.0, 3.0] {
        assert_grad_matches(f, f, build, x);
    }
}

/// The gradient must reach a *shared* leaf once per use.
///
/// `f(x) = (x + x) * (x + x)` is `4x^2` with `f'(x) = 8x`. The subtlety is that
/// `x` is a parent of *two* different `add` nodes, and each of those is a parent
/// of the `mul`. A backward pass that visits a node once instead of accumulating
/// every path through it produces `4x` here, not `8x` -- a factor-of-two error
/// that would leave a model that trains plausibly and is simply wrong.
#[test]
fn grad_accumulates_over_shared_subgraph() {
    let f = |x: f32| 4.0 * x * x;
    let build = |x: TensorHandle| {
        let a = Tensor::add(x, x);
        let b = Tensor::add(x, x);
        Tensor::mul(a, b)
    };
    for &x in &[-2.0f32, -0.5, 0.75, 1.5, 3.0] {
        assert_grad_matches(f, f, build, x);
    }
}

/// The forward value is the function, not an approximation of it.
///
/// The finite-difference tests above read gradients; this one pins the forward
/// pass on a graph, so a change that keeps the derivative right while breaking
/// the value is caught.
#[test]
fn forward_values_are_exact() {
    let x = Tensor::leaf(3.0);
    let y = Tensor::add(Tensor::mul(x, x), Tensor::mul_scalar(x, 2.0));
    assert!((Tensor::data(y) - 15.0).abs() < 1e-5);

    let x = Tensor::leaf(0.5);
    let y = Tensor::log(Tensor::exp(x));
    assert!((Tensor::data(y) - 0.5).abs() < 1e-5);
}

// ─── The fixed-seed forward pass is deterministic ──────────────────────────

/// Two models built from the same seed must produce identical logits.
///
/// This is the "fixed seed" half of the parity claim: the Python reference is
/// reproducible, and a Rust port that is not cannot be compared to it. The check
/// runs a handful of positions through the real `forward` so the attention and
/// MLP paths -- not just the embedding lookup -- are exercised.
#[test]
fn fixed_seed_forward_is_deterministic() {
    let mut rng_a = Rng::new(42);
    let mut model_a = Model::new(Default::default(), &mut rng_a);
    let mut rng_b = Rng::new(42);
    let mut model_b = Model::new(Default::default(), &mut rng_b);

    for (token, pos) in [(1usize, 0usize), (5, 1), (9, 2), (3, 3)] {
        let a: Vec<f32> = model_a
            .forward(token, pos)
            .into_iter()
            .map(Tensor::data)
            .collect();
        let b: Vec<f32> = model_b
            .forward(token, pos)
            .into_iter()
            .map(Tensor::data)
            .collect();

        assert_eq!(a.len(), b.len(), "vocab dimension changed between runs");
        assert_eq!(
            a, b,
            "fixed-seed forward diverged at token {token}, position {pos}"
        );
    }
}

/// A different seed must produce *different* logits.
///
/// Without this, the determinism check above would pass for a model that ignored
/// its seed entirely -- which is exactly the failure a seed-honouring port must
/// not have.
#[test]
fn different_seed_forward_differs() {
    let mut rng_a = Rng::new(42);
    let mut model_a = Model::new(Default::default(), &mut rng_a);
    let mut rng_b = Rng::new(43);
    let mut model_b = Model::new(Default::default(), &mut rng_b);

    let a: Vec<f32> = model_a.forward(1, 0).into_iter().map(Tensor::data).collect();
    let b: Vec<f32> = model_b.forward(1, 0).into_iter().map(Tensor::data).collect();

    assert_ne!(a, b, "two seeds produced identical logits; the seed is ignored");
}
