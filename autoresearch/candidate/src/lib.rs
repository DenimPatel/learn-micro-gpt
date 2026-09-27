//! A Rust port of Andrej Karpathy's 199-line `microgpt.py`.
//!
//! # Why this looks like the Python file
//!
//! The most interesting thing about `microgpt.py` is that the *model* is 199
//! lines of unremarkable code: a linear layer is a loop over a matrix, softmax
//! is four lines, attention is a loop over heads. All the subtlety lives in
//! autograd, which is 40 lines of class.
//!
//! So this port keeps that structure exactly. `linear` here is a loop over a
//! matrix, not a call into BLAS. The point is that a reader can put the two
//! files side by side and see nothing but Rust.
//!
//! The one thing that could not be transliterated is autograd itself, because
//! Rust has no operator overloading. So [`Tensor`] below is a small explicit
//! tape: every operation records its operands and its local derivative, and
//! [`Tensor::backward`] walks the result. It is Karpathy's `class Value` in the
//! one style Rust allows — which is precisely the interesting thing to see about
//! Rust, and the reason this file teaches anything.
//!
//! # What Rust changes, and what it does not
//!
//! Two things, and it is worth being concrete about both.
//!
//! **It changes the arithmetic's *type*, not its shape.** `f32` throughout, to
//! match the C port, with `mul_add` where the contraction loop wants it. The
//! parameter count is identical — 4,192 — and the loss curve lands in the same
//! place.
//!
//! **It does not change the algorithm.** No batching, no fused kernels, no SIMD
//! intrinsics. Those are the C port's job, and they belong in a different file
//! with a different purpose. This one is here to be read.
//!
//! Compare [`../../../docs/KNOWN-ISSUES.md`]: the C port hand-writes its backward
//! pass and gets it wrong. This one cannot, because the tape computes the
//! derivatives.

// ─── The autoresearch candidate ──────────────────────────────────────────────
//
// This file began as a byte-for-byte copy of `implementations/rust/src/lib.rs`,
// plus the one method `Tensor::set_data` below. That baseline is frozen and
// sha256-pinned in `tools/provenance.py`, and the site renders it, so the tuned
// copy lives here instead of there. See `autoresearch/README.md`.
//
// A model is free to rewrite everything below this line. It is not free to
// change `tests/gradient_check.rs`, whose digest `tools/autoresearch.py` checks
// before every experiment, and it must keep the three things the harness reads:
// `run()`, the `--input/--steps/--seed` arguments, and the
// `step N / M | loss X` line on stdout.

use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};

/// The hyperparameter configuration. Read from `--n-embd` and friends, with the
/// reference's values as the defaults.
#[derive(Clone, Debug)]
pub struct Config {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub block_size: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_steps: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n_embd: 32,
            n_head: 4,
            n_layer: 1,
            block_size: 16,
            head_dim: 8,
            vocab_size: 27,
            num_steps: 1000,
        }
    }
}

// ─── Autograd ──────────────────────────────────────────────────────────────

/// A scalar in the computation graph.
///
/// This is `class Value` in the reference, field for field:
///
/// * `data` is the number — the only thing the forward pass cares about.
/// * `grad` is the derivative of the final loss with respect to `data`, zero
///   until [`Tensor::backward`] runs.
/// * `parents` are the Tensors that produced this one.
/// * `local_grads` is how much each of those contributed, positionally.
///
/// Rust has no operator overloading, so every method here corresponds to one
/// Python dunder: [`Tensor::add`] is `__add__`, [`Tensor::mul`] is `__mul__`,
/// and so on. The correspondence is deliberate — it is what lets the two files
/// be read together.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: f32,
    pub grad: f32,
    pub parents: Vec<usize>,
    pub local_grads: Vec<f32>,
}

thread_local! {
    /// The tape. A thread-local arena rather than `Rc<RefCell<..>>` on every
    /// value: the reference stores each node's parents inline, whereas here
    /// nodes are indexed into a flat arena, which keeps a tensor a `Copy`-able
    /// 4-word handle instead of a heap allocation per arithmetic operation.
    ///
    /// That is the single biggest structural difference between this file and the
    /// Python one, and it is invisible when reading the model code below — which
    /// is the point of doing it this way.
    static ARENA: std::cell::RefCell<Arena> = std::cell::RefCell::new(Arena::new());
}

struct Arena {
    nodes: Vec<Node>,
    /// Scratch for `backward`'s topological sort, reused across calls.
    visited: Vec<bool>,
}

#[derive(Clone, Debug)]
struct Node {
    data: f32,
    grad: f32,
    parents: Vec<usize>,
    local_grads: Vec<f32>,
}

impl Arena {
    fn new() -> Self {
        Arena {
            nodes: Vec::new(),
            visited: Vec::new(),
        }
    }

    fn push(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }
}

/// A handle into the arena. Copyable, and the thing the model code manipulates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorHandle(usize);

impl fmt::Display for TensorHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", Tensor::data(*self))
    }
}

impl Tensor {
    /// `Value(data)` — wrap a plain number.
    pub fn leaf(data: f32) -> TensorHandle {
        ARENA.with(|a| {
            TensorHandle(a.borrow_mut().push(Node {
                data,
                grad: 0.0,
                parents: Vec::new(),
                local_grads: Vec::new(),
            }))
        })
    }

    /// Read the value. Reads the field directly rather than cloning the node.
    ///
    /// The first version cloned the whole `Node` -- which owns two `Vec`s -- on
    /// every single read, and `linear` reads one value per multiply-accumulate.
    /// That made the port roughly a hundred times slower than the TypeScript one
    /// running the identical algorithm, which is not a useful thing for a
    /// "compare the languages" file to be.
    pub fn data(handle: TensorHandle) -> f32 {
        ARENA.with(|a| a.borrow().nodes[handle.0].data)
    }

    pub fn grad(handle: TensorHandle) -> f32 {
        ARENA.with(|a| a.borrow().nodes[handle.0].grad)
    }

    /// Write `value` into a parameter.
    ///
    /// This method exists for exactly one reason, and it is the only difference
    /// between this file and the frozen baseline it was copied from. An external
    /// finite-difference probe has to move the model to `w ± h·d` to check the
    /// tape, and the arena is private, so there is no way to do that from
    /// outside without a way in. `tests/gradient_check.rs` is that probe.
    ///
    /// It deliberately does *not* touch `grad`. The training loop zeroes
    /// gradients as it updates, because `backward` accumulates rather than
    /// assigns; a caller that reuses a model for a second `backward` has the
    /// same obligation, and hiding it inside a setter would be worse.
    pub fn set_data(handle: TensorHandle, value: f32) {
        ARENA.with(|a| a.borrow_mut().nodes[handle.0].data = value);
    }

    fn spawn(data: f32, parents: Vec<usize>, local_grads: Vec<f32>) -> TensorHandle {
        debug_assert_eq!(
            parents.len(),
            local_grads.len(),
            "parents and local_grads must be positionally aligned"
        );
        ARENA.with(|a| {
            TensorHandle(a.borrow_mut().push(Node {
                data,
                grad: 0.0,
                parents,
                local_grads,
            }))
        })
    }

    /// `__add__`. The local gradients are 1 and 1: `d(a+b)/da = d(a+b)/db = 1`.
    pub fn add(a: TensorHandle, b: TensorHandle) -> TensorHandle {
        Self::spawn(
            Self::data(a) + Self::data(b),
            vec![a.0, b.0],
            vec![1.0, 1.0],
        )
    }

    /// `__mul__`. The local gradients are *the other operand's value*, which is
    /// the product rule — nothing memorises a rule table.
    pub fn mul(a: TensorHandle, b: TensorHandle) -> TensorHandle {
        Self::spawn(
            Self::data(a) * Self::data(b),
            vec![a.0, b.0],
            vec![Self::data(b), Self::data(a)],
        )
    }

    /// `__neg__`, defined as `self * -1` so the graph never grows a special case.
    pub fn neg(a: TensorHandle) -> TensorHandle {
        Self::mul(a, Self::leaf(-1.0))
    }

    /// `__sub__`, i.e. `self + (-other)`.
    pub fn sub(a: TensorHandle, b: TensorHandle) -> TensorHandle {
        Self::add(a, Self::neg(b))
    }

    /// `self + n` for a plain float.
    pub fn add_scalar(a: TensorHandle, n: f32) -> TensorHandle {
        Self::spawn(Self::data(a) + n, vec![a.0], vec![1.0])
    }

    /// `self - n` for a plain float.
    pub fn sub_scalar(a: TensorHandle, n: f32) -> TensorHandle {
        Self::spawn(Self::data(a) - n, vec![a.0], vec![1.0])
    }

    /// `self * n` for a plain float.
    pub fn mul_scalar(a: TensorHandle, n: f32) -> TensorHandle {
        Self::spawn(Self::data(a) * n, vec![a.0], vec![n])
    }

    /// `self / n` for a plain float, as a multiplication so the graph records
    /// only the numerator's gradient — which is what makes the loss a mean whose
    /// gradient is scaled by `1/n` without a constant entering the tape.
    pub fn div_scalar(a: TensorHandle, n: f32) -> TensorHandle {
        Self::spawn(Self::data(a) * (1.0 / n), vec![a.0], vec![1.0 / n])
    }

    fn linear(w: &[TensorHandle], x: &[TensorHandle]) -> TensorHandle {
        debug_assert_eq!(w.len(), x.len());
        let (data, parents, local_grads) = ARENA.with(|a| {
            let arena = a.borrow();
            let mut data = 0.0f32;
            let mut local_grads = Vec::with_capacity(w.len() + x.len());

            // Weight parents come first and receive x as their local gradient.
            for (weight, input) in w.iter().zip(x.iter()) {
                let weight_data = arena.nodes[weight.0].data;
                let input_data = arena.nodes[input.0].data;
                data = weight_data.mul_add(input_data, data);
                local_grads.push(input_data);
            }

            // Input parents come second and receive w as their local gradient.
            for weight in w {
                local_grads.push(arena.nodes[weight.0].data);
            }

            let parents = w
                .iter()
                .chain(x.iter())
                .copied()
                .map(|handle| handle.0)
                .collect();
            (data, parents, local_grads)
        });
        Self::spawn(data, parents, local_grads)
    }

    /// One weighted sum over the cached values, recorded as a single tape
    /// node rather than a multiply and add for every cached position.
    fn weighted_sum(
        weights: &[TensorHandle],
        values: &[&[TensorHandle]],
        dimension: usize,
    ) -> TensorHandle {
        debug_assert_eq!(weights.len(), values.len());
        let (data, parents, local_grads) = ARENA.with(|a| {
            let arena = a.borrow();
            let mut data = 0.0f32;
            let mut parents = Vec::with_capacity(weights.len() * 2);
            let mut local_grads = Vec::with_capacity(weights.len() * 2);

            // Weight parents receive the corresponding value as their local
            // gradient.
            for (weight, value) in weights.iter().zip(values) {
                let weight_data = arena.nodes[weight.0].data;
                let value_data = arena.nodes[value[dimension].0].data;
                data = weight_data.mul_add(value_data, data);
                parents.push(weight.0);
                local_grads.push(value_data);
            }

            // Value parents come second and receive their weight.
            for (weight, value) in weights.iter().zip(values) {
                parents.push(value[dimension].0);
                local_grads.push(arena.nodes[weight.0].data);
            }

            (data, parents, local_grads)
        });
        Self::spawn(data, parents, local_grads)
    }

    /// `__pow__` for a constant exponent.
    pub fn pow(a: TensorHandle, exponent: f32) -> TensorHandle {
        let data = Self::data(a).powf(exponent);
        Self::spawn(data, vec![a.0], vec![exponent * data / Self::data(a)])
    }

    /// `__truediv__`, i.e. `self * other**-1`.
    pub fn div(a: TensorHandle, b: TensorHandle) -> TensorHandle {
        Self::mul(a, Self::pow(b, -1.0))
    }

    /// `log()`.
    pub fn log(a: TensorHandle) -> TensorHandle {
        let data = Self::data(a);
        Self::spawn(data.ln(), vec![a.0], vec![1.0 / data])
    }

    /// `exp()`.
    pub fn exp(a: TensorHandle) -> TensorHandle {
        let data = Self::data(a).exp();
        Self::spawn(data, vec![a.0], vec![data])
    }

    /// `exp(value - shift)` as one correctly differentiated tape node.
    fn shifted_exp(value: TensorHandle, shift: f32) -> TensorHandle {
        ARENA.with(|a| {
            let mut arena = a.borrow_mut();
            let shifted = arena.nodes[value.0].data - shift;
            let data = shifted.exp();
            TensorHandle(arena.push(Node {
                data,
                grad: 0.0,
                parents: vec![value.0],
                local_grads: vec![data],
            }))
        })
    }

    /// Sum cached values in one correctly differentiated tape node.
    fn sum(values: &[TensorHandle]) -> TensorHandle {
        ARENA.with(|a| {
            let mut arena = a.borrow_mut();
            let mut data = 0.0f32;
            for value in values {
                data += arena.nodes[value.0].data;
            }
            TensorHandle(arena.push(Node {
                data,
                grad: 0.0,
                parents: values.iter().map(|value| value.0).collect(),
                local_grads: vec![1.0; values.len()],
            }))
        })
    }

    /// Sum squared values in one correctly differentiated tape node.
    fn sum_squares(values: &[TensorHandle]) -> TensorHandle {
        ARENA.with(|a| {
            let mut arena = a.borrow_mut();
            let mut data = 0.0f32;
            let mut local_grads = Vec::with_capacity(values.len());
            for value in values {
                let value_data = arena.nodes[value.0].data;
                data += value_data * value_data;
                local_grads.push(2.0 * value_data);
            }
            TensorHandle(arena.push(Node {
                data,
                grad: 0.0,
                parents: values.iter().map(|value| value.0).collect(),
                local_grads,
            }))
        })
    }

    /// `relu()`. The derivative is 1 above zero and 0 below, and ambiguous
    /// exactly at zero, where this picks 0 — the same choice the reference makes.
    pub fn relu(a: TensorHandle) -> TensorHandle {
        let data = Self::data(a);
        Self::spawn(
            data.max(0.0),
            vec![a.0],
            vec![if data > 0.0 { 1.0 } else { 0.0 }],
        )
    }

    /// The whole backward pass, in one function.
    ///
    /// Phase 1 sorts the graph parents-first with an explicit-stack depth-first
    /// walk — iterative rather than recursive, because the reference's
    /// `build_topo` blows the Python recursion limit somewhere around a thousand
    /// nodes deep, and a Rust port that inherited that ceiling would fail on a
    /// deeper model for a reason that has nothing to do with Rust.
    ///
    /// Phase 2 walks it in reverse, accumulating
    /// `child.grad += local_grad * parent.grad` — the chain rule, as a loop.
    /// `+=` is load-bearing: a tensor is reused by every document, every
    /// position and every head, so its gradient has many contributions to sum.
    pub fn backward(root: TensorHandle) {
        let mut topo: Vec<usize> = Vec::new();

        ARENA.with(|a| {
            let mut arena = a.borrow_mut();

            // `visited` lives in the arena and is only ever grown, never
            // reallocated. The reference's `build_topo` allocates a fresh `set()`
            // per call; here the arena spans the whole run, so a per-step
            // allocation would be sized to every node ever created.
            let node_count = arena.nodes.len();
            if arena.visited.len() < node_count {
                arena.visited.resize(node_count, false);
            }

            // Iterative depth-first walk, parents-first.
            //
            // The visited check has to happen when a child is *pushed*, not when
            // it is popped. Checking on pop re-walks the children of an
            // already-visited node, and since a weight is used by every position
            // and every head, the number of distinct paths through the graph is
            // exponential in the depth. The first version of this function did it
            // the other way round and took over ten minutes for twenty steps.
            let mut stack: Vec<(usize, usize)> = vec![(root.0, 0)];
            while let Some((node_index, cursor)) = stack.pop() {
                if cursor == 0 {
                    if arena.visited[node_index] {
                        continue;
                    }
                    arena.visited[node_index] = true;
                }
                match arena.nodes[node_index].parents.get(cursor).copied() {
                    Some(child_index) => {
                        stack.push((node_index, cursor + 1));
                        if !arena.visited[child_index] {
                            stack.push((child_index, 0));
                        }
                    }
                    // No parent at this cursor means every parent is done, so this
                    // node is a leaf of the remaining work and can be emitted.
                    None => topo.push(node_index),
                }
            }

            // Phase 2: the chain rule, as a loop. `+=` is load-bearing -- a
            // tensor is reused by every document, every position and every head,
            // so its gradient has many contributions to sum.
            arena.nodes[root.0].grad = 1.0;
            for index in topo.iter().rev() {
                let grad = arena.nodes[*index].grad;
                let parent_count = arena.nodes[*index].parents.len();
                for slot in 0..parent_count {
                    let child = arena.nodes[*index].parents[slot];
                    let local_grad = arena.nodes[*index].local_grads[slot];
                    arena.nodes[child].grad += local_grad * grad;
                }
                arena.visited[*index] = false;
            }
        });
    }
}

// ─── Model ─────────────────────────────────────────────────────────────────

/// `[[Value]]` — a list of rows, exactly as in the reference. A `Vec<Vec<..>>` of
/// handles rather than a flat buffer because the correspondence is the point.
type Matrix = Vec<Vec<TensorHandle>>;

pub struct Model {
    pub config: Config,
    state: Vec<(String, Matrix)>,
    /// `state` keyed for O(1) lookup. The Vec above is the readable form, kept in
    /// order so the parameters flatten deterministically; this is the same data
    /// for the hot path, because a linear scan with string comparisons inside the
    /// forward pass costs more than the arithmetic it guards.
    index: std::collections::HashMap<String, usize>,
    keys: Vec<Vec<Vec<TensorHandle>>>,
    values: Vec<Vec<Vec<TensorHandle>>>,
}

/// `matrix` — a Gaussian draw with std 0.08, a deliberately small number that
/// keeps the initial distribution nearly uniform.
fn new_matrix(rng: &mut Rng, nout: usize, nin: usize, std: f32) -> Matrix {
    (0..nout)
        .map(|_| (0..nin).map(|_| Tensor::leaf(rng.gauss() * std)).collect())
        .collect()
}

impl Model {
    pub fn new(config: Config, rng: &mut Rng) -> Model {
        let mut state: Vec<(String, Matrix)> = Vec::new();
        let std = 0.08;
        state.push(("wte".into(), new_matrix(rng, config.vocab_size, config.n_embd, std)));
        state.push(("wpe".into(), new_matrix(rng, config.block_size, config.n_embd, 0.02)));
        state.push(("lm_head".into(), new_matrix(rng, config.vocab_size, config.n_embd, std)));
        for i in 0..config.n_layer {
            let p = format!("layer{i}.");
            state.push((format!("{p}attn_wq"), new_matrix(rng, config.n_embd, config.n_embd, std)));
            state.push((format!("{p}attn_wk"), new_matrix(rng, config.n_embd, config.n_embd, std)));
            state.push((format!("{p}attn_wv"), new_matrix(rng, config.n_embd, config.n_embd, std)));
            state.push((format!("{p}attn_wo"), new_matrix(rng, config.n_embd, config.n_embd, std)));
            state.push((format!("{p}mlp_fc1"), new_matrix(rng, 4 * config.n_embd, config.n_embd, std)));
            state.push((format!("{p}mlp_fc2"), new_matrix(rng, config.n_embd, 4 * config.n_embd, std)));
        }
        let index: std::collections::HashMap<String, usize> = state
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), i))
            .collect();
        let mut model = Model {
            config,
            state,
            index,
            keys: Vec::new(),
            values: Vec::new(),
        };
        model.reset_cache();
        model
    }

    fn reset_cache(&mut self) {
        self.keys = vec![Vec::new(); self.config.n_layer];
        self.values = vec![Vec::new(); self.config.n_layer];
    }

    fn get(&self, key: &str) -> &Matrix {
        &self.state[self.index[key]].1
    }

    /// Every parameter, flattened. The optimizer does not care about structure,
    /// and neither does this.
    pub fn params(&self) -> Vec<TensorHandle> {
        self.state
            .iter()
            .flat_map(|(_, matrix)| matrix.iter())
            .flat_map(|row| row.iter().copied())
            .collect()
    }

    /// `linear(x, w)` — the only place a weight matrix is used, and most of the
    /// arithmetic in the model.
    ///
    /// The inner loop uses `mul_add` so the contraction is a fused multiply-add,
    /// which on `f32` is *not* the same as `acc + x*y` written separately: the
    /// fused form keeps one rounding instead of two. That makes this port's loss
    /// curve differ from the Python reference in the last bits, which is exactly
    /// why the parity gate is statistical rather than exact. See
    /// `docs/BENCHMARKS.md`.
    fn linear(x: &[TensorHandle], w: &Matrix) -> Vec<TensorHandle> {
        w.iter()
            .map(|row| Tensor::linear(row, x))
            .collect()
    }

    /// `softmax(logits)` — subtract the max before exponentiating. Not a
    /// numerical nicety: it is what makes the function usable, since the
    /// subtraction leaves the result exactly unchanged.
    fn softmax(logits: &[TensorHandle]) -> Vec<TensorHandle> {
        let max_val = logits.iter().map(|l| Tensor::data(*l)).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<TensorHandle> = logits
            .iter()
            .map(|l| Tensor::shifted_exp(*l, max_val))
            .collect();
        let total = Tensor::sum(&exps);
        exps.iter().map(|e| Tensor::div(*e, total)).collect()
    }

    /// `rmsnorm(x)` — rescale by the reciprocal of the root-mean-square, with eps
    /// inside the square root to bound the division.
    fn rmsnorm(x: &[TensorHandle]) -> Vec<TensorHandle> {
        let ms = Tensor::div_scalar(Tensor::sum_squares(x), x.len() as f32);
        let scale = Tensor::pow(Tensor::add_scalar(ms, 1e-5), -0.5);
        x.iter().map(|v| Tensor::mul(*v, scale)).collect()
    }

    /// `gpt(token_id, pos_id, keys, values)` — a stateless function from one
    /// token id at one position to logits over the vocabulary, given the keys
    /// and values of earlier positions. That is the whole design: the four
    /// arguments *are* the KV cache, spelled the long way.
    pub fn forward(&mut self, token_id: usize, pos_id: usize) -> Vec<TensorHandle> {
        let tok_emb = self.get("wte")[token_id].clone();
        let pos_emb = self.get("wpe")[pos_id].clone();
        let mut x: Vec<TensorHandle> = tok_emb
            .iter()
            .zip(pos_emb.iter())
            .map(|(t, p)| Tensor::add(*t, *p))
            .collect();
        x = Self::rmsnorm(&x);

        for li in 0..self.config.n_layer {
            let p = format!("layer{li}.");

            // 1) Multi-head attention block
            let x_residual = x.clone();
            x = Self::rmsnorm(&x);
            let q = Self::linear(&x, self.get(&format!("{p}attn_wq")));
            let k = Self::linear(&x, self.get(&format!("{p}attn_wk")));
            let v = Self::linear(&x, self.get(&format!("{p}attn_wv")));
            self.keys[li].push(k);
            self.values[li].push(v);

            let mut x_attn: Vec<TensorHandle> = Vec::with_capacity(self.config.n_embd);
            for h in 0..self.config.n_head {
                let hs = h * self.config.head_dim;
                let q_h = &q[hs..hs + self.config.head_dim];
                let k_h: Vec<&[TensorHandle]> = self.keys[li]
                    .iter()
                    .map(|ki| &ki[hs..hs + self.config.head_dim])
                    .collect();
                let v_h: Vec<&[TensorHandle]> = self.values[li]
                    .iter()
                    .map(|vi| &vi[hs..hs + self.config.head_dim])
                    .collect();

                // The division by sqrt(head_dim) is not optional: a dot product
                // grows like sqrt(d), and without it softmax saturates to a hard
                // argmax and most dimensions receive no gradient.
                let inv_sqrt_head = (self.config.head_dim as f32).sqrt();
                let attn_logits: Vec<TensorHandle> = k_h
                    .iter()
                    .map(|kt| {
                        Tensor::div_scalar(Tensor::linear(q_h, kt), inv_sqrt_head)
                    })
                    .collect();
                let attn_weights = Self::softmax(&attn_logits);

                // A convex combination of the values, so the output magnitude
                // does not grow with sequence length.
                let head_out: Vec<TensorHandle> = (0..self.config.head_dim)
                    .map(|j| Tensor::weighted_sum(&attn_weights, &v_h, j))
                    .collect();
                x_attn.extend(head_out);
            }
            x = Self::linear(&x_attn, self.get(&format!("{p}attn_wo")));
            x = x
                .iter()
                .zip(x_residual.iter())
                .map(|(a, b)| Tensor::add(*a, *b))
                .collect();

            // 2) MLP block. The same four steps: save, normalise, transform, add.
            let x_residual = x.clone();
            x = Self::rmsnorm(&x);
            x = Self::linear(&x, self.get(&format!("{p}mlp_fc1")));
            x = x.iter().map(|v| Tensor::relu(*v)).collect();
            x = Self::linear(&x, self.get(&format!("{p}mlp_fc2")));
            x = x
                .iter()
                .zip(x_residual.iter())
                .map(|(a, b)| Tensor::add(*a, *b))
                .collect();
        }

        Self::linear(&x, self.get("lm_head"))
    }
}

// ─── Data ──────────────────────────────────────────────────────────────────

/// `input.txt` read as a list of documents, one per line.
///
/// Read from an explicit path rather than the current directory, and never
/// downloaded. The Python reference does both of those things; see issue 3 in
/// `docs/KNOWN-ISSUES.md`.
pub fn load_docs(path: &str) -> io::Result<Vec<String>> {
    let file = File::open(path)?;
    BufReader::new(file)
        .lines()
        .map(|line| line.map(|l| l.trim().to_string()))
        .filter(|line| matches!(line, Ok(ref l) if !l.is_empty()))
        .collect()
}

/// `[BOS] + chars + [BOS]`. The trailing BOS is the whole trick: it turns a
/// closed-ended task into an open-ended one, so the model learns where names
/// stop and can therefore learn to stop.
pub fn tokenize(doc: &str, char_index: &std::collections::HashMap<char, usize>, bos: usize) -> Vec<usize> {
    let mut tokens = Vec::with_capacity(doc.len() + 2);
    tokens.push(bos);
    for ch in doc.chars() {
        tokens.push(char_index[&ch]);
    }
    tokens.push(bos);
    tokens
}

// ─── PRNG ──────────────────────────────────────────────────────────────────

/// xoshiro256++ — the same generator the C port uses, so the two are comparable
/// for reasons other than the PRNG.
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        // SplitMix64 seeding, exactly as the C port does it.
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box–Muller, so the normal draws are comparable with the C port's.
    ///
    /// Computed in `f64` and narrowed once at the end, rather than in `f32`
    /// throughout. The uniforms come out of a 53-bit mantissa division, so
    /// squaring and taking a log of them in `f32` would throw away most of the
    /// precision for no benefit -- the model's own arithmetic is `f32`, but the
    /// *sampler* does not have to be.
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z as f32
    }

    /// Fisher–Yates, the same shuffle the reference and the C port perform.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.uniform() * (i + 1) as f64) as usize;
            items.swap(i, j.min(i));
        }
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

/// Everything the CLI can be told, in one place.
///
/// Returning a struct rather than setting a `Config` and hoping the caller reads
/// the rest off `std::env::args()` again: the first version of this parsed
/// `--seed` and `--input` into locals, threw them away, and then re-derived
/// `--input` by walking the arguments a second time. `--seed` was silently
/// ignored, so every run used seed 42 no matter what the caller asked for. The
/// compiler noticed, via "value assigned to `seed` is never read", which is
/// exactly what warnings are for.
struct Args {
    config: Config,
    input: String,
    seed: u64,
}

fn parse_args() -> Args {
    let mut config = Config::default();
    let mut input = String::from("../../data/input.txt");
    let mut seed: u64 = 42;

    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let flag = raw[i].as_str();
        let value = raw.get(i + 1).map(String::as_str);
        let mut next = |name: &str| -> Option<&str> {
            if flag == name {
                i += 1;
                value
            } else {
                None
            }
        };
        if let Some(v) = next("--input") {
            input = v.to_string();
        } else if let Some(v) = next("--seed").and_then(|s| s.parse().ok()) {
            seed = v;
        } else if let Some(v) = next("--steps").and_then(|s| s.parse().ok()) {
            config.num_steps = v;
        } else if let Some(v) = next("--n-embd").and_then(|s| s.parse().ok()) {
            config.n_embd = v;
        } else if let Some(v) = next("--n-head").and_then(|s| s.parse().ok()) {
            config.n_head = v;
        } else if let Some(v) = next("--n-layer").and_then(|s| s.parse().ok()) {
            config.n_layer = v;
        } else if let Some(v) = next("--block-size").and_then(|s| s.parse().ok()) {
            config.block_size = v;
        } else if let Some(v) = next("--vocab-size").and_then(|s| s.parse().ok()) {
            config.vocab_size = v;
        }
        i += 1;
    }

    config.head_dim = config.n_embd / config.n_head;
    Args {
        config,
        input,
        seed,
    }
}

/// The binary's entry point, so `src/main.rs` stays a one-liner and the model
/// remains reachable from tests.
pub fn run() {
    let args = parse_args();
    let input = args.input;
    let mut config = args.config;

    let mut docs = match load_docs(&input) {
        Ok(docs) => docs,
        Err(err) => {
            eprintln!("cannot read {input}: {err}");
            std::process::exit(1);
        }
    };

    // The vocabulary is the sorted set of every character in the dataset, so it
    // is the same on every machine and every run.
    let mut alphabet: Vec<char> = docs.iter().flat_map(|d| d.chars()).collect();
    alphabet.sort_unstable();
    alphabet.dedup();
    let uchars: Vec<char> = alphabet.clone();
    let char_index: std::collections::HashMap<char, usize> = uchars
        .iter()
        .enumerate()
        .map(|(i, c)| (*c, i))
        .collect();
    let bos = uchars.len();
    config.vocab_size = bos + 1;

    let mut rng = Rng::new(args.seed);
    rng.shuffle(&mut docs);

    let mut model = Model::new(config.clone(), &mut rng);
    let params = model.params();
    println!("num docs: {}", docs.len());
    println!("vocab size: {}", config.vocab_size);
    println!("num params: {}", params.len());

    // Adam, with the reference's betas. The bias correction is what makes the
    // first step the same size as every other one.
    const LEARNING_RATE: f32 = 0.012;
    const BETA1: f32 = 0.85;
    const BETA2: f32 = 0.98;
    const EPS_ADAM: f32 = 1e-8;
    const BATCH_SIZE: usize = 8;
    let mut moments = vec![0.0f32; params.len()];
    let mut velocities = vec![0.0f32; params.len()];

    let started = std::time::Instant::now();
    for step in 0..config.num_steps {
        let mut loss = Tensor::leaf(0.0);
        let mut token_count = 0usize;
        for batch_index in 0..BATCH_SIZE {
            let doc_index = (step * BATCH_SIZE + batch_index) % docs.len();
            let doc = &docs[doc_index];
            let tokens = tokenize(doc, &char_index, bos);
            let n = config.block_size.min(tokens.len() - 1);

            model.reset_cache();
            let mut losses = Vec::with_capacity(n);
            for pos_id in 0..n {
                let logits = model.forward(tokens[pos_id], pos_id);
                let max_logit = logits
                    .iter()
                    .map(|logit| Tensor::data(*logit))
                    .fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<TensorHandle> = logits
                    .iter()
                    .map(|logit| Tensor::shifted_exp(*logit, max_logit))
                    .collect();
                let exp_sum = Tensor::sum(&exps);
                let shifted_target =
                    Tensor::sub_scalar(logits[tokens[pos_id + 1]], max_logit);
                let nll = Tensor::sub(Tensor::log(exp_sum), shifted_target);
                losses.push(nll);
            }
            for l in &losses {
                loss = Tensor::add(loss, *l);
            }
            token_count += n;
        }
        let loss = Tensor::div_scalar(loss, token_count as f32);

        Tensor::backward(loss);

        let lr_t = LEARNING_RATE * (1.0 - step as f32 / config.num_steps as f32);
        let beta1_power = BETA1.powi(step as i32 + 1);
        let beta2_power = BETA2.powi(step as i32 + 1);
        ARENA.with(|a| {
            let arena = &mut a.borrow_mut();
            for (i, p) in params.iter().enumerate() {
                let node = &mut arena.nodes[p.0];
                let g = node.grad;
                moments[i] = BETA1 * moments[i] + (1.0 - BETA1) * g;
                velocities[i] = BETA2 * velocities[i] + (1.0 - BETA2) * g * g;
                let m_hat = moments[i] / (1.0 - beta1_power);
                let v_hat = velocities[i] / (1.0 - beta2_power);
                node.data -= lr_t * m_hat / (v_hat.sqrt() + EPS_ADAM);
                node.grad = 0.0;
            }
        });

        println!(
            "step {:4} / {:4} | loss {:.4}",
            step + 1,
            config.num_steps,
            Tensor::data(loss)
        );

        // Model parameters are the first arena nodes; discard the graph and its
        // allocations before the next step so the working set stays small.
        ARENA.with(|a| {
            let mut arena = a.borrow_mut();
            arena.nodes.truncate(params.len());
            arena.visited.truncate(params.len());
        });
    }

    let elapsed = started.elapsed();
    println!("\n--- inference (new, hallucinated names) ---");
    let temperature: f32 = 0.5;
    for sample_idx in 0..20 {
        model.reset_cache();
        let mut token_id = bos;
        let mut out = String::new();
        for pos_id in 0..config.block_size {
            let logits = model.forward(token_id, pos_id);
            // Temperature goes outside softmax, on the logits -- softmax itself
            // has no notion of it.
            let scaled: Vec<TensorHandle> = logits
                .iter()
                .map(|l| Tensor::div_scalar(*l, temperature))
                .collect();
            let probs = Model::softmax(&scaled);
            let weights: Vec<f32> = probs.iter().map(|p| Tensor::data(*p)).collect();
            let total: f32 = weights.iter().sum();
            let mut target = rng.uniform() as f32 * total;
            let mut chosen = weights.len() - 1;
            for (i, w) in weights.iter().enumerate() {
                target -= w;
                if target <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            token_id = chosen;
            if token_id == bos {
                break;
            }
            out.push(uchars[token_id]);
        }
        println!("sample {:2}: {}", sample_idx + 1, out);
    }

    let _ = io::stdout().flush();
    println!(
        "\nTotal time: {:.3} ms ({:.1} steps/sec)",
        elapsed.as_secs_f64() * 1000.0,
        config.num_steps as f64 / elapsed.as_secs_f64()
    );
}
