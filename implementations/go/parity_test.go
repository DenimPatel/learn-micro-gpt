// Parity and gradient tests for the frozen Go track.
//
// ## Why this file exists, and why it is named `parity_test.go`
//
// The track had no tests at all, so `go test ./...` printed "no test files" and
// proved nothing. This gives it real ones.
//
// The name matters. `autoresearch` measures this track by diffing it against
// the tuned candidate, and it pins the candidate's probe by path:
// `tools/autoresearch.py` sets `probe_path = "probe_test.go"` for the Go track.
// A test file at that path in *this* directory would be read as the frozen
// track gaining the candidate's probe, and
// `test_every_probe_lives_in_the_candidate_and_not_the_baseline` exists to fail
// on exactly that. So the frozen track's tests live in `parity_test.go`, the
// same way the Rust and TypeScript tracks keep theirs in `parity.rs` and
// `parity.test.ts` while their probes are `gradient_check.rs` and `probe.ts`.
//
// This file is therefore *not* a copy of the candidate probe. It is the
// operator-level counterpart: where the probe measures one directional
// derivative of the whole loss, these check each operator's derivative directly,
// which is what localises a break the aggregate number would only smear.
//
// ## What is checked, and why each earns its place
//
//  1. **Every operator, by finite difference.** The value of an autograd library
//     is that `Backward` produces the derivative of the forward function. A sign
//     error or a dropped term still compiles, still trains, and shows up in no
//     loss curve -- `implementations/c` shipped a backward pass off by -8x that
//     way. Perturbing an input and re-evaluating the forward function gives a
//     definitionally-correct number, and no internal cleverness can fake it.
//
//     Each case composes several operators rather than testing one in isolation,
//     so a mistake in any of them shows up in at least one graph. `Pow` and
//     `Div` are the ones to watch: both go through `Pow`, which uses
//     `n * data**(n-1)` and is therefore `NaN` at a base of zero for negative n.
//
//  2. **Shared subgraphs accumulate.** `(x+x)*(x+x)` reuses the same node on both
//     sides, so its gradient is `8x`, not `4x`. `Backward` uses `+=`, and this is
//     the test that would catch someone "simplifying" it to `=`.
//
//  3. **The known `rmsnorm` omission, pinned.** `rmsnorm` normalises with a plain
//     `float64` scale applied via `MulScalar`, so the mean sits outside the tape
//     and the analytic gradient is that of a slightly different function than the
//     loss evaluates. `autoresearch/candidate-go` fixes this and its probe then
//     measures ~1.0 while this track measures ~1.127 for the same direction. That
//     gap is the whole experiment, so it is asserted here rather than left as a
//     comment somebody would "fix" the band around.
//
//  4. **The forward pass is deterministic for a fixed seed.** The model's weights
//     live in a Go map, and Go randomises map iteration order, so anything that
//     ranged over `state` instead of the fixed `stateKeyOrder` would produce
//     different parameters -- and thus a different loss -- run to run.
//
// Run as:
//
//	go test ./...
//
// in `implementations/go`. `go test` compiles the file as part of `package main`,
// which is the only way a test reaches the unexported `Value`, `model` and
// `softmax` this track keeps private on purpose.
package main

import (
	"math"
	"math/rand"
	"testing"
)

// ─── The finite-difference harness ─────────────────────────────────────────

// centralDifference returns (f(x+h) - f(x-h)) / (2h) for the composed function
// `build`, rebuilding the whole graph from a fresh leaf each time.
//
// Fresh leaves are not tidiness: `Backward` accumulates into `Grad` and never
// clears it, so reusing a leaf would add this probe's gradient to the previous
// one's and measure the sum. A new `Value` per evaluation gives `Grad` its zero
// value, which is the state a caller legitimately starts from.
func centralDifference(build func(*Value) *Value, x, h float64) float64 {
	plus := build(New(x + h))
	plus.Backward()
	minus := build(New(x - h))
	minus.Backward()
	return (plus.Data - minus.Data) / (2 * h)
}

// analyticGradient returns d(build)/dx at x, by a single backward pass on a
// fresh leaf.
func analyticGradient(build func(*Value) *Value, x float64) float64 {
	leaf := New(x)
	build(leaf).Backward()
	return leaf.Grad
}

// assertGradMatches checks the analytic gradient against a central difference at
// x, using a relative tolerance scaled by the magnitude of the analytic value so
// the check means the same thing at 1e-3 and at 1e3.
//
// Two step sizes, and the smaller one twice: a central difference has a
// truncation error falling as h^2 and a roundoff error growing as 1/h, and if
// the two steps disagreed the number would be an artifact of the step size
// rather than a property of the tape.
func assertGradMatches(t *testing.T, build func(*Value) *Value, x float64) {
	t.Helper()
	analytic := analyticGradient(build, x)
	for _, h := range []float64{1e-4, 1e-5} {
		numeric := centralDifference(build, x, h)
		tolerance := 1e-4 * math.Max(1, math.Abs(analytic))
		if math.Abs(analytic-numeric) > tolerance {
			t.Fatalf("d/dx at x=%g via h=%g: analytic %.8f, finite difference %.8f "+
				"(|diff| %.3g > tolerance %.3g); a backward pass that disagrees with "+
				"the forward function is the one bug a loss curve cannot show",
				x, h, analytic, numeric, math.Abs(analytic-numeric), tolerance)
		}
	}
}

// ─── Per-operator gradients ────────────────────────────────────────────────

// Each `build` composes several operators, so a mistake in any one of them
// lands in at least one of these graphs.
func TestGradientsThroughEveryOperator(t *testing.T) {
	cases := []struct {
		name  string
		build func(*Value) *Value
	}{
		{"add and mul", func(x *Value) *Value { return x.Mul(x).Add(x.MulScalar(2)) }},
		{"sub and add", func(x *Value) *Value { return x.Sub(x.MulScalar(0.5)).AddScalar(1) }},
		{"exp, pow and log", func(x *Value) *Value { return x.Exp().Add(x.Pow(2)).Log() }},
		{"pow and exp", func(x *Value) *Value { return x.Pow(2).Exp() }},
		{"relu and mul", func(x *Value) *Value { return x.ReLU().Mul(x) }},
		{"div and addScalar", func(x *Value) *Value { return x.Mul(x).Sub(x).Div(x.AddScalar(2)) }},
		{"scalar helpers", func(x *Value) *Value { return x.DivScalar(2).SubScalar(0.5).MulScalar(3) }},
	}
	// Positive points only: `log` and `pow` with a fractional exponent are
	// undefined below zero, and `div` above divides by x+2, so the domain is
	// x > -2. Zero is excluded because it sits on `pow`'s NaN edge.
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			for _, x := range []float64{0.25, 0.5, 1.0, 1.5, 2.75} {
				assertGradMatches(t, tc.build, x)
			}
		})
	}
}

// ─── The shared-subgraph case ──────────────────────────────────────────────

// `(x + x) * (x + x)` reuses the sum node twice, so the gradient has to
// accumulate through both uses: f = (2x)^2 = 4x^2, f' = 8x. `Backward` uses
// `+=` for exactly this, and a `=` would still pass the single-use cases above
// while silently halving every gradient in a graph that reuses a node -- which
// the model does on every forward pass.
func TestGradientAccumulatesOverASharedSubgraph(t *testing.T) {
	build := func(x *Value) *Value {
		sum := x.Add(x)
		return sum.Mul(sum)
	}
	for _, x := range []float64{-2, 0, 1.5, 3} {
		assertGradMatches(t, build, x)
	}
}

// ─── The known omission, pinned ────────────────────────────────────────────

// `rmsnorm` here normalises half outside the tape. It computes
// `scale = (mean(x^2) + eps)^-0.5` as a plain float64 and applies it with
// `MulScalar`, so `Backward` sees a multiply by a *constant* and never learns
// that `scale` depends on `x` -- the `dscale/dx` term is dropped.
//
// The consequence is sharp and worth pinning: the analytic gradient of
// `sum(rmsnorm(x))` w.r.t. every `x_i` comes out identical (it is just `scale`),
// while the true derivative differs per element because a larger `|x_i|` both
// scales its own output up and shrinks the shared scale. The directional
// derivative of the whole loss is therefore off by a fixed factor -- ~1.127 on
// this track versus ~1.0 for the tuned `autoresearch/candidate-go`, which builds
// the mean as a `Value` with local gradients attached. That gap is the
// experiment's finding, not a bug to paper over, so it is asserted here rather
// than left as a comment somebody would "fix" by moving the band.
func TestRMSNormGradientDropsTheScaleTerm(t *testing.T) {
	xs := []float64{0.5, -1.25, 2.0}

	// Analytic: one backward pass over the sum of the outputs.
	leaves := make([]*Value, len(xs))
	for i := range xs {
		leaves[i] = New(xs[i])
	}
	outs := rmsnorm(leaves)
	Sum(outs).Backward()

	// The forward value is right: each output is x_i * scale.
	ms := 0.0
	for _, v := range xs {
		ms += v * v
	}
	ms /= float64(len(xs))
	scale := math.Pow(ms+1e-5, -0.5)
	for i, o := range outs {
		if want := xs[i] * scale; math.Abs(o.Data-want) > 1e-12 {
			t.Fatalf("rmsnorm forward value %d is %.12f, want %.12f; the test is "+
				"measuring the wrong function", i, o.Data, want)
		}
	}

	// The omission: every gradient equals `scale`, independent of x_i.
	for i, l := range leaves {
		if math.Abs(l.Grad-scale) > 1e-12 {
			t.Fatalf("x[%d] analytic gradient is %.12f, not the constant scale "+
				"%.12f. If rmsnorm has been put back on the tape this is the intended "+
				"signal that the recorded 1.127 / 1.0 gap no longer holds -- update "+
				"the record rather than reverting this test.", i, l.Grad, scale)
		}
	}

	// And the true derivative is provably not constant, which is what makes the
	// equality above an omission rather than a coincidence. Central difference on
	// sum(rmsnorm(x)) at each coordinate.
	sumAt := func(x []float64) float64 {
		in := make([]*Value, len(x))
		for i := range x {
			in[i] = New(x[i])
		}
		total := 0.0
		for _, o := range rmsnorm(in) {
			total += o.Data
		}
		return total
	}
	trueGrads := make([]float64, len(xs))
	for i := range xs {
		const h = 1e-6
		plus := append([]float64(nil), xs...)
		plus[i] += h
		minus := append([]float64(nil), xs...)
		minus[i] -= h
		trueGrads[i] = (sumAt(plus) - sumAt(minus)) / (2 * h)
	}
	if math.Abs(trueGrads[0]-trueGrads[1]) < 1e-6 {
		t.Fatalf("the true per-coordinate gradients came out equal (%v); the "+
			"forward function is not the rmsnorm this test assumes", trueGrads)
	}
}

// ─── Forward determinism ───────────────────────────────────────────────────

// The same seed must give bit-identical parameters, and therefore a
// bit-identical loss. `newModel` draws in source order and writes into a map, so
// this is a real check on the draw order rather than a tautology: a Go map
// ranges in randomised order, so a reader that ranged over `state` to rebuild
// the model would produce a different loss here on some runs and not others.
func TestForwardPassIsDeterministicForAFixedSeed(t *testing.T) {
	cfg := parityConfig()
	tokens := []int{0, 1, 2, 3, 4, 5}

	lossOf := func() float64 {
		m := newModel(cfg, rand.New(rand.NewSource(paritySeed)))
		return parityLoss(m, tokens)
	}

	first := lossOf()
	for i := 0; i < 5; i++ {
		if got := lossOf(); got != first {
			t.Fatalf("loss changed across identical runs: %.12f then %.12f at run %d; "+
				"the parameter draw is not deterministic, which is map iteration order "+
				"leaking into the model", first, got, i+1)
		}
	}
}

// A different seed must give a different loss, so the test above cannot pass by
// the model ignoring its seed entirely.
func TestForwardPassRespondsToTheSeed(t *testing.T) {
	cfg := parityConfig()
	tokens := []int{0, 1, 2, 3, 4, 5}
	a := parityLoss(newModel(cfg, rand.New(rand.NewSource(paritySeed))), tokens)
	b := parityLoss(newModel(cfg, rand.New(rand.NewSource(paritySeed+1))), tokens)
	if a == b {
		t.Fatalf("two different seeds gave the same loss %.12f; the seed is not "+
			"reaching the parameters", a)
	}
}

// ─── Helpers ───────────────────────────────────────────────────────────────

const paritySeed int64 = 0x5EED1234ABCD0001

// A deliberately tiny model: these checks are about the tape's arithmetic, not
// about the model, and this keeps `go test` well under a second.
func parityConfig() Config {
	return Config{
		NEmbd:     8,
		NHead:     2,
		NLayer:    1,
		BlockSize: 8,
		HeadDim:   4,
		VocabSize: 8,
		Steps:     1,
	}
}

// parityLoss is the mean cross-entropy over a fixed token sequence: the same
// loss the model trains on, on a tiny input, so the check needs no dataset file.
func parityLoss(m *model, tokens []int) float64 {
	losses := make([]*Value, 0, len(tokens)-1)
	for pos := 0; pos < len(tokens)-1; pos++ {
		logits := m.forward(tokens[pos], pos)
		probs := softmax(logits)
		losses = append(losses, probs[tokens[pos+1]].Log().Neg())
	}
	return Mean(losses).Data
}
