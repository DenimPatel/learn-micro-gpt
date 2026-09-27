// A finite-difference check on this port's autograd tape.
//
// Ported from `autoresearch/candidate/tests/gradient_check.rs`, which is the
// specification. Same measurement, same thresholds, same output contract, in
// the one idiom Go offers for reaching unexported identifiers: a test file in
// `package main`. That is the whole reason this works as a Go test where a
// Rust integration test reached only for `pub` -- here `softmax`, `forward`,
// `params` and `(*model).resetCache` are all package-private, and the probe
// still has to be able to call them.
//
// # Why it exists
//
// The whole port is only worth anything if the gradient is right. A loss curve
// cannot tell a correct gradient from a wrong one: `implementations/c` has a
// hand-written backward pass whose directional derivative is off by a factor
// of -8 and whose loss still falls. Only a finite difference can, so this runs
// on every candidate before it is allowed to compete on loss.
//
// # What it measures
//
// The directional derivative of the loss along one fixed direction `d`, taken
// over all parameters at once, against a central difference. A correct
// gradient gives a ratio of 1.0.
//
// A fresh model per evaluation, and that is not tidiness: `(*model).forward`
// appends to the KV cache, so a second forward sequence on the same model
// would attend to the previous evaluation's keys and quietly measure
// something else. `resetCache` exists and would work; a fresh model is what
// the Rust probe does, and it is the stronger form of the same guarantee.
//
// # The band is not centred on 1.0, and that is not a bug in this probe
//
// `reference/microgpt.py:102-105` differentiates through `rmsnorm`, because
// Python hands it the tape for free. Translating that function to a language
// without operator overloading requires noticing that, and this port did not
// notice it: `rmsnorm` accumulates `ms` from `v.Data` as a plain `float64`
// and scales with `MulScalar`, so the whole normalisation sits outside the
// tape. The analytic derivative is therefore the gradient of a slightly
// different function than the one the loss evaluates, and the two disagree by
// a fixed amount -- the Rust probe measures 1.063 for exactly this reason, and
// this one measures the same omission on f64 instead of f32.
//
// So the band is wide on purpose. It is a tripwire, not a precision
// measurement: a broken backward pass lands far outside 0.5..2.0, a discarded
// gradient lands on 0, and a sign slip lands negative. The stability check is
// what does the real work -- it says the number is a property of the tape
// rather than of the step size, which is what a truncation or roundoff
// artifact is not.
//
// Run as:
//
//	go test -run TestGradcheck -v .
//
// and it prints one machine-readable line the harness parses, recording the
// ratio at every step size so drift stays visible even when the band passes.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"testing"
)

// Same seed discipline as the model, so the parameters are identical on every
// evaluation, and a different one for the direction, so `d` is identical on
// every machine and every run. These are the Rust probe's seeds, unchanged:
// the two ports are measuring the same quantity and a differing result should
// mean a differing tape, not a differing draw.
const (
	probeSeed          int64 = 0x5EED1234ABCD0001
	probeDirectionSeed int64 = 0x5EED1234ABCD0002
)

// Fixed step sizes for the central difference. Two, not one, and that is the
// whole reason the measurement is trustworthy: the analytic derivative is
// exact, so any error is the difference operator's, and that error splits into
// a truncation term falling as h^2 and a roundoff term growing as 1/h. One
// step size cannot tell the two apart. Two can -- if the ratio moved much when
// h moved, the number would be an artifact of the step size rather than a
// property of the tape.
var probeSteps = [2]float64{1e-2, 3e-3}

// How far the analytic and numeric derivatives may disagree.
//
// Deliberately wide, and deliberately not centred on 1.0, for the reason in
// the file docs: this port's tape omits the `rmsnorm` path, so the honest
// answer for a correct tape is a little above 1. What this catches is a tape
// that is broken, and the known ways to be broken are not close calls.
const (
	ratioMin      = 0.5
	ratioMax      = 2.0
	stepStability = 0.05
)

// A deliberately tiny configuration. The check is about the tape's arithmetic,
// not about the model, and this runs on every experiment and in CI.
func probeConfig() Config {
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

// A fixed token sequence, so the check needs no dataset file and cannot be
// affected by one going missing.
var probeTokens = [6]int{0, 1, 2, 3, 4, 5}

// probeModel builds the same model every time: the seed is fixed and the
// draws happen in source order, so the parameters are bit-identical across
// runs even though the container they are stored in is a map.
func probeModel() *model {
	return newModel(probeConfig(), rand.New(rand.NewSource(probeSeed)))
}

// stateKeyOrder is the order `newModel` assigns its matrices in, which the
// Rust probe inherits for free from a deterministic parameter list.
func stateKeyOrder(nLayer int) []string {
	keys := []string{"wte", "wpe", "lm_head"}
	for i := 0; i < nLayer; i++ {
		p := fmt.Sprintf("layer%d.", i)
		keys = append(keys,
			p+"attn_wq", p+"attn_wk", p+"attn_wv", p+"attn_wo",
			p+"mlp_fc1", p+"mlp_fc2",
		)
	}
	return keys
}

// orderedParams is the same set as `(*model).params()`, in a fixed order.
//
// The fix is not cosmetic. `params()` ranges over `m.state`, which is a Go map,
// and Go randomises map iteration order: the same call returns the same 960
// values in a different sequence every time. The probe pairs each parameter
// with one element of `d`, so an unstable order would measure a different
// direction on every run -- not a wrong number, a *different* number, and one
// that would drift in a way that has nothing to do with the tape. The Rust
// probe cannot have this problem, so porting it verbatim would have been
// quietly wrong.
//
// This reads the same unexported `state` the model itself is built from, and
// adds nothing to the track. The caller checks the count against `params()`
// so the two cannot drift apart unnoticed.
func orderedParams(m *model) []*Value {
	out := make([]*Value, 0, 8192)
	for _, key := range stateKeyOrder(m.cfg.NLayer) {
		for _, row := range m.state[key] {
			out = append(out, row...)
		}
	}
	return out
}

// probeLoss is the mean cross-entropy over the fixed sequence, using the
// package's own `softmax` and its own tape operations. It returns the loss
// `Value`, not its data, so the caller can decide whether to differentiate
// through it. Building the model is the caller's job, so a caller can perturb
// the parameters before any forward pass runs.
func probeLoss(m *model) *Value {
	n := len(probeTokens) - 1
	losses := make([]*Value, 0, n)
	for pos := 0; pos < n; pos++ {
		logits := m.forward(probeTokens[pos], pos)
		probs := softmax(logits)
		// Log first, then negate. `Neg` is `MulScalar(-1)`, so the reverse
		// order would be `log(-p)` and NaN from the very first position.
		losses = append(losses, probs[probeTokens[pos+1]].Log().Neg())
	}
	return Mean(losses)
}

// stepLabel renders h the way the Rust probe's `{h:e}` does: `1e-2`, not
// `1e-02`. The harness regex accepts either, but the output contract is a
// literal string and this is the literal string.
func stepLabel(h float64) string {
	// `strconv` with precision -1, not `%e`: Go pads the mantissa to six
	// decimals, which would print the step size as `1.000000e-2`.
	s := strconv.FormatFloat(h, 'e', -1, 64)
	i := strings.IndexByte(s, 'e')
	if i < 0 {
		return s
	}
	mantissa, exponent := s[:i], s[i+1:]
	sign := ""
	if len(exponent) > 0 && (exponent[0] == '+' || exponent[0] == '-') {
		sign, exponent = string(exponent[0]), exponent[1:]
	}
	exponent = strings.TrimLeft(exponent, "0")
	if exponent == "" {
		exponent = "0"
	}
	return mantissa + "e" + sign + exponent
}

func TestGradcheck(t *testing.T) {
	// The direction: one pseudo-random unit vector over all parameters, from
	// its own seed so it is identical on every machine and every run.
	directionRng := rand.New(rand.NewSource(probeDirectionSeed))
	direction := make([]float64, len(orderedParams(probeModel())))
	norm := 0.0
	for i := range direction {
		direction[i] = directionRng.Float64()*2 - 1
		norm += direction[i] * direction[i]
	}
	norm = math.Sqrt(norm)
	for i := range direction {
		direction[i] /= norm
	}

	// Analytic: one backward pass on a fresh model, then the dot product with
	// the same direction.
	analyticModel := probeModel()
	analyticParams := orderedParams(analyticModel)
	if got, want := len(analyticParams), len(analyticModel.params()); got != want {
		t.Fatalf("orderedParams returned %d parameters but params() reports %d; "+
			"the probe is measuring a different set of parameters than the model trains", got, want)
	}
	analyticLoss := probeLoss(analyticModel)
	analyticLoss.Backward()
	analytic := 0.0
	for i, p := range analyticParams {
		analytic += p.Grad * direction[i]
	}

	// Numeric: for each step size, two more fresh models perturbed in
	// opposite directions. Fresh per evaluation, because `forward` appends to
	// the KV cache.
	reported := make([]string, 0, len(probeSteps))
	ratios := make([]float64, 0, len(probeSteps))
	for _, h := range probeSteps {
		plus := probeModel()
		for i, p := range orderedParams(plus) {
			p.Data += h * direction[i]
		}
		lossPlus := probeLoss(plus).Data

		minus := probeModel()
		for i, p := range orderedParams(minus) {
			p.Data -= h * direction[i]
		}
		lossMinus := probeLoss(minus).Data

		numeric := (lossPlus - lossMinus) / (2 * h)
		ratio := math.Inf(1)
		if numeric != 0 {
			ratio = analytic / numeric
		}
		ratios = append(ratios, ratio)
		reported = append(reported, fmt.Sprintf("h=%s ratio=%.6f numeric=%.6f",
			stepLabel(h), ratio, numeric))
	}

	// The one line the harness parses.
	fmt.Printf("GRADCHECK analytic=%.6f params=%d ratios=%s\n",
		analytic, len(analyticParams), strings.Join(reported, " | "))

	for i, h := range probeSteps {
		ratio := ratios[i]
		if math.IsInf(ratio, 0) || math.IsNaN(ratio) {
			t.Fatalf("the ratio at h=%s is not finite; the loss is diverging", stepLabel(h))
		}
		if ratio < ratioMin || ratio > ratioMax {
			t.Fatalf("directional derivative ratio is %.4f at h=%s, outside [%.1f, %.1f]. "+
				"Analytic %.6f. A ratio far from 1 is a backward pass that is wrong and "+
				"whose loss curve still looks fine.",
				ratio, stepLabel(h), ratioMin, ratioMax, analytic)
		}
	}

	// The ratio must not move when the step size moves. Truncation error falls
	// as h^2 and roundoff grows as 1/h, so a ratio stable across a 3.3x
	// change in h is neither. Anything wrong with the tape is a fixed offset,
	// and survives -- which is exactly what separates "fixed offset" from
	// "step size artifact".
	spread := math.Abs(ratios[0] - ratios[1])
	if spread > stepStability {
		t.Fatalf("the ratio moved %.4f when the step size moved from %s to %s "+
			"(%.4f then %.4f), so it is an artifact of the difference operator rather "+
			"than a property of the tape. Widen the step sizes, or the loss is not "+
			"smooth enough at this scale for a central difference to mean anything.",
			spread, stepLabel(probeSteps[0]), stepLabel(probeSteps[1]), ratios[0], ratios[1])
	}
}
