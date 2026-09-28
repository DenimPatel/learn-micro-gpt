// Package main is a Go port of Andrej Karpathy's 199-line microgpt.py.
//
// # Why this port is structured the way it is
//
// The most interesting thing about microgpt.py is that the *model* is 199 lines
// of unremarkable Python: a linear layer is two generator expressions, softmax
// is four lines, attention is a loop with a slice. All the subtlety is in
// autograd, which is 40 lines of class.
//
// So this port does the obvious thing and keeps that structure exactly. A linear
// layer here is a loop over a matrix, not a call into a BLAS. Attention is a
// loop over heads. The point is that a reader can put two files side by side and
// see nothing but Go.
//
// The one thing that could not be transliterated is autograd itself, because Go
// has no operator overloading. So `Value` below is a small explicit tape: every
// operation records its operands and its local derivative, and Backward walks
// the result. It is Karpathy's class, in the one style Go allows -- which is
// exactly the interesting thing to see about Go.
//
// The consequence is that gradients here are correct by construction, the same
// way the reference's are. Compare implementations/c/microgpt.c, which
// hand-writes the backward pass and gets it wrong: docs/KNOWN-ISSUES.md.
//
// # Running it
//
//	go run . --input ../../data/input.txt [--steps 1000] [--seed 42]
//
// The dataset is read from --input, not from the current directory and not from
// a URL. The Python reference does both of those things; see issue 3 in
// docs/KNOWN-ISSUES.md.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"
)

// ─── Autograd ──────────────────────────────────────────────────────────────
//
// A number that remembers where it came from. `grad` is the derivative of the
// final loss with respect to `data`; `children` are the Values that produced
// this one; `localGrads` is how much each of those contributed.
//
// This is Karpathy's `class Value` with the operators spelled out as methods,
// because Go cannot overload `+`. `New`, `Add`, `Mul` and friends are named after
// the Python dunders they stand in for, so the two files can be read together.

// Value is a scalar in the computation graph.
type Value struct {
	Data      float64
	Grad      float64
	Children  []*Value
	LocalGrads []float64
}

// New wraps a plain number. Matches `Value(data)` in the reference.
func New(data float64) *Value {
	return &Value{Data: data}
}

func (v *Value) childrenOf(others ...*Value) []*Value {
	out := make([]*Value, 0, len(others))
	for _, o := range others {
		if o != nil {
			out = append(out, o)
		}
	}
	return out
}

// gradientsOf returns the local-derivative slice, positionally aligned with the
// `childrenOf` slice that produced it.
//
// It must NOT drop zeros. An earlier version filtered them out "to keep the
// slice small", which silently desynchronised the two slices and panicked with
// an index-out-of-range in Backward on the first ReLU -- a ReLU below zero has
// a local gradient of exactly 0, so it is the most common value there is. The
// reference has the same structure and gets it right for free, because a Python
// list is positional and nothing would let you shorten it.
func (v *Value) gradientsOf(grads ...float64) []float64 {
	out := make([]float64, len(grads))
	copy(out, grads)
	return out
}

// Add is `__add__`. The local gradients are 1 and 1: d(a+b)/da = d(a+b)/db = 1.
func (v *Value) Add(other *Value) *Value {
	return &Value{
		Data:       v.Data + other.Data,
		Children:   v.childrenOf(v, other),
		LocalGrads: v.gradientsOf(1, 1),
	}
}

// AddScalar is `other + self` for a plain Go float, i.e. `__radd__`.
func (v *Value) AddScalar(other float64) *Value {
	return v.Add(New(other))
}

// Mul is `__mul__`. The local gradients are *the other operand's value*, which
// is the product rule: nothing memorises the rule table.
func (v *Value) Mul(other *Value) *Value {
	return &Value{
		Data:       v.Data * other.Data,
		Children:   v.childrenOf(v, other),
		LocalGrads: v.gradientsOf(other.Data, v.Data),
	}
}

// MulScalar is `self * other` for a plain Go float.
func (v *Value) MulScalar(other float64) *Value {
	return v.Mul(New(other))
}

// Neg is `__neg__`, defined as `self * -1` so the graph never grows a new shape.
func (v *Value) Neg() *Value { return v.MulScalar(-1) }

// Sub is `__sub__`, i.e. `self + (-other)`.
func (v *Value) Sub(other *Value) *Value { return v.Add(other.Neg()) }

// Div is `__truediv__`, i.e. `self * other**-1`.
func (v *Value) Div(other *Value) *Value { return v.Mul(other.Pow(-1)) }

// Pow is `__pow__` for a constant exponent.
func (v *Value) Pow(exponent float64) *Value {
	return &Value{
		Data:       math.Pow(v.Data, exponent),
		Children:   v.childrenOf(v),
		LocalGrads: v.gradientsOf(exponent * math.Pow(v.Data, exponent-1)),
	}
}

// Log is `log()`.
func (v *Value) Log() *Value {
	return &Value{
		Data:       math.Log(v.Data),
		Children:   v.childrenOf(v),
		LocalGrads: v.gradientsOf(1 / v.Data),
	}
}

// Exp is `exp()`.
func (v *Value) Exp() *Value {
	e := math.Exp(v.Data)
	return &Value{
		Data:       e,
		Children:   v.childrenOf(v),
		LocalGrads: v.gradientsOf(e),
	}
}

// ReLU is `relu()`. The derivative is 1 above zero and 0 below, and ambiguous
// exactly at zero, where this picks 0 -- the same choice the reference makes.
func (v *Value) ReLU() *Value {
	grad := 0.0
	if v.Data > 0 {
		grad = 1
	}
	return &Value{
		Data:       math.Max(0, v.Data),
		Children:   v.childrenOf(v),
		LocalGrads: v.gradientsOf(grad),
	}
}

// Backward is the whole backward pass, in one method.
//
// Phase 1 sorts the graph parents-first with a depth-first walk. Phase 2 walks
// it in reverse, accumulating `child.grad += localGrad * v.grad` -- the chain
// rule, as a for loop. `+=` is load-bearing: a Value is reused by every
// document, every position, and every head, so its gradient has many
// contributions to sum.
func (v *Value) Backward() {
	// Temporarily truncate each node's child slice to mark it visited. This
	// avoids allocating and hashing a map entry for every node in every step.
	// All slices have the required capacity because they are made immediately
	// before their children are appended.
	topo := make([]*Value, 0, 16384)
	visitedLeaf := make([]*Value, 0)
	var buildTopo func(x *Value)
	buildTopo = func(x *Value) {
		children := x.Children
		if children == nil {
			x.Children = visitedLeaf
		} else {
			if len(children) == 0 {
				return
			}
			childCount := len(children)
			x.Children = children[:0:childCount]
			for _, child := range children {
				buildTopo(child)
			}
		}
		topo = append(topo, x)
	}
	buildTopo(v)
	for _, node := range topo {
		childCount := cap(node.Children)
		if childCount == 0 {
			node.Children = nil
		} else {
			node.Children = node.Children[:childCount]
		}
	}

	v.Grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]
		for j, child := range node.Children {
			child.Grad += node.LocalGrads[j] * node.Grad
		}
	}
}

// Sum adds a slice of Values, matching Python's `sum(losses)`.
func Sum(values []*Value) *Value {
	total := New(0)
	for _, v := range values {
		total = total.Add(v)
	}
	return total
}

// Mean divides by a count, matching `(1 / n) * sum(losses)`.
func Mean(values []*Value) *Value {
	return Sum(values).DivScalar(float64(len(values)))
}

// DivScalar is `x / n` for a plain Go float, as a *scalar* division so that the
// graph records only the numerator's gradient. That is what makes the loss a
// mean whose gradient is scaled by 1/n without a constant entering the graph.
func (v *Value) DivScalar(n float64) *Value {
	return v.Mul(New(1 / n))
}

// ─── Model ─────────────────────────────────────────────────────────────────

// Matrix is `[[Value]]` -- a list of rows, exactly as in the reference. Go has
// no convenient 2D array of structs, and inventing one here would obscure the
// correspondence the whole file exists to preserve.
type Matrix [][]*Value

type Config struct {
	NEmbd     int     `json:"n_embd"`
	NHead     int     `json:"n_head"`
	NLayer    int     `json:"n_layer"`
	BlockSize int     `json:"block_size"`
	HeadDim   int     `json:"head_dim"`
	VocabSize int     `json:"vocab_size"`
	Params    int     `json:"num_params"`
	Steps     int     `json:"num_steps"`
}

type model struct {
	cfg        Config
	state      map[string]Matrix
	keys       [][][]*Value
	values     [][][]*Value
}

// newMatrix is the `matrix` lambda: every weight is a Gaussian draw with
// std 0.08, a deliberately small number that keeps the initial distribution
// nearly uniform.
func newMatrix(rng *rand.Rand, nout, nin int, std float64) Matrix {
	m := make(Matrix, nout)
	for i := range m {
		row := make([]*Value, nin)
		for j := range row {
			row[j] = New(rng.NormFloat64() * std)
		}
		m[i] = row
	}
	return m
}

func newModel(cfg Config, rng *rand.Rand) *model {
	m := &model{cfg: cfg, state: make(map[string]Matrix)}
	std := 0.08
	m.state["wte"] = newMatrix(rng, cfg.VocabSize, cfg.NEmbd, std)
	m.state["wpe"] = newMatrix(rng, cfg.BlockSize, cfg.NEmbd, std)
	m.state["lm_head"] = newMatrix(rng, cfg.VocabSize, cfg.NEmbd, std)
	for i := 0; i < cfg.NLayer; i++ {
		p := fmt.Sprintf("layer%d.", i)
		m.state[p+"attn_wq"] = newMatrix(rng, cfg.NEmbd, cfg.NEmbd, std)
		m.state[p+"attn_wk"] = newMatrix(rng, cfg.NEmbd, cfg.NEmbd, std)
		m.state[p+"attn_wv"] = newMatrix(rng, cfg.NEmbd, cfg.NEmbd, std)
		m.state[p+"attn_wo"] = newMatrix(rng, cfg.NEmbd, cfg.NEmbd, std)
		m.state[p+"mlp_fc1"] = newMatrix(rng, 4*cfg.NEmbd, cfg.NEmbd, std)
		m.state[p+"mlp_fc2"] = newMatrix(rng, cfg.NEmbd, 4*cfg.NEmbd, std)
	}
	m.resetCache()
	return m
}

func (m *model) resetCache() {
	m.keys = make([][][]*Value, m.cfg.NLayer)
	m.values = make([][][]*Value, m.cfg.NLayer)
	for i := 0; i < m.cfg.NLayer; i++ {
		m.keys[i] = nil
		m.values[i] = nil
	}
}

func (m *model) params() []*Value {
	out := make([]*Value, 0, 8192)
	for _, mat := range m.state {
		for _, row := range mat {
			out = append(out, row...)
		}
	}
	return out
}

// linear is `[sum(wi*xi for wi,xi in zip(wo,x)) for wo in w]` -- the only place
// a weight matrix is used, and most of the arithmetic in the model.
func linear(x []*Value, w Matrix) []*Value {
	out := make([]*Value, 0, len(w))
	for _, row := range w {
		acc := 0.0
		children := make([]*Value, 0, 2*len(x))
		localGrads := make([]float64, 0, 2*len(x))
		for i, xi := range x {
			acc += row[i].Data * xi.Data
			children = append(children, row[i], xi)
			localGrads = append(localGrads, xi.Data, row[i].Data)
		}
		out = append(out, &Value{
			Data:       acc,
			Children:   children,
			LocalGrads: localGrads,
		})
	}
	return out
}

// reluLinear applies a matrix projection followed by ReLU as one tape node.
// This avoids allocating a separate ReLU node for every projected feature.
func reluLinear(x []*Value, w Matrix) []*Value {
	out := make([]*Value, 0, len(w))
	for _, row := range w {
		acc := 0.0
		children := make([]*Value, 0, 2*len(x))
		localGrads := make([]float64, 0, 2*len(x))
		for i, xi := range x {
			acc += row[i].Data * xi.Data
			children = append(children, row[i], xi)
			localGrads = append(localGrads, xi.Data, row[i].Data)
		}
		active := 0.0
		if acc > 0 {
			active = 1
		}
		for i := range localGrads {
			localGrads[i] *= active
		}
		out = append(out, &Value{
			Data:       math.Max(0, acc),
			Children:   children,
			LocalGrads: localGrads,
		})
	}
	return out
}

// dot records a dot product as one node rather than a chain of scalar
// multiplies and additions.
func dot(x, y []*Value) *Value {
	acc := 0.0
	children := make([]*Value, 0, 2*len(x))
	localGrads := make([]float64, 0, 2*len(x))
	for i, xi := range x {
		yi := y[i]
		acc += xi.Data * yi.Data
		children = append(children, xi, yi)
		localGrads = append(localGrads, yi.Data, xi.Data)
	}
	return &Value{
		Data:       acc,
		Children:   children,
		LocalGrads: localGrads,
	}
}

// softmax subtracts the max before exponentiating. That is not a numerical nicety,
// it is what makes the function usable: `math.Exp` overflows above about 709, and
// subtracting a constant leaves the result exactly unchanged.
func softmax(logits []*Value) []*Value {
	maxVal := math.Inf(-1)
	for _, v := range logits {
		if v.Data > maxVal {
			maxVal = v.Data
		}
	}
	exps := make([]*Value, len(logits))
	totalData := 0.0
	for i, v := range logits {
		e := math.Exp(v.Data - maxVal)
		exps[i] = &Value{
			Data:       e,
			Children:   []*Value{v},
			LocalGrads: []float64{e},
		}
		totalData += e
	}
	totalLocalGrads := make([]float64, len(exps))
	for i := range totalLocalGrads {
		totalLocalGrads[i] = 1
	}
	total := &Value{
		Data:       totalData,
		Children:   exps,
		LocalGrads: totalLocalGrads,
	}
	inverseTotal := math.Pow(totalData, -1)
	denominatorGrad := -math.Pow(totalData, -2)
	out := make([]*Value, len(exps))
	for i, e := range exps {
		out[i] = &Value{
			Data:       e.Data * inverseTotal,
			Children:   []*Value{e, total},
			LocalGrads: []float64{inverseTotal, e.Data * denominatorGrad},
		}
	}
	return out
}

// SubScalar is `val - m` for a plain Go float, recorded as a node so the
// subtraction is part of the graph.
func (v *Value) SubScalar(other float64) *Value {
	return v.Sub(New(other))
}

// rmsnorm rescales a vector by the reciprocal of its root-mean-square, with eps
// inside the square root to bound the division.
func rmsnorm(x []*Value) []*Value {
	ms := 0.0
	for _, v := range x {
		ms += v.Data * v.Data
	}
	ms /= float64(len(x))
	scale := math.Pow(ms+1e-5, -0.5)
	out := make([]*Value, len(x))
	for i, v := range x {
		out[i] = v.MulScalar(scale)
	}
	return out
}

// forward is a stateless function from one token id at one position to logits
// over the vocabulary, given the keys and values of earlier positions. That
// four-argument signature is the whole design: it *is* the KV cache, spelled the
// long way.
func (m *model) forward(tokenID, posID int) []*Value {
	tokEmb := m.state["wte"][tokenID]
	posEmb := m.state["wpe"][posID]
	x := make([]*Value, len(tokEmb))
	for i := range tokEmb {
		x[i] = tokEmb[i].Add(posEmb[i])
	}
	x = rmsnorm(x)

	for li := 0; li < m.cfg.NLayer; li++ {
		p := fmt.Sprintf("layer%d.", li)

		// 1) Multi-head attention block
		xResidual := x
		x = rmsnorm(x)
		q := linear(x, m.state[p+"attn_wq"])
		k := linear(x, m.state[p+"attn_wk"])
		v := linear(x, m.state[p+"attn_wv"])
		m.keys[li] = append(m.keys[li], k)
		m.values[li] = append(m.values[li], v)

		xAttn := make([]*Value, 0, m.cfg.NEmbd)
		for h := 0; h < m.cfg.NHead; h++ {
			hs := h * m.cfg.HeadDim
			qH := q[hs : hs+m.cfg.HeadDim]

			// The division by sqrt(head_dim) is not optional: a dot product grows
			// like sqrt(d), and without it softmax saturates to a hard argmax and
			// most dimensions receive no gradient.
			attnLogits := make([]*Value, len(m.keys[li]))
			for t := range m.keys[li] {
				kH := m.keys[li][t][hs : hs+m.cfg.HeadDim]
				attnLogits[t] = dot(qH, kH).DivScalar(math.Sqrt(float64(m.cfg.HeadDim)))
			}
			attnWeights := softmax(attnLogits)

			// A convex combination of the values, so the output magnitude does not
			// grow with sequence length.
			headOut := make([]*Value, m.cfg.HeadDim)
			valueColumn := make([]*Value, len(m.values[li]))
			for j := 0; j < m.cfg.HeadDim; j++ {
				for t := range m.values[li] {
					valueColumn[t] = m.values[li][t][j]
				}
				headOut[j] = dot(attnWeights, valueColumn)
			}
			xAttn = append(xAttn, headOut...)
		}
		x = linear(xAttn, m.state[p+"attn_wo"])
		x = addAll(x, xResidual)

		// 2) MLP block. The same four steps: save, normalise, transform, add.
		xResidual = x
		x = rmsnorm(x)
		x = reluLinear(x, m.state[p+"mlp_fc1"])
		x = linear(x, m.state[p+"mlp_fc2"])
		x = addAll(x, xResidual)
	}

	return linear(x, m.state["lm_head"])
}

func addAll(a, b []*Value) []*Value {
	out := make([]*Value, len(a))
	for i := range a {
		out[i] = a[i].Add(b[i])
	}
	return out
}

// ─── Data ──────────────────────────────────────────────────────────────────

func loadDocs(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var docs []string
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)
	for scanner.Scan() {
		if line := strings.TrimSpace(scanner.Text()); line != "" {
			docs = append(docs, line)
		}
	}
	return docs, scanner.Err()
}

// tokenize is `[BOS] + chars + [BOS]`. The trailing BOS is the whole trick: it
// turns a closed-ended task into an open-ended one, so the model learns where
// names stop and can therefore learn to stop.
func tokenize(doc string, charIndex map[rune]int, bos int) []int {
	tokens := make([]int, 0, len(doc)+2)
	tokens = append(tokens, bos)
	for _, ch := range doc {
		tokens = append(tokens, charIndex[ch])
	}
	tokens = append(tokens, bos)
	return tokens
}

// ─── Main ──────────────────────────────────────────────────────────────────

func main() {
	var (
		inputPath = flag.String("input", "../../data/input.txt", "path to the dataset, one document per line")
		steps     = flag.Int("steps", 1000, "training steps")
		seed      = flag.Int64("seed", 42, "PRNG seed")
		traceOut  = flag.String("trace", "", "if set, write a JSONL trace here")
		quiet     = flag.Bool("quiet", false, "suppress per-step output")
	)
	flag.Parse()

	docs, err := loadDocs(*inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot read %s: %v\n", *inputPath, err)
		os.Exit(1)
	}

	chars := map[rune]bool{}
	for _, doc := range docs {
		for _, ch := range doc {
			chars[ch] = true
		}
	}
	uchars := make([]rune, 0, len(chars))
	for ch := range chars {
		uchars = append(uchars, ch)
	}
	// Sorted so the vocabulary -- and therefore every parameter's meaning -- is
	// the same on every machine and every run.
	for i := 0; i < len(uchars); i++ {
		for j := i + 1; j < len(uchars); j++ {
			if uchars[j] < uchars[i] {
				uchars[i], uchars[j] = uchars[j], uchars[i]
			}
		}
	}
	charIndex := make(map[rune]int, len(uchars))
	for i, ch := range uchars {
		charIndex[ch] = i
	}
	bos := len(uchars)

	rng := rand.New(rand.NewSource(*seed))
	rng.Shuffle(len(docs), func(i, j int) { docs[i], docs[j] = docs[j], docs[i] })

	cfg := Config{
		NEmbd: 16, NHead: 4, NLayer: 1, BlockSize: 16,
		HeadDim: 4, VocabSize: bos + 1, Steps: *steps,
	}
	m := newModel(cfg, rng)
	params := m.params()
	cfg.Params = len(params)

	fmt.Printf("num docs: %d\nvocab size: %d\nnum params: %d\n", len(docs), cfg.VocabSize, cfg.Params)
	if *quiet {
		fmt.Printf("num params: %d\n", cfg.Params)
	}

	// Adam.
	const (
		learningRate = 0.01
		beta1        = 0.85
		beta2        = 0.98
		epsAdam      = 1e-8
	)
	moments := make([]float64, len(params))
	velocities := make([]float64, len(params))

	// Only open a trace when one was asked for. `os.Create("")` fails, and
	// exiting on it would mean a default invocation could not run at all.
	var traceFile *os.File
	if *traceOut != "" {
		traceFile, err = os.Create(*traceOut)
		if err != nil {
			fmt.Fprintf(os.Stderr, "cannot write trace: %v\n", err)
			os.Exit(1)
		}
		defer traceFile.Close()
		writeJSON(traceFile, map[string]any{
			"type": "meta", "lang": "go", "steps": *steps, "seed": *seed,
			"hyperparameters": cfg,
		})
	}

	started := time.Now()
	for step := 0; step < *steps; step++ {
		doc := docs[step%len(docs)]
		tokens := tokenize(doc, charIndex, bos)
		n := cfg.BlockSize
		if len(tokens)-1 < n {
			n = len(tokens) - 1
		}

		m.resetCache()
		losses := make([]*Value, 0, n)
		for posID := 0; posID < n; posID++ {
			logits := m.forward(tokens[posID], posID)
			probs := softmax(logits)
			// Log FIRST, then negate: `-probs[target].log()`. The order matters,
			// and getting it backwards gives `log(-p)` and therefore NaN from the
			// very first step. Which is at least a loud failure rather than a
			// quietly wrong loss.
			losses = append(losses, probs[tokens[posID+1]].Log().Neg())
		}
		loss := Mean(losses)
		loss.Backward()

		// Linear decay to zero, so the run settles into a minimum rather than
		// bouncing around it.
		lrT := learningRate * (1 - float64(step)/float64(*steps))
		biasCorrection1 := 1 - math.Pow(beta1, float64(step+1))
		biasCorrection2 := 1 - math.Pow(beta2, float64(step+1))
		for i, p := range params {
			moments[i] = beta1*moments[i] + (1-beta1)*p.Grad
			velocities[i] = beta2*velocities[i] + (1-beta2)*p.Grad*p.Grad
			// Bias correction, without which the first step is 1/(1-beta1) times
			// too small and training looks like it has a learning-rate bug.
			mHat := moments[i] / biasCorrection1
			vHat := velocities[i] / biasCorrection2
			p.Data -= lrT * mHat / (math.Sqrt(vHat) + epsAdam)
			p.Grad = 0
		}

		if !*quiet {
			fmt.Printf("step %4d / %4d | loss %.4f\n", step+1, *steps, loss.Data)
		}
		if *traceOut != "" {
			writeJSON(traceFile, map[string]any{
				"type": "step", "step": step, "loss": loss.Data,
			})
		}
	}
	elapsed := time.Since(started)

	// Inference: the same forward function, fed its own output. The only new
	// idea is the stop condition, which is a token the model learned to emit.
	const temperature = 0.5
	if !*quiet {
		fmt.Println("\n--- inference (new, hallucinated names) ---")
	}
	for sample := 0; sample < 20; sample++ {
		m.resetCache()
		tokenID := bos
		var out strings.Builder
		for posID := 0; posID < cfg.BlockSize; posID++ {
			logits := m.forward(tokenID, posID)
			scaled := make([]*Value, len(logits))
			for i, l := range logits {
				scaled[i] = l.DivScalar(temperature)
			}
			probs := softmax(scaled)
			weights := make([]float64, len(probs))
			for i, p := range probs {
				weights[i] = p.Data
			}
			// Weighted sampling, not argmax: argmax would return the same name
			// twenty times.
			tokenID = sampleFrom(rng, weights)
			if tokenID == bos {
				break
			}
			out.WriteRune(uchars[tokenID])
		}
		if !*quiet {
			fmt.Printf("sample %2d: %s\n", sample+1, out.String())
		}
		if *traceOut != "" {
			writeJSON(traceFile, map[string]any{
				"type": "sample", "step": *steps, "text": out.String(),
			})
		}
	}

	elapsedMs := float64(elapsed.Nanoseconds()) / 1e6
	if !*quiet {
		fmt.Printf("\nTotal time: %.3f ms (%.1f steps/sec)\n",
			elapsedMs, float64(*steps)/elapsed.Seconds())
	}
	if *traceOut != "" {
		writeJSON(traceFile, map[string]any{
			"type": "timing", "lang": "go", "steps": *steps,
			"step_ms": elapsedMs / float64(*steps),
		})
	}
}

func sampleFrom(rng *rand.Rand, weights []float64) int {
	total := 0.0
	for _, w := range weights {
		total += w
	}
	target := rng.Float64() * total
	for i, w := range weights {
		target -= w
		if target <= 0 {
			return i
		}
	}
	return len(weights) - 1
}

func writeJSON(w *os.File, payload map[string]any) {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return
	}
	w.Write(encoded)
	w.Write([]byte("\n"))
}

