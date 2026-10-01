/**
 * A TypeScript port of Andrej Karpathy's 199-line microgpt.py.
 *
 * ## Why this file looks like the Python one
 *
 * The most interesting thing about microgpt.py is that the *model* is 199 lines
 * of unremarkable code: a linear layer is a loop over a matrix, softmax is four
 * lines, attention is a loop over heads. All the subtlety lives in autograd,
 * which is 40 lines of class.
 *
 * So this port keeps that structure exactly. `linear` here is a loop over an
 * array of arrays, not a call into a BLAS. The point is that a reader can put
 * the two files side by side and see nothing but TypeScript.
 *
 * TypeScript *does* have operator overloading in the sense that matters here —
 * `+` and `*` work on objects if the methods exist — so the `Value` class below
 * is almost a transliteration of the reference's, with the dunders spelled
 * `add`/`mul` because JavaScript has no `__add__` and arithmetic operators on
 * objects require `valueOf`, which would silently reintroduce a *numeric*
 * coercion everywhere and make the graph impossible to reason about. Naming the
 * operations is the honest choice: `a.add(b)` cannot be mistaken for a number.
 *
 * ## The one number that will surprise you
 *
 * Everything is float64, like the Python reference, not float32 like the C and
 * Rust ports. That is deliberate: this is the *reference-faithful* port, and the
 * value of putting the same algorithm in the language the UI is written in is
 * that a reader can diff the arithmetic line by line against microgpt.py. The
 * speed comparison lives in the C, Rust and Go tracks, where it belongs.
 */

/** A scalar in the computation graph. Karpathy's `class Value`, field for field. */
export class Value {
  /** The number. The only thing the forward pass cares about. */
  data: number
  /** The derivative of the final loss with respect to `data`, 0 until `backward()`. */
  grad = 0
  /** The Values that produced this one. */
  children: Value[] = []
  /** How much each of those children contributed, positionally aligned. */
  localGrads: number[] = []

  constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
    if (children.length !== localGrads.length) {
      // Never drop a zero from localGrads to "keep the array small": the two
      // arrays are positional, and a ReLU below zero has a local gradient of
      // exactly 0, which is the most common value there is.
      throw new Error(
        `Value: ${children.length} children but ${localGrads.length} local gradients`,
      )
    }
    this.data = data
    this.children = children
    this.localGrads = localGrads
  }

  /** `__add__`. The local gradients are 1 and 1: d(a+b)/da = d(a+b)/db = 1. */
  add(other: Value): Value {
    return new Value(this.data + other.data, [this, other], [1, 1])
  }

  /** `self + n` for a plain number. */
  addScalar(n: number): Value {
    return this.add(new Value(n))
  }

  /**
   * `__mul__`. The local gradients are *the other operand's value*, which is the
   * product rule: d(ab)/da = b. Nothing memorises a rule table.
   */
  mul(other: Value): Value {
    return new Value(this.data * other.data, [this, other], [other.data, this.data])
  }

  /** `self * n` for a plain number. */
  mulScalar(n: number): Value {
    return this.mul(new Value(n))
  }

  /** `__neg__`, defined as `self * -1` so the graph never grows a special case. */
  neg(): Value {
    return this.mulScalar(-1)
  }

  /** `__sub__`, i.e. `self + (-other)`. */
  sub(other: Value): Value {
    return this.add(other.neg())
  }

  /** `self - n` for a plain number. */
  subScalar(n: number): Value {
    return this.addScalar(-n)
  }

  /** `__truediv__`, i.e. `self * other ** -1`. */
  div(other: Value): Value {
    return this.mul(other.pow(-1))
  }

  /** `self / n` for a plain number, as a multiplication so the graph records only the numerator's gradient. */
  divScalar(n: number): Value {
    return this.mulScalar(1 / n)
  }

  /** `__pow__` for a constant exponent. */
  pow(exponent: number): Value {
    return new Value(Math.pow(this.data, exponent), [this], [
      exponent * Math.pow(this.data, exponent - 1),
    ])
  }

  /** `log()`. */
  log(): Value {
    return new Value(Math.log(this.data), [this], [1 / this.data])
  }

  /** `exp()`. */
  exp(): Value {
    const e = Math.exp(this.data)
    return new Value(e, [this], [e])
  }

  /**
   * `relu()`. The derivative is 1 above zero and 0 below, and ambiguous exactly
   * at zero, where this picks 0 -- the same choice the reference makes.
   */
  relu(): Value {
    return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0])
  }

  /**
   * The whole backward pass, in one method.
   *
   * Phase 1 sorts the graph parents-first with a depth-first walk, so reversing
   * it gives children-first -- the only order in which the chain rule is
   * computable. `Set` gives identity hashing, which is correct here and only
   * correct because `Value` defines neither `equals` nor `valueOf`.
   *
   * Phase 2 walks it in reverse, accumulating
   * `child.grad += localGrad * v.grad` -- the chain rule, as a loop. The `+=` is
   * load-bearing: a Value is reused by every document, every position and every
   * head, so its gradient has many contributions to sum.
   */
  backward(): void {
    const topo: Value[] = []
    const visited = new Set<Value>()

    const buildTopo = (v: Value): void => {
      if (visited.has(v)) return
      visited.add(v)
      for (const child of v.children) buildTopo(child)
      topo.push(v)
    }
    buildTopo(this)

    this.grad = 1
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i]!
      for (let j = 0; j < v.children.length; j++) {
        v.children[j]!.grad += v.localGrads[j]! * v.grad
      }
    }
  }
}

/** `sum(values)` */
export function sum(values: Value[]): Value {
  let total = new Value(0)
  for (const v of values) total = total.add(v)
  return total
}

/** `(1 / n) * sum(losses)` */
export function mean(values: Value[]): Value {
  return sum(values).divScalar(values.length)
}

export type Matrix = Value[][]

export interface Config {
  n_embd: number
  n_head: number
  n_layer: number
  block_size: number
  head_dim: number
  vocab_size: number
  num_steps: number
}

export const DEFAULT_CONFIG: Config = {
  n_embd: 16,
  n_head: 4,
  n_layer: 1,
  block_size: 16,
  head_dim: 4,
  vocab_size: 27,
  num_steps: 1000,
}

/**
 * The `matrix` lambda: a Gaussian draw with std 0.08, a deliberately small
 * number that keeps the initial distribution nearly uniform.
 */
export function matrix(rng: () => number, nout: number, nin: number, std = 0.08): Matrix {
  return Array.from({ length: nout }, () => Array.from({ length: nin }, () => new Value(gauss(rng) * std)))
}

export class Model {
  readonly config: Config
  readonly state: Record<string, Matrix>
  private keys: Value[][][]
  private values: Value[][][]

  constructor(config: Config, rng: () => number) {
    this.config = config
    this.state = {}
    this.state['wte'] = matrix(rng, config.vocab_size, config.n_embd)
    this.state['wpe'] = matrix(rng, config.block_size, config.n_embd)
    this.state['lm_head'] = matrix(rng, config.vocab_size, config.n_embd)
    for (let i = 0; i < config.n_layer; i++) {
      const p = `layer${i}.`
      this.state[`${p}attn_wq`] = matrix(rng, config.n_embd, config.n_embd)
      this.state[`${p}attn_wk`] = matrix(rng, config.n_embd, config.n_embd)
      this.state[`${p}attn_wv`] = matrix(rng, config.n_embd, config.n_embd)
      this.state[`${p}attn_wo`] = matrix(rng, config.n_embd, config.n_embd)
      this.state[`${p}mlp_fc1`] = matrix(rng, 4 * config.n_embd, config.n_embd)
      this.state[`${p}mlp_fc2`] = matrix(rng, config.n_embd, 4 * config.n_embd)
    }
    this.keys = []
    this.values = []
    this.resetCache()
  }

  resetCache(): void {
    this.keys = Array.from({ length: this.config.n_layer }, () => [])
    this.values = Array.from({ length: this.config.n_layer }, () => [])
  }

  /** Every parameter, flattened. The optimizer does not care about structure. */
  params(): Value[] {
    const out: Value[] = []
    for (const mat of Object.values(this.state)) for (const row of mat) for (const p of row) out.push(p)
    return out
  }

  /** `linear(x, w)` -- the only place a weight matrix is used, and most of the arithmetic in the model. */
  static linear(x: Value[], w: Matrix): Value[] {
    return w.map((row) => {
      const children = new Array<Value>(row.length * 2)
      const localGrads = new Array<number>(row.length * 2)
      let data = 0
      for (let i = 0; i < row.length; i++) {
        const weight = row[i]!
        const input = x[i]!
        const childIndex = i * 2
        children[childIndex] = weight
        children[childIndex + 1] = input
        localGrads[childIndex] = input.data
        localGrads[childIndex + 1] = weight.data
        data += weight.data * input.data
      }
      return new Value(data, children, localGrads)
    })
  }

  /**
   * `softmax(logits)` -- subtract the max before exponentiating. Not a numerical
   * nicety: it is what makes the function usable, since the subtraction leaves
   * the result exactly unchanged and bounds every exponential to (0, 1].
   */
  static softmax(logits: Value[]): Value[] {
    let maxVal = -Infinity
    for (const v of logits) if (v.data > maxVal) maxVal = v.data
    const exps = logits.map((v) => v.subScalar(maxVal).exp())
    const total = sum(exps)
    return exps.map((e) => e.div(total))
  }

  /** `rmsnorm(x)` -- rescale by the reciprocal of the root-mean-square. */
  static rmsnorm(x: Value[]): Value[] {
    let ms = 0
    for (const v of x) ms += v.data * v.data
    ms /= x.length
    const scale = Math.pow(ms + 1e-5, -0.5)
    return x.map((v) => v.mulScalar(scale))
  }

  /**
   * `gpt(token_id, pos_id, keys, values)` -- a stateless function from one token
   * id at one position to logits over the vocabulary, given the keys and values
   * of earlier positions. That four-argument signature is the whole design: it
   * *is* the KV cache, spelled the long way.
   */
  forward(tokenId: number, posId: number): Value[] {
    const tokEmb = this.state['wte']![tokenId]!
    const posEmb = this.state['wpe']![posId]!
    let x = tokEmb.map((t, i) => t.add(posEmb[i]!))
    x = Model.rmsnorm(x)

    for (let li = 0; li < this.config.n_layer; li++) {
      const p = `layer${li}.`

      // 1) Multi-head attention block
      const xResidual = x
      x = Model.rmsnorm(x)
      const q = Model.linear(x, this.state[`${p}attn_wq`]!)
      const k = Model.linear(x, this.state[`${p}attn_wk`]!)
      const v = Model.linear(x, this.state[`${p}attn_wv`]!)
      this.keys[li]!.push(k)
      this.values[li]!.push(v)

      const xAttn: Value[] = []
      for (let h = 0; h < this.config.n_head; h++) {
        const hs = h * this.config.head_dim
        const qH = q.slice(hs, hs + this.config.head_dim)
        const kH = this.keys[li]!.map((ki) => ki.slice(hs, hs + this.config.head_dim))
        const vH = this.values[li]!.map((vi) => vi.slice(hs, hs + this.config.head_dim))

        // The division by sqrt(head_dim) is not optional: a dot product grows
        // like sqrt(d), and without it softmax saturates to a hard argmax and most
        // dimensions receive no gradient.
        const attnLogits = kH.map((kt) => {
          let acc = new Value(0)
          for (let j = 0; j < this.config.head_dim; j++) acc = acc.add(qH[j]!.mul(kt[j]!))
          return acc.divScalar(Math.sqrt(this.config.head_dim))
        })
        const attnWeights = Model.softmax(attnLogits)

        // A convex combination of the values, so the output magnitude does not
        // grow with sequence length.
        const headOut: Value[] = []
        for (let j = 0; j < this.config.head_dim; j++) {
          let acc = new Value(0)
          for (let t = 0; t < vH.length; t++) acc = acc.add(attnWeights[t]!.mul(vH[t]![j]!))
          headOut.push(acc)
        }
        xAttn.push(...headOut)
      }
      x = Model.linear(xAttn, this.state[`${p}attn_wo`]!)
      x = x.map((a, i) => a.add(xResidual[i]!))

      // 2) MLP block. The same four steps: save, normalise, transform, add.
      const mlpResidual = x
      x = Model.rmsnorm(x)
      x = Model.linear(x, this.state[`${p}mlp_fc1`]!)
      x = x.map((v) => v.relu())
      x = Model.linear(x, this.state[`${p}mlp_fc2`]!)
      x = x.map((a, i) => a.add(mlpResidual[i]!))
    }

    return Model.linear(x, this.state['lm_head']!)
  }
}

// ─── PRNG ─────────────────────────────────────────────────────────────────

/**
 * A seeded PRNG, so a run is reproducible.
 *
 * The reference uses Python's Mersenne Twister via `random.gauss`, the C and Rust
 * ports use xoshiro256++. This is xoshiro256++ as well, so this port and the C
 * port differ only in the language. That is the point of the parity gate being
 * statistical rather than exact: with a different PRNG and a different float
 * width, bit-identical loss curves are not available, and pretending otherwise
 * would be a claim nobody could check.
 */
export function makeRng(seed: number): () => number {
  // SplitMix64 seeding, as the C port does it.
  let z = BigInt(seed) & 0xffffffffffffffffn
  const next = (): bigint => {
    z = (z + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn
    let x = z
    x = ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n) & 0xffffffffffffffffn
    x = ((x ^ (x >> 27n)) * 0x94d049bb133111ebn) & 0xffffffffffffffffn
    return (x ^ (x >> 31n)) & 0xffffffffffffffffn
  }
  const s = [next(), next(), next(), next()]
  const MASK = 0xffffffffffffffffn
  const rotl = (x: bigint, k: bigint): bigint => ((x << k) | (x >> (64n - k))) & MASK

  const u64 = (): bigint => {
    const result = (s[0]! + s[3]!) & MASK
    const t = (s[1]! << 17n) & MASK
    s[2] = (s[2]! ^ s[0]!) & MASK
    s[3] = (s[3]! ^ s[1]!) & MASK
    s[1] = (s[1]! ^ s[2]!) & MASK
    s[0] = (s[0]! ^ s[3]!) & MASK
    s[2] = (s[2]! ^ t) & MASK
    s[3] = rotl(s[3]!, 45n)
    return rotl(result, 23n)
  }

  return () => Number(u64() >> 11n) / 9007199254740992
}

/** Box-Muller, so the normal draws are comparable with the C and Rust ports. */
export function gauss(rng: () => number): number {
  const u1 = Math.max(rng(), Number.MIN_VALUE)
  const u2 = rng()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

/** Fisher-Yates, the same shuffle the reference performs. */
export function shuffle<T>(rng: () => number, items: T[]): T[] {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.min(i, Math.floor(rng() * (i + 1)))
    const tmp = items[i]!
    items[i] = items[j]!
    items[j] = tmp
  }
  return items
}

// ─── Data ─────────────────────────────────────────────────────────────────

/** `input.txt` as a list of documents, one per line. */
export function loadDocs(text: string): string[] {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
}

/**
 * `[BOS] + chars + [BOS]`. The trailing BOS is the whole trick: it turns a
 * closed-ended task into an open-ended one, so the model learns where names stop
 * and can therefore learn to stop.
 */
export function tokenize(doc: string, charIndex: Map<string, number>, bos: number): number[] {
  const tokens = [bos]
  for (const ch of doc) tokens.push(charIndex.get(ch) ?? 0)
  tokens.push(bos)
  return tokens
}

/** The sorted set of every character in the dataset, so it is the same everywhere. */
export function buildVocab(docs: string[]): { uchars: string[]; charIndex: Map<string, number> } {
  const all = new Set<string>()
  for (const doc of docs) for (const ch of doc) all.add(ch)
  const uchars = [...all].sort()
  return { uchars, charIndex: new Map(uchars.map((c, i) => [c, i])) }
}

/** Weighted sampling, not argmax -- argmax would return the same name 20 times. */
export function sampleFrom(rng: () => number, weights: number[]): number {
  const total = weights.reduce((a, b) => a + b, 0)
  let target = rng() * total
  for (let i = 0; i < weights.length; i++) {
    target -= weights[i]!
    if (target <= 0) return i
  }
  return weights.length - 1
}

// ─── Training loop ────────────────────────────────────────────────────────

/**
 * One in every `DEFAULT_VAL_STRIDE` documents of the corpus is held out of
 * training and used only for the `val_loss` measurement at the end of the run.
 *
 * This is a default rather than a requirement because the loss axis reads the
 * *training* loss, and a training loss is a measurement of the run rather than
 * of the model -- while which documents the run trains on is the candidate's to
 * choose. The C track worked that out inside twenty experiments. Run 0307 --
 * "Pretrain the 50 candidates twice then replay the easiest for 900 steps" --
 * pointed the measured window at documents it had already learned and then spent
 * 900 of its 1,000 steps on the single easiest one. Its loss read 0.000000,
 * which the ledger recorded as a 100% gain, and it took the speed axis at the
 * same time: a shorter document is fewer tokens per step, so the same
 * non-result ran 261% faster. A model that had learned nothing, reported as the
 * track's best run so far.
 *
 * So a run reports two numbers now. The training loss says what the run did,
 * which is a thing the loop is allowed to change. The held-out loss says what
 * the model does with documents the run never reached, which is a thing the run
 * is not allowed to change, because those documents are not in the training pool
 * at all. The harness reads the second one and refuses a keep whose training
 * loss moved while its held-out loss did not follow, whichever axis the
 * candidate arrived on; 0307's signature is exactly that gap.
 *
 * 128 holds out 251 of the 32,033 documents in `data/input.txt` -- about 1,760
 * prediction positions -- which is enough documents that the measurement's own
 * noise stays under the 5% the harness treats as a regression.
 */
export const DEFAULT_VAL_STRIDE = 128

/**
 * `--val-stride N` read off the process argument vector, defaulting to
 * `DEFAULT_VAL_STRIDE`.
 *
 * Parsed here rather than in `cli.ts` deliberately. The flag is a property of the
 * run and the run lives in this file, and `cli.ts` is byte-identical between the
 * frozen track and the candidate -- which is the point of this port, since the
 * harness measures the candidate against the frozen track and every extra line
 * in the CLI is a line that has to be kept identical by hand for no gain. Reading
 * `process.argv` here also leaves the flag's default in one place instead of two.
 *
 * The `typeof process` guard is not decoration: this module is imported by the
 * browser playground, where there is no `process` and where the split has to fall
 * back to its default rather than throw.
 */
export function valStrideFromArgv(argv: readonly string[]): number {
  const at = argv.indexOf('--val-stride')
  if (at < 0) return DEFAULT_VAL_STRIDE
  const raw = argv[at + 1]
  const stride = raw === undefined ? Number.NaN : Number(raw)
  // A flag that silently fell back to its default would be a held-out measurement
  // that quietly stopped happening, and the harness cannot tell the difference
  // between "this track reports no held-out loss" and "this track was told to
  // report one and did not". So an unusable value is refused by name, rather than
  // becoming 128.
  if (!Number.isInteger(stride)) {
    throw new Error(
      `--val-stride wants a whole number, got ${raw === undefined ? 'nothing' : JSON.stringify(raw)}`,
    )
  }
  // A stride of 1 holds out every document and leaves the training loop an empty
  // pool to index; a stride of 0 is a remainder by zero. Both used to fail
  // further down with a message about arithmetic rather than about the flag, and
  // this number is not something a run should be able to make meaningless.
  if (stride < 2) {
    throw new Error(
      `--val-stride must be at least 2, got ${stride}: it says how many documents to skip between held-out ones`,
    )
  }
  return stride
}

export interface TrainOptions {
  config?: Partial<Config>
  seed?: number
  /**
   * How many corpus documents to skip between held-out ones. Defaults to
   * `--val-stride` off the command line, and to `DEFAULT_VAL_STRIDE` when there
   * is no command line -- a library caller with no process wants the split too,
   * and gets it.
   */
  valStride?: number
  /** Called after each step, for a trace. */
  onStep?: (step: number, loss: number) => void
  onSample?: (index: number, text: string) => void
  quiet?: boolean
}

export interface TrainResult {
  losses: number[]
  samples: string[]
  params: number
  /** Loss on the held-out documents, measured once on the final model.
   *
   *  Returned rather than printed. The other three tracks print `val_loss` as the
   *  last line of the process, and this one used to print it from inside `train()`
   *  -- which put it *before* the inference block, the timing and the trace line,
   *  because `cli.ts` prints those after `train()` returns. The harness's regex is
   *  anchored per line and searched across the whole output, so it parsed fine
   *  either way, and that is exactly why it was worth fixing: a line's position
   *  being load-bearing is a property no test catches and every future edit can
   *  quietly break. Printing it from the caller keeps the four tracks alike and
   *  keeps the last thing the process says a number the gate reads. */
  valLoss: number
}

/**
 * The training loop and the inference loop, in one function.
 *
 * Identical in structure to the reference's bottom half: one document per step,
 * framed with BOS, run one position at a time, averaged into a loss,
 * backpropagated, updated with Adam. There is no batching, no gradient
 * accumulation and no checkpointing -- every one of those is a real technique
 * that this file deliberately leaves out.
 *
 * There is a held-out split, which the reference also lacks and which is here for
 * a reason that has nothing to do with technique: the loss axis measures the
 * *training* loss, and a training loss is a measurement of the run rather than of
 * the model. See `DEFAULT_VAL_STRIDE` for how the C track turned that into a
 * 0.000000.
 */
export function train(docs: string[], options: TrainOptions = {}): TrainResult {
  const config: Config = { ...DEFAULT_CONFIG, ...options.config }
  const seed = options.seed ?? 42
  config.head_dim = Math.floor(config.n_embd / config.n_head)

  const { uchars, charIndex } = buildVocab(docs)
  const bos = uchars.length
  config.vocab_size = bos + 1

  // The harness passes `--val-stride` on the command line; a library caller with
  // no command line gets the default, and an explicit option beats both.
  const argv: readonly string[] =
    typeof process !== 'undefined' && Array.isArray(process.argv) ? process.argv : []
  const valStride = options.valStride ?? valStrideFromArgv(argv)

  const rng = makeRng(seed)

  // ─── The held-out split ───────────────────────────────────────────────────
  //
  // Every `valStride`th document *of the corpus* -- that is, of `docs` as it
  // arrived, before any shuffling -- is validation, and every other document is
  // the training pool. The index is deliberately the pre-shuffle one: a split
  // that followed the shuffle would move when the seed moved, which would make
  // the held-out set a second thing the loop can choose, and choosing the split
  // is as much a choice as choosing the training documents was. Held out by
  // position in the corpus, it is the same 251 documents for every seed, every
  // candidate and every run, so the two numbers are comparable.
  //
  // What is shuffled is the *indices*, not the documents, and that is not
  // tidiness. `shuffle` draws `length - 1` times whatever the elements are, so
  // shuffling an index vector of 32,033 consumes exactly the PRNG stream that
  // shuffling 32,033 documents did -- and the 4,192 gaussians of parameter
  // initialisation drawn from that same stream afterwards, with nothing but this
  // bookkeeping in between, are bit for bit what they were before the split
  // existed, and so is every training step. Shuffling the surviving documents
  // instead would have drawn a different 4,192 and turned the harness's
  // frozen-versus-candidate comparison into a comparison of two different models.
  const order = Array.from({ length: docs.length }, (_, i) => i)
  shuffle(rng, order)
  const trainOrder: number[] = []
  const valOrder: number[] = []
  for (const index of order) {
    if (index % valStride === 0) valOrder.push(index)
    else trainOrder.push(index)
  }
  if (trainOrder.length === 0) {
    // Unreachable for any stride `valStrideFromArgv` accepts and any corpus with
    // more than one document, which is why it is a check and not an argument:
    // `docs[step % 0]` is `undefined`, and a TypeError about `tokenize` says
    // nothing about the dataset being smaller than the stride.
    throw new Error(
      `--val-stride ${valStride} held out all ${docs.length} documents, leaving the training loop nothing to read`,
    )
  }

  const model = new Model(config, rng)
  const params = model.params()
  if (!options.quiet) {
    console.log(`num docs: ${docs.length}`)
    console.log(`vocab size: ${config.vocab_size}`)
    console.log(`num params: ${params.length}`)
    // Both counts, because "num docs" is the size of the file and the size of the
    // pool a step can draw from stopped being the same number at this change.
    console.log(`held out: ${valOrder.length} of ${docs.length} docs (--val-stride ${valStride})`)
  }

  // Adam, with the reference's betas. The bias correction is what makes the
  // first step the same size as every other one.
  const learningRate = 0.01
  const beta1 = 0.85
  const beta2 = 0.99
  const epsAdam = 1e-8
  const m = new Array(params.length).fill(0)
  const v = new Array(params.length).fill(0)

  const losses: number[] = []
  for (let step = 0; step < config.num_steps; step++) {
    // The only line of the training loop this change touches, and it is a change
    // of which document is read rather than of what a step computes. Before, the
    // pool was `docs` itself; now it is the shuffled list of *training* indices,
    // so a held-out document cannot be reached from a step at all. Everything
    // below this line -- the framing, the forward pass, the loss, Adam -- is
    // untouched, because a held-out split is a change to the corpus the run sees
    // and not a change to the arithmetic of a step.
    const doc = docs[trainOrder[step % trainOrder.length]!]!
    const tokens = tokenize(doc, charIndex, bos)
    const n = Math.min(config.block_size, tokens.length - 1)

    model.resetCache()
    const perPosition: Value[] = []
    for (let posId = 0; posId < n; posId++) {
      const logits = model.forward(tokens[posId]!, posId)
      const probs = Model.softmax(logits)
      // Log first, then negate: `-probs[target].log()`. The order matters, and
      // backwards gives `log(-p)` and therefore NaN immediately.
      perPosition.push(probs[tokens[posId + 1]!]!.log().neg())
    }
    const loss = mean(perPosition)
    loss.backward()

    // Linear decay to zero, so the run settles into a minimum rather than
    // bouncing around it.
    const lrT = learningRate * (1 - step / config.num_steps)
    for (let i = 0; i < params.length; i++) {
      const p = params[i]!
      m[i] = beta1 * m[i]! + (1 - beta1) * p.grad
      v[i] = beta2 * v[i]! + (1 - beta2) * p.grad * p.grad
      const mHat = m[i]! / (1 - Math.pow(beta1, step + 1))
      const vHat = v[i]! / (1 - Math.pow(beta2, step + 1))
      p.data -= (lrT * mHat) / (Math.pow(vHat, 0.5) + epsAdam)
      p.grad = 0
    }

    losses.push(loss.data)
    options.onStep?.(step, loss.data)
    if (!options.quiet && (step < 5 || (step + 1) % 100 === 0)) {
      console.log(`step ${String(step + 1).padStart(4)} / ${config.num_steps} | loss ${loss.data.toFixed(4)}`)
    }
  }

  // ─── The held-out measurement ─────────────────────────────────────────────
  //
  // Forward passes only: no `backward()`, no Adam, no parameter written. This
  // reads the model the training loop ended with and nothing else, which is the
  // whole reason the number means anything. The loss axis cannot say that -- it
  // measures each step *after* the optimiser step on that very document, so a
  // training loss is allowed to read below what an identically-measured
  // training document reads on a model that has not yet been stepped on it. The
  // two numbers are therefore not expected to agree, and neither one is expected
  // to be the larger.
  //
  // The per-position cross-entropy is the same expression the step above
  // computes, written out again rather than factored into a helper: a training
  // step that had to call that helper would be a training step this change had
  // edited. Summed over every prediction position of every held-out document and
  // divided by the number of positions there were, so the mean is over prediction
  // positions rather than over documents. The loss axis measures tokens, not
  // names, and a document two characters longer must not get half the vote.
  //
  // Measured before the inference block below on purpose. That block draws from
  // the PRNG and writes no parameter, so the model it would see is the same one,
  // but reading the held-out loss while the measured model is still unambiguously
  // the training loop's last model is worth the two lines of ordering.
  let valNll = 0
  let valPositions = 0
  for (const index of valOrder) {
    const doc = docs[index]!
    const tokens = tokenize(doc, charIndex, bos)
    const n = Math.min(config.block_size, tokens.length - 1)

    model.resetCache()
    for (let posId = 0; posId < n; posId++) {
      const logits = model.forward(tokens[posId]!, posId)
      const probs = Model.softmax(logits)
      // Log first, then negate: `-probs[target].log()`. The order matters, and
      // backwards gives `log(-p)` and therefore NaN immediately.
      valNll += probs[tokens[posId + 1]!]!.log().neg().data
      valPositions++
    }
    // One document's tape at a time, and that is all this needs: `Value` holds its
    // children and never its parents, so a graph becomes unreachable the moment
    // the loop's last reference to it goes out of scope. The Rust port has to
    // truncate an arena to get the same effect; here the garbage collector does
    // it, and 251 documents' worth of held-out tape is never alive at once.
  }
  if (valPositions === 0) {
    // Every held-out document was empty, which `loadDocs` makes impossible. A
    // `NaN` here would print as `val_loss NaN`, which the harness's regex does not
    // match at all -- so this would look like a track that reports no held-out
    // loss rather than like a bug.
    throw new Error('the held-out set has no prediction positions, so its loss is undefined')
  }
  const valLoss = valNll / valPositions

  // One line, once, on stdout, printed by the caller from `TrainResult.valLoss`.
  // The harness matches it with
  // `^val_loss\s+([0-9.]+)\s*$` against the run's output. Not gated on `quiet`,
  // because a `val_loss` a caller can silence by forgetting an option is one that
  // eventually goes missing from a run that was supposed to report it, and the
  // harness cannot tell that apart from a track with no held-out split. Not
  // repeated per step either: a curve of held-out losses is a third curve to
  // watch and therefore a third curve to fit, and the C run at 0307 would have
  // found it in a single experiment. The gate needs to know what the finished
  // model does with documents it has never seen, and that is one number.
  //
  // And it goes in neither the trace nor the loss curve: the trace is the
  // per-step record the loss axis averages, and this is not a step of it.
  // Printed by the caller, not here. See `TrainResult.valLoss`.

  // Inference: the same forward function, fed its own output. The only new idea
  // is the stop condition, which is a token the model was trained to emit.
  const temperature = 0.5
  const samples: string[] = []
  for (let sample = 0; sample < 20; sample++) {
    model.resetCache()
    let tokenId = bos
    let out = ''
    for (let posId = 0; posId < config.block_size; posId++) {
      const logits = model.forward(tokenId, posId)
      // Temperature goes outside softmax, on the logits -- softmax itself has no
      // notion of it.
      const scaled = logits.map((l) => l.divScalar(temperature))
      const probs = Model.softmax(scaled)
      tokenId = sampleFrom(rng, probs.map((p) => p.data))
      if (tokenId === bos) break
      out += uchars[tokenId]
    }
    samples.push(out)
    options.onSample?.(sample, out)
  }

  return { losses, samples, params: params.length, valLoss: valLoss }
}
