/**
 * CLI wrapper around `train()`. The model lives in `index.ts` with no I/O so the
 * browser playground can import it directly -- which is the reason this port
 * exists in this repository: it is the same algorithm as the reference, in the
 * language the reader is already reading this page in.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { loadDocs, train } from './index.ts'

function arg(name: string, fallback: string): string {
  const index = process.argv.indexOf(name)
  return index >= 0 && index + 1 < process.argv.length ? process.argv[index + 1]! : fallback
}

const input = arg('--input', '../../data/input.txt')
const steps = Number(arg('--steps', '1000'))
const seed = Number(arg('--seed', '42'))
const trace = arg('--trace', '')

const docs = loadDocs(readFileSync(input, 'utf-8'))
const started = performance.now()

const result = train(docs, {
  config: { num_steps: steps },
  seed,
  quiet: false,
})

const elapsed = performance.now() - started
console.log('\n--- inference (new, hallucinated names) ---')
result.samples.forEach((text, i) => console.log(`sample ${String(i + 1).padStart(2)}: ${text}`))
console.log(
  `\nTotal time: ${elapsed.toFixed(1)} ms (${(steps / (elapsed / 1000)).toFixed(1)} steps/sec)`,
)

if (trace) {
  const lines = [
    JSON.stringify({ type: 'meta', lang: 'typescript', steps, seed }),
    ...result.losses.map((loss, step) => JSON.stringify({ type: 'step', step, loss })),
    ...result.samples.map((text, index) => JSON.stringify({ type: 'sample', step: steps, text, index })),
    JSON.stringify({ type: 'timing', lang: 'typescript', steps, step_ms: elapsed / steps }),
  ]
  writeFileSync(trace, lines.join('\n') + '\n')
  console.log(`wrote ${trace}`)
}

// The held-out loss, as the last thing the process says.
//
// Last is the whole point, and it is a property of this file rather than of the
// harness: the other three tracks print it last, and the harness's regex is
// anchored per line and searched across the whole output, so a line that sits
// earlier still parses. That is precisely what made the earlier placement worth
// changing -- nothing fails when a line moves, so nothing would have caught it.
//
// Not gated on `quiet`, unlike the per-step loss and the samples. `quiet` exists to
// shorten a watchable run; the harness passes `--input --steps --seed --val-stride`
// and nothing else, so quiet is off in every measured run -- and a flag that could
// silence the one number the keep gate reads is a flag that could make a run look
// like a track that reports no held-out loss at all.
console.log(`val_loss ${result.valLoss.toFixed(6)}`)
