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
