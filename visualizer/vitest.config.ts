import { defineConfig } from 'vitest/config'

/**
 * Vitest config, kept separate from `vite.config.ts`.
 *
 * The `test` key is not part of Vite's `UserConfig` type in v8, so putting it
 * there is a type error rather than a stylistic choice. It used to live in
 * `vite.config.ts` and was commented as "Vite reads this", which was true
 * historically and is not now.
 */
export default defineConfig({
  test: {
    // Node, not jsdom: the unit tests cover pure functions -- segmenting maths,
    // the loss smoothing, the selector data that reached the browser. Anything
    // needing a DOM is a Playwright test, which runs against a real build.
    environment: 'node',
    // Both the `tests/` directory and tests colocated with the source.
    //
    // The glob used to be `tests/**` only, which meant a test written beside the
    // module it covers (`src/app/router.test.ts`) was *silently* not run: it was
    // a green suite that had never executed the file. Colocated tests are the
    // pattern here, so the glob has to follow them rather than the other way
    // round. Keep the two families in sync -- a pattern present in one list and
    // not the other reintroduces exactly this bug for that family.
    include: ['tests/**/*.test.ts', 'tests/**/*.test.tsx', 'src/**/*.test.ts', 'src/**/*.test.tsx'],
    reporters: 'default',
  },
})
