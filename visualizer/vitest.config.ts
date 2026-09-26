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
    include: ['tests/**/*.test.ts', 'tests/**/*.test.tsx'],
    reporters: 'default',
  },
})
