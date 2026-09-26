import js from '@eslint/js'
import reactHooks from 'eslint-plugin-react-hooks'
import tseslint from 'typescript-eslint'
import globals from 'globals'

/**
 * Flat ESLint config.
 *
 * A deliberately small rule set, with the two defaults off that would otherwise
 * fight the code:
 *
 *  - `@typescript-eslint/no-unused-vars` ignores names starting with `_`, because
 *    a destructured `const { min: _min } = ...` is a legitimate way to say "I know
 *    this is unused and I mean it".
 *  - `no-console` is off. This is a documentation site whose *point* is a
 *    playground that logs, and a linter that objects to `console.log` in a repo
 *    whose subject is a script that prints a loss curve every step is not
 *    earning its keep.
 */
export default tseslint.config(
  { ignores: ['dist', 'node_modules', 'playwright-report', 'test-results', 'src/data/generated'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      globals: { ...globals.browser, ...globals.node },
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    plugins: { 'react-hooks': reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_', caughtErrors: 'none' },
      ],
      'no-console': 'off',
      eqeqeq: ['error', 'always', { null: 'ignore' }],
      'prefer-const': 'error',
      'no-var': 'error',
    },
  },
  {
    // The e2e suite and the static server are Node programs, not browser code.
    // `console` is off here for the same reason it is off everywhere: the static
    // server logs its mount point on startup and a test harness that cannot say
    // anything is not a harness.
    files: ['e2e/**/*.{ts,mjs}'],
    languageOptions: { globals: { ...globals.node } },
    rules: { 'no-console': 'off' },
  },
)
