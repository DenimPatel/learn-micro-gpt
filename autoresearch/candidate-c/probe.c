/*
 * probe.c — the C track's finite-difference gate, for tools/autoresearch.py.
 *
 * Same measurement as the other three probes and the same output contract, so
 * the harness parses it with one regex:
 *
 *   GRADCHECK analytic=<f> params=<n> ratios=h=<h> ratio=<r> | h=<h> ratio=<r>
 *
 * ## What it is checking, and why it is the only thing that would
 *
 * `backward_all` is a hand-written chain rule. docs/KNOWN-ISSUES.md issue 1 is
 * the argument for checking it rather than reading it: that pass was wrong by a
 * factor of -0.12 -- the wrong sign, an order of magnitude out -- and the loss
 * curve stayed inside the parity band the whole time, because Adam divides by an
 * estimate of the gradient's own magnitude, so a gradient that is off by a
 * constant factor barely moves the step. A loss curve is evidence that something
 * learned, not evidence that the thing that learned was the gradient.
 *
 * The issue is fixed and the port now measures 1.03, but the tripwire is the
 * point: the loop rewrites this file without supervision, and the only thing
 * standing between a model and a new broken gradient that still trains is a
 * check like this one. Its digest is asserted by the harness against the
 * committed copy, precisely so a candidate cannot ship a weakened probe next to
 * a weakened gradient and have the two agree with each other.
 *
 * ## Why the derivative is over the whole parameter vector
 *
 * Per-parameter central differences are mostly not viable here, and
 * docs/KNOWN-ISSUES.md issue 2 has the arithmetic. In float32 the loss is ~3.2
 * with an absolute resolution near 4e-7, so a difference quotient cannot resolve
 * a gradient below roughly 2e-4 at h = 1e-3 -- and most attention and MLP
 * gradients at initialisation are 1e-5 to 1e-2. Per-parameter checks would be
 * comparing noise with noise.
 *
 * A directional derivative over every parameter at once sidesteps it. Both sides
 * are O(1), so the comparison keeps full float32 precision, and a gradient that
 * is wrong in a *correlated* way -- a whole block missing, or double-counted --
 * shows up at full size instead of being buried in each parameter's own noise.
 * That is the failure this gate exists for.
 *
 * ## The band is 0.5 to 2.0 and that is deliberate
 *
 * It is a tripwire against a collapsed or inverted gradient, not a precision
 * measurement, and narrowing it to each port's own ratio would break the reason
 * it exists: it would also stop catching the day a port's *baseline* moved, which
 * is the one failure mode a tripwire must not have. A discarded gradient
 * measures 0 and a sign slip measures negative, and both are more than a factor
 * of four from the band. A uniform scale error of ~2x does slip through, on
 * every port including this one, and that is the price; issue 5 names the real
 * fix.
 *
 * The step sizes are checked for agreement, but that check is not expected to
 * validate the constant either: a uniform scale error is h-independent by
 * construction. It is there to separate a fixed offset from a
 * difference-operator artifact, which it does well.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* The candidate's own `main` is not this program's entry point, but it is still
 * compiled, because `#include` is how this probe reaches the static weight
 * arrays and the backward pass without the candidate having to grow an API
 * purely for testing. `implementations/c/test_gradients.c` does the same thing
 * for the same reason. */
#define main microgpt_candidate_main_unused
#include "microgpt.c"
#undef main

/* wte, wpe, lm_head, then per layer wq/wk/wv/wo/fc1/fc2. Must agree with
 * `total_params()` below or the walk runs off the end of the arrays -- which is
 * what happened once, when wpe was counted at vocab_size*N_EMBD instead of
 * BLOCK_SIZE*N_EMBD. */
static float *param_at(int i) {
  if (i < vocab_size * N_EMBD) return wte + i;
  i -= vocab_size * N_EMBD;
  if (i < BLOCK_SIZE * N_EMBD) return wpe + i;
  i -= BLOCK_SIZE * N_EMBD;
  if (i < vocab_size * N_EMBD) return lm_head + i;
  i -= vocab_size * N_EMBD;
  for (int l = 0; l < N_LAYER; l++) {
    if (i < N_EMBD * N_EMBD) return attn_wq[l] + i;
    i -= N_EMBD * N_EMBD;
    if (i < N_EMBD * N_EMBD) return attn_wk[l] + i;
    i -= N_EMBD * N_EMBD;
    if (i < N_EMBD * N_EMBD) return attn_wv[l] + i;
    i -= N_EMBD * N_EMBD;
    if (i < N_EMBD * N_EMBD) return attn_wo[l] + i;
    i -= N_EMBD * N_EMBD;
    if (i < MLP_DIM * N_EMBD) return mlp_fc1[l] + i;
    i -= MLP_DIM * N_EMBD;
    if (i < N_EMBD * MLP_DIM) return mlp_fc2[l] + i;
    i -= N_EMBD * MLP_DIM;
  }
  return NULL;
}

static float grad_at(int i) {
  if (i < vocab_size * N_EMBD) return g_wte[i];
  i -= vocab_size * N_EMBD;
  if (i < BLOCK_SIZE * N_EMBD) return g_wpe[i];
  i -= BLOCK_SIZE * N_EMBD;
  if (i < vocab_size * N_EMBD) return g_lm_head[i];
  i -= vocab_size * N_EMBD;
  for (int l = 0; l < N_LAYER; l++) {
    if (i < N_EMBD * N_EMBD) return g_attn_wq[l][i];
    i -= N_EMBD * N_EMBD;
    if (i < N_EMBD * N_EMBD) return g_attn_wk[l][i];
    i -= N_EMBD * N_EMBD;
    if (i < N_EMBD * N_EMBD) return g_attn_wv[l][i];
    i -= N_EMBD * N_EMBD;
    if (i < N_EMBD * N_EMBD) return g_attn_wo[l][i];
    i -= N_EMBD * N_EMBD;
    if (i < MLP_DIM * N_EMBD) return g_mlp_fc1[l][i];
    i -= MLP_DIM * N_EMBD;
    if (i < N_EMBD * MLP_DIM) return g_mlp_fc2[l][i];
    i -= N_EMBD * MLP_DIM;
  }
  return 0.0f;
}

static int total_params(void) {
  return 2 * vocab_size * N_EMBD + BLOCK_SIZE * N_EMBD +
         N_LAYER * (4 * N_EMBD * N_EMBD + 2 * MLP_DIM * N_EMBD);
}

static void zero_gradients(void) {
  memset(g_wte, 0, sizeof(g_wte));
  memset(g_wpe, 0, sizeof(g_wpe));
  memset(g_lm_head, 0, sizeof(g_lm_head));
  for (int l = 0; l < N_LAYER; l++) {
    memset(g_attn_wq[l], 0, sizeof(g_attn_wq[l]));
    memset(g_attn_wk[l], 0, sizeof(g_attn_wk[l]));
    memset(g_attn_wv[l], 0, sizeof(g_attn_wv[l]));
    memset(g_attn_wo[l], 0, sizeof(g_attn_wo[l]));
    memset(g_mlp_fc1[l], 0, sizeof(g_mlp_fc1[l]));
    memset(g_mlp_fc2[l], 0, sizeof(g_mlp_fc2[l]));
  }
}

#define MAX_PARAMS 8192
static int probe_tokens[BLOCK_SIZE + 2];
static int probe_n;

static void build_test_document(void) {
  const char *name = "mary";
  int at = 0;
  probe_tokens[at++] = BOS_TOKEN;
  for (const char *c = name; *c; c++) {
    if (at >= BLOCK_SIZE + 1) break;
    probe_tokens[at++] = char_to_idx[(int)*c];
  }
  probe_tokens[at++] = BOS_TOKEN;
  /* `n` counts predictions, so it is one less than the token count: the last
   * token is a target and never an input. The candidate's training loop uses
   * the same convention, and getting it wrong here both mis-scales the loss by
   * 1/n and reads one token past the end. */
  probe_n = at - 1;
  if (probe_n < 1 || probe_n > BLOCK_SIZE) {
    fprintf(stderr, "probe: bad test document length %d\n", probe_n);
    exit(2);
  }
}

/* Forward and loss for the probe document, and nothing else -- no Adam update.
 * Perturbing a weight and calling this is the "numeric" side of the
 * comparison, so it has to be the pure loss. */
static float loss_for_document(void) {
  for (int pos = 0; pos < probe_n; pos++)
    forward_pos(probe_tokens[pos], pos, pos + 1);
  float loss = 0.0f;
  for (int pos = 0; pos < probe_n; pos++) {
    float p = saved_probs[pos][probe_tokens[pos + 1]];
    if (p < 1e-30f) p = 1e-30f;
    loss -= logf(p);
  }
  return loss / (float)probe_n;
}

#define BAND_LO 0.5f
#define BAND_HI 2.0f
/* The two step sizes the harness and the other probes use. The harness reads
 * the coarsest one, which is the least exposed to f32 roundoff in the
 * difference quotient. */
static const float STEP_SIZES[2] = {1e-2f, 3e-3f};
/* How far apart the two ratios may sit before the measurement itself is
 * suspect. 6% is roughly where float32 cancellation in a ~3.2 loss stops being
 * a detail. */
#define STEP_STABILITY 0.06f

int main(int argc, char **argv) {
  const char *data = "input.txt";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--input") && i + 1 < argc) {
      data = argv[++i];
    } else {
      fprintf(stderr, "probe: unrecognised argument '%s'\n", argv[i]);
      return 2;
    }
  }
  if (access(data, R_OK) != 0) {
    fprintf(stderr, "probe: cannot read %s -- the harness passes --input.\n", data);
    return 2;
  }

  /* Same order the candidate's main() uses. Seeding before load_data matters:
   * an un-seeded xoshiro state produced a degenerate forward pass that looked
   * exactly like a broken backward pass, and chasing that cost an hour. */
  seed_rng(42);
  load_data(data);
  init_weights();
  build_test_document();

  const int n = total_params();
  if (n > MAX_PARAMS) {
    fprintf(stderr, "probe: %d parameters exceeds MAX_PARAMS %d\n", n, MAX_PARAMS);
    return 2;
  }
  if (param_at(n - 1) == NULL || param_at(n) != NULL) {
    fprintf(stderr, "probe: total_params() and the walk disagree\n");
    return 2;
  }

  /* The direction comes from a fixed LCG, not from the port's PRNG, so the same
   * direction is checked on every run, every machine, and every candidate. A
   * direction drawn from whatever the program happened to consume would be a
   * different check each time, and a check that changes cannot be compared
   * against a previous one. */
  static float direction[MAX_PARAMS], original[MAX_PARAMS];
  unsigned long long state = 987654321ULL;
  for (int i = 0; i < n; i++) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    direction[i] = (float)((state >> 33) % 2001) / 1000.0f - 1.0f;
  }

  zero_gradients();
  (void)loss_for_document();
  backward_all(probe_tokens, probe_n);

  double analytic = 0.0;
  for (int i = 0; i < n; i++) analytic += (double)grad_at(i) * direction[i];

  double ratios[2];
  char reported[128];
  reported[0] = '\0';
  for (int s = 0; s < 2; s++) {
    const float h = STEP_SIZES[s];
    for (int i = 0; i < n; i++) {
      original[i] = *param_at(i);
      *param_at(i) = original[i] + h * direction[i];
    }
    const float up = loss_for_document();
    for (int i = 0; i < n; i++) *param_at(i) = original[i] - h * direction[i];
    const float down = loss_for_document();
    for (int i = 0; i < n; i++) *param_at(i) = original[i];

    const double numeric = ((double)up - (double)down) / (2.0 * h);
    ratios[s] = analytic != 0.0 ? numeric / analytic : 0.0;
    char piece[64];
    snprintf(piece, sizeof(piece), "%sh=%g ratio=%.6f", s ? " | " : "",
             (double)h, ratios[s]);
    strncat(reported, piece, sizeof(reported) - strlen(reported) - 1);
  }

  printf("GRADCHECK analytic=%.6f params=%d ratios=%s\n", analytic, n, reported);

  int failed = 0;
  for (int s = 0; s < 2; s++) {
    if (!(ratios[s] >= BAND_LO && ratios[s] <= BAND_HI)) {
      fprintf(stderr,
              "probe: ratio %.4f at h=%g is outside the %.1f-%.1f band. The "
              "backward pass is not a correct gradient; see "
              "docs/KNOWN-ISSUES.md issue 1 for why a loss curve will not tell "
              "you this.\n",
              ratios[s], (double)STEP_SIZES[s], (double)BAND_LO, (double)BAND_HI);
      failed = 1;
    }
  }
  const double drift = ratios[0] - ratios[1];
  if (drift < 0 ? -drift > STEP_STABILITY : drift > STEP_STABILITY) {
    fprintf(stderr,
            "probe: ratios %.6f and %.6f disagree by more than %.0f%%, so the "
            "measurement itself is not trustworthy and no verdict should be read "
            "from it.\n",
            ratios[0], ratios[1], (double)STEP_STABILITY * 100.0);
    failed = 1;
  }
  if (!(analytic > -1e30 && analytic < 1e30)) {
    fprintf(stderr, "probe: the analytic directional derivative is not finite\n");
    failed = 1;
  }
  return failed;
}
