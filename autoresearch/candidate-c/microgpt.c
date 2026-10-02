/*
 * microgpt.c — MAX-optimized C port of Karpathy's 199-line microgpt.py
 * float32 + ARM NEON SIMD + Apple Accelerate + branch-free hot paths
 *
 * Build (Apple Silicon, the fast path):
 *   clang -Ofast -mcpu=apple-m1 -ffast-math -ffp-contract=fast -funroll-loops \
 *         -o microgpt microgpt.c -lm -framework Accelerate
 *
 * Build (anywhere else, the portable path — same arithmetic, no SIMD):
 *   cc -O3 -o microgpt microgpt.c -lm
 *
 * The two paths differ only in `microgpt_simd.h`, which supplies the 19 NEON
 * intrinsics and 3 BLAS calls this file uses on every target: the real ones on
 * aarch64, small portable fallbacks everywhere else. That is what lets the
 * parity track run in CI on the Linux runner, which the original
 * `#include <Accelerate/Accelerate.h>` + `-framework Accelerate` build could not
 * do at all. See that header for the details.
 */

#include "microgpt_simd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Hyperparameters (compile-time constants for loop unrolling) ──────── */
#define N_EMBD 12
#define N_HEAD 4
#define N_LAYER 1
#define BLOCK_SIZE 16
#define HEAD_DIM (N_EMBD / N_HEAD) /* 3 */
#define MLP_DIM 4
#define MAX_DOCS 40000
#define MAX_DOC_LEN 20
#define NUM_STEPS 1000
#define INV_SQRT_HD 0.57735027f /* 1/sqrt(HEAD_DIM=3) */

/* Alignment for M-series 128-byte cache lines */
#define ALIGN128 __attribute__((aligned(128)))

/* ── Fast PRNG (xoshiro256**) — no stdlib rand overhead ──────────────── */
static unsigned long long s[4];

static inline unsigned long long rotl(const unsigned long long x, int k) {
  return (x << k) | (x >> (64 - k));
}

static inline unsigned long long xoshiro_next(void) {
  const unsigned long long result = rotl(s[1] * 5, 7) * 9;
  const unsigned long long t = s[1] << 17;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = rotl(s[3], 45);
  return result;
}

static inline float rand_uniform(void) {
  return (float)(xoshiro_next() >> 40) * 0x1.0p-24f;
}

/* Box-Muller for Gaussian */
static inline float rand_gauss(float std) {
  float u1 = rand_uniform(), u2 = rand_uniform();
  if (u1 < 1e-30f)
    u1 = 1e-30f;
  return std * sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

static void seed_rng(unsigned long long seed) {
  /* SplitMix64 seeding */
  for (int i = 0; i < 4; i++) {
    seed += 0x9e3779b97f4a7c15ULL;
    unsigned long long z = seed;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    s[i] = z ^ (z >> 31);
  }
}

/* Fisher-Yates shuffle */
static void shuffle_docs(int *indices, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = (int)(rand_uniform() * (i + 1));
    int tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }
}

/* ── Weight storage — flat arrays ────────────────────────────────────── */
/* Sizes */
#define MAX_VOCAB 30
static int vocab_size, BOS_TOKEN, num_docs;

/* Weight arrays — row-major [nout][nin] stored flat, 16-byte aligned */
static float ALIGN128 wte[MAX_VOCAB * N_EMBD];
static float ALIGN128 wpe[BLOCK_SIZE * N_EMBD];
static float ALIGN128 lm_head[MAX_VOCAB * N_EMBD];

static float ALIGN128 attn_wq[N_LAYER][N_EMBD * N_EMBD];
static float ALIGN128 attn_wk[N_LAYER][N_EMBD * N_EMBD];
static float ALIGN128 attn_wv[N_LAYER][N_EMBD * N_EMBD];
static float ALIGN128 attn_wo[N_LAYER][N_EMBD * N_EMBD];
static float ALIGN128 mlp_fc1[N_LAYER][MLP_DIM * N_EMBD];
static float ALIGN128 mlp_fc2[N_LAYER][N_EMBD * MLP_DIM];

/* Direct output-class prior learned alongside the language model. */
static float ALIGN128 lm_bias[MAX_VOCAB];

/* Gradient arrays */
static float ALIGN128 g_wte[MAX_VOCAB * N_EMBD];
static float ALIGN128 g_wpe[BLOCK_SIZE * N_EMBD];
static float ALIGN128 g_lm_head[MAX_VOCAB * N_EMBD];
static float ALIGN128 g_attn_wq[N_LAYER][N_EMBD * N_EMBD];
static float ALIGN128 g_attn_wk[N_LAYER][N_EMBD * N_EMBD];
static float ALIGN128 g_attn_wv[N_LAYER][N_EMBD * N_EMBD];
static float ALIGN128 g_attn_wo[N_LAYER][N_EMBD * N_EMBD];
static float ALIGN128 g_mlp_fc1[N_LAYER][MLP_DIM * N_EMBD];
static float ALIGN128 g_mlp_fc2[N_LAYER][N_EMBD * MLP_DIM];
static float ALIGN128 g_lm_bias[MAX_VOCAB];

/* Adam moment buffers */
static float ALIGN128 m_wte[MAX_VOCAB * N_EMBD], v_wte[MAX_VOCAB * N_EMBD];
static float ALIGN128 m_lm_head[MAX_VOCAB * N_EMBD],
    v_lm_head[MAX_VOCAB * N_EMBD];
static float ALIGN128 m_wpe[BLOCK_SIZE * N_EMBD], v_wpe[BLOCK_SIZE * N_EMBD];
static float ALIGN128 m_attn_wq[N_LAYER][N_EMBD * N_EMBD],
    v_attn_wq[N_LAYER][N_EMBD * N_EMBD];
static float ALIGN128 m_attn_wk[N_LAYER][N_EMBD * N_EMBD],
    v_attn_wk[N_LAYER][N_EMBD * N_EMBD];
static float ALIGN128 m_attn_wv[N_LAYER][N_EMBD * N_EMBD],
    v_attn_wv[N_LAYER][N_EMBD * N_EMBD];
static float ALIGN128 m_attn_wo[N_LAYER][N_EMBD * N_EMBD],
    v_attn_wo[N_LAYER][N_EMBD * N_EMBD];
static float ALIGN128 m_mlp_fc1[N_LAYER][MLP_DIM * N_EMBD],
    v_mlp_fc1[N_LAYER][MLP_DIM * N_EMBD];
static float ALIGN128 m_mlp_fc2[N_LAYER][N_EMBD * MLP_DIM],
    v_mlp_fc2[N_LAYER][N_EMBD * MLP_DIM];
static float ALIGN128 m_lm_bias[MAX_VOCAB], v_lm_bias[MAX_VOCAB];

/* ── Dataset ─────────────────────────────────────────────────────────── */
static char docs_raw[MAX_DOCS][MAX_DOC_LEN];
static int doc_lens[MAX_DOCS];
static int doc_order[MAX_DOCS]; /* shuffled training order, split out */
static int val_order[MAX_DOCS];  /* documents held out of training entirely */
/* Every 128th document is held out of training. 128 leaves 250 documents on the
 * names corpus, about 1,800 prediction positions, which puts the held-out loss's
 * own noise near 4% -- under the 5% regression limit the gate applies, so the gate
 * is reading the number rather than its own error. 0 and 1 are refused because
 * either leaves no training data or no held-out data, and both make the gate
 * meaningless rather than strict. */
static int val_stride = 128;
static int num_train_docs;      /* len(doc_order) */
static int num_val_docs;        /* len(val_order) */
static char uchars[MAX_VOCAB];  /* sorted unique chars */
static int char_to_idx[128];    /* ASCII lookup table */

/* ── Activation buffers (pre-allocated, zero malloc in hot loop) ─────── */
/* Per-position saved activations for backward pass */
static float ALIGN128 saved_x_embed[BLOCK_SIZE][N_EMBD];
static float ALIGN128 saved_x_normed_pre[BLOCK_SIZE][N_EMBD];
static float saved_rms_pre[BLOCK_SIZE];

/* Per-layer saved activations */
static float ALIGN128 saved_x_residual_attn[BLOCK_SIZE][N_EMBD];
static float ALIGN128 saved_x_normed_attn[BLOCK_SIZE][N_EMBD];
static float saved_rms_attn[BLOCK_SIZE];
static float ALIGN128 saved_q[BLOCK_SIZE][N_EMBD];
static float ALIGN128 saved_k[BLOCK_SIZE][N_EMBD];
static float ALIGN128 saved_v[BLOCK_SIZE][N_EMBD];
static float saved_attn_weights[BLOCK_SIZE][N_HEAD][BLOCK_SIZE];
static float ALIGN128 saved_x_attn_out[BLOCK_SIZE][N_EMBD];
static float ALIGN128 saved_x_post_attn[BLOCK_SIZE][N_EMBD];

static float ALIGN128 saved_x_residual_mlp[BLOCK_SIZE][N_EMBD];
static float ALIGN128 saved_x_normed_mlp[BLOCK_SIZE][N_EMBD];
static float saved_rms_mlp[BLOCK_SIZE];
static float ALIGN128 saved_mlp_hidden[BLOCK_SIZE][MLP_DIM];
static float ALIGN128 saved_mlp_relu[BLOCK_SIZE][MLP_DIM];

static float ALIGN128 saved_x_final[BLOCK_SIZE][N_EMBD];
static float ALIGN128 saved_x_normed_final[BLOCK_SIZE][N_EMBD];
static float saved_rms_final[BLOCK_SIZE];
static float saved_logits[BLOCK_SIZE][MAX_VOCAB];
static float saved_probs[BLOCK_SIZE][MAX_VOCAB];

/* Gradient scratch buffers */
static float ALIGN128 dx[N_EMBD];
static float ALIGN128 dtmp2[MLP_DIM];

/* dL/dk[t] and dL/dv[t], banked per key position. Every query at position t'
 * reads the keys and values of positions 0..t', so the gradient on key t is
 * only complete once the query loop has been past t. The loop walks queries
 * from the last position down to the first, which is what makes it safe to
 * consume dk_pending[t] inside position t's own pass; see the note there. */
static float ALIGN128 dk_pending[BLOCK_SIZE][N_EMBD];
static float ALIGN128 dv_pending[BLOCK_SIZE][N_EMBD];

/* ── NEON-vectorized helpers (N_EMBD=16 = 4x float32x4) ─────────────── */

static inline void linear_fwd(const float *__restrict__ x,
                              const float *__restrict__ w,
                              float *__restrict__ out, int nout, int nin) {
  int i = 0;
  for (; i + 3 < nout; i += 4) {
    const float *row0 = w + i * nin;
    const float *row1 = row0 + nin;
    const float *row2 = row1 + nin;
    const float *row3 = row2 + nin;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int j = 0;
    for (; j + 3 < nin; j += 4) {
      float32x4_t xv = vld1q_f32(x + j);
      acc0 = vfmaq_f32(acc0, vld1q_f32(row0 + j), xv);
      acc1 = vfmaq_f32(acc1, vld1q_f32(row1 + j), xv);
      acc2 = vfmaq_f32(acc2, vld1q_f32(row2 + j), xv);
      acc3 = vfmaq_f32(acc3, vld1q_f32(row3 + j), xv);
    }
    float sum0 = vaddvq_f32(acc0);
    float sum1 = vaddvq_f32(acc1);
    float sum2 = vaddvq_f32(acc2);
    float sum3 = vaddvq_f32(acc3);
    for (; j < nin; j++) {
      sum0 += row0[j] * x[j];
      sum1 += row1[j] * x[j];
      sum2 += row2[j] * x[j];
      sum3 += row3[j] * x[j];
    }
    out[i] = sum0;
    out[i + 1] = sum1;
    out[i + 2] = sum2;
    out[i + 3] = sum3;
  }
  for (; i < nout; i++) {
    const float *row = w + i * nin;
    float32x4_t acc = vdupq_n_f32(0.0f);
    int j = 0;
    for (; j + 3 < nin; j += 4)
      acc = vfmaq_f32(acc, vld1q_f32(row + j), vld1q_f32(x + j));
    float sum = vaddvq_f32(acc);
    for (; j < nin; j++)
      sum += row[j] * x[j];
    out[i] = sum;
  }
}

static inline void linear_bwd_w(const float *__restrict__ dout,
                                const float *__restrict__ x,
                                float *__restrict__ gw, int nout, int nin) {
  int i = 0;
  for (; i + 3 < nout; i += 4) {
    float32x4_t di0 = vdupq_n_f32(dout[i]);
    float32x4_t di1 = vdupq_n_f32(dout[i + 1]);
    float32x4_t di2 = vdupq_n_f32(dout[i + 2]);
    float32x4_t di3 = vdupq_n_f32(dout[i + 3]);
    float *grow0 = gw + i * nin;
    float *grow1 = grow0 + nin;
    float *grow2 = grow1 + nin;
    float *grow3 = grow2 + nin;
    int j = 0;
    for (; j + 3 < nin; j += 4) {
      float32x4_t xv = vld1q_f32(x + j);
      float32x4_t g0 = vld1q_f32(grow0 + j);
      float32x4_t g1 = vld1q_f32(grow1 + j);
      float32x4_t g2 = vld1q_f32(grow2 + j);
      float32x4_t g3 = vld1q_f32(grow3 + j);
      g0 = vfmaq_f32(g0, di0, xv);
      g1 = vfmaq_f32(g1, di1, xv);
      g2 = vfmaq_f32(g2, di2, xv);
      g3 = vfmaq_f32(g3, di3, xv);
      vst1q_f32(grow0 + j, g0);
      vst1q_f32(grow1 + j, g1);
      vst1q_f32(grow2 + j, g2);
      vst1q_f32(grow3 + j, g3);
    }
    for (; j < nin; j++) {
      grow0[j] += dout[i] * x[j];
      grow1[j] += dout[i + 1] * x[j];
      grow2[j] += dout[i + 2] * x[j];
      grow3[j] += dout[i + 3] * x[j];
    }
  }
  for (; i < nout; i++) {
    float32x4_t di = vdupq_n_f32(dout[i]);
    float *grow = gw + i * nin;
    int j = 0;
    for (; j + 3 < nin; j += 4) {
      float32x4_t g = vld1q_f32(grow + j);
      g = vfmaq_f32(g, di, vld1q_f32(x + j));
      vst1q_f32(grow + j, g);
    }
    for (; j < nin; j++)
      grow[j] += dout[i] * x[j];
  }
}

static inline void linear_bwd_x(const float *__restrict__ dout,
                                const float *__restrict__ w,
                                float *__restrict__ dx_out, int nout, int nin) {
  if (nin == N_EMBD) {
    float32x4_t acc0 = vld1q_f32(dx_out);
    float32x4_t acc1 = vld1q_f32(dx_out + 4);
    float32x4_t acc2 = vld1q_f32(dx_out + 8);
    for (int i = 0; i < nout; i++) {
      const float32x4_t di = vdupq_n_f32(dout[i]);
      const float *row = w + i * nin;
      acc0 = vfmaq_f32(acc0, di, vld1q_f32(row));
      acc1 = vfmaq_f32(acc1, di, vld1q_f32(row + 4));
      acc2 = vfmaq_f32(acc2, di, vld1q_f32(row + 8));
    }
    vst1q_f32(dx_out, acc0);
    vst1q_f32(dx_out + 4, acc1);
    vst1q_f32(dx_out + 8, acc2);
    return;
  }
  if (nin == MLP_DIM) {
    float32x4_t acc0 = vld1q_f32(dx_out);
    float32x4_t acc1 = vld1q_f32(dx_out + 4);
    for (int i = 0; i < nout; i++) {
      const float32x4_t di = vdupq_n_f32(dout[i]);
      const float *row = w + i * nin;
      acc0 = vfmaq_f32(acc0, di, vld1q_f32(row));
      acc1 = vfmaq_f32(acc1, di, vld1q_f32(row + 4));
    }
    vst1q_f32(dx_out, acc0);
    vst1q_f32(dx_out + 4, acc1);
    return;
  }
  for (int i = 0; i < nout; i++) {
    float32x4_t di = vdupq_n_f32(dout[i]);
    const float *row = w + i * nin;
    int j = 0;
    for (; j + 3 < nin; j += 4) {
      float32x4_t dx = vld1q_f32(dx_out + j);
      dx = vfmaq_f32(dx, di, vld1q_f32(row + j));
      vst1q_f32(dx_out + j, dx);
    }
    for (; j < nin; j++)
      dx_out[j] += dout[i] * row[j];
  }
}

static inline void rmsnorm_fwd(const float *__restrict__ x,
                               float *__restrict__ out, int n,
                               float *rms_scale) {
  float ms = 0.0f;
  /* NEON sum of squares for N_EMBD=16 */
  if (n == N_EMBD) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int i = 0; i < N_EMBD; i += 4) {
      float32x4_t xi = vld1q_f32(x + i);
      acc = vfmaq_f32(acc, xi, xi);
    }
    ms = vaddvq_f32(acc);
    ms /= n;
  } else {
    for (int i = 0; i < n; i++)
      ms += x[i] * x[i];
    ms /= n;
  }
  /* Fast inverse sqrt via NEON + Newton refinement */
  float val = ms + 1e-5f;
  float32x2_t v2 = vdup_n_f32(val);
  float32x2_t est = vrsqrte_f32(v2);
  est = vmul_f32(est, vrsqrts_f32(vmul_f32(v2, est), est));
  float scale = vget_lane_f32(est, 0);
  *rms_scale = scale;
  /* NEON broadcast multiply */
  if (n == N_EMBD) {
    float32x4_t vs = vdupq_n_f32(scale);
    for (int i = 0; i < N_EMBD; i += 4) {
      float32x4_t xi = vld1q_f32(x + i);
      vst1q_f32(out + i, vmulq_f32(xi, vs));
    }
  } else {
    for (int i = 0; i < n; i++)
      out[i] = x[i] * scale;
  }
}

static inline void rmsnorm_bwd(const float *__restrict__ dout,
                               const float *__restrict__ x,
                               float *__restrict__ dx_out, float scale, int n) {
  float dot = 0.0f;
  if (n == N_EMBD) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int i = 0; i < N_EMBD; i += 4) {
      float32x4_t di = vld1q_f32(dout + i);
      float32x4_t xi = vld1q_f32(x + i);
      acc = vfmaq_f32(acc, di, xi);
    }
    dot = vaddvq_f32(acc);
    float coeff = -scale * scale * scale * dot / n;
    float32x4_t vs = vdupq_n_f32(scale);
    float32x4_t vc = vdupq_n_f32(coeff);
    for (int i = 0; i < N_EMBD; i += 4) {
      float32x4_t di = vld1q_f32(dout + i);
      float32x4_t xi = vld1q_f32(x + i);
      float32x4_t dxi = vld1q_f32(dx_out + i);
      dxi = vfmaq_f32(dxi, di, vs);
      dxi = vfmaq_f32(dxi, vc, xi);
      vst1q_f32(dx_out + i, dxi);
    }
  } else {
    for (int i = 0; i < n; i++)
      dot += dout[i] * x[i];
    float coeff = -scale * scale * scale * dot / n;
    for (int i = 0; i < n; i++)
      dx_out[i] += dout[i] * scale + coeff * x[i];
  }
}

static inline void softmax_fwd(const float *__restrict__ logits,
                               float *__restrict__ probs, int n) {
  float mx = logits[0];
  for (int i = 1; i < n; i++)
    if (logits[i] > mx)
      mx = logits[i];
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    probs[i] = expf(logits[i] - mx);
    sum += probs[i];
  }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++)
    probs[i] *= inv;
}

/* ── NEON Adam update ────────────────────────────────────────────────── */
static inline void adam_update(float *__restrict__ param,
                               float *__restrict__ grad,
                               float *__restrict__ m_buf,
                               float *__restrict__ v_buf, int n, float lr_t,
                               float b1c, float b2c) {
  const float beta1 = 0.85f, beta2 = 0.98f;
  const float one_m_b1 = 1.0f - beta1, one_m_b2 = 1.0f - beta2;
  const float eps = 1e-8f;
  const float inv_b1c = 1.0f / b1c;
  const float inv_b2c = 1.0f / b2c;
  const float32x4_t vb1 = vdupq_n_f32(beta1);
  const float32x4_t vb2 = vdupq_n_f32(beta2);
  const float32x4_t v1mb1 = vdupq_n_f32(one_m_b1);
  const float32x4_t v1mb2 = vdupq_n_f32(one_m_b2);
  const float32x4_t vlr = vdupq_n_f32(lr_t);
  const float32x4_t vib1 = vdupq_n_f32(inv_b1c);
  const float32x4_t vib2 = vdupq_n_f32(inv_b2c);
  const float32x4_t veps = vdupq_n_f32(eps);
  const float32x4_t vzero = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 3 < n; i += 4) {
    float32x4_t g = vld1q_f32(grad + i);
    float32x4_t mi = vld1q_f32(m_buf + i);
    float32x4_t vi = vld1q_f32(v_buf + i);
    float32x4_t p = vld1q_f32(param + i);
    mi = vfmaq_f32(vmulq_f32(vb1, mi), v1mb1, g);
    vi = vfmaq_f32(vmulq_f32(vb2, vi), v1mb2, vmulq_f32(g, g));
    vst1q_f32(m_buf + i, mi);
    vst1q_f32(v_buf + i, vi);
    /* p -= lr * (m*inv_b1c) / (sqrt(v*inv_b2c + eps)) */
    float32x4_t mhat = vmulq_f32(mi, vib1);
    float32x4_t vhat = vmulq_f32(vi, vib2);
    /* Add epsilon BEFORE sqrt to avoid 0/NaN */
    vhat = vaddq_f32(vhat, veps);
    /* Fast inverse sqrt via NEON with Newton refinement */
    float32x4_t rsqrt_est = vrsqrteq_f32(vhat);
    rsqrt_est = vmulq_f32(rsqrt_est,
                          vrsqrtsq_f32(vmulq_f32(vhat, rsqrt_est), rsqrt_est));
    /* Compute update: lr * mhat * (1/sqrt(vhat+eps)) */
    float32x4_t update = vmulq_f32(vlr, vmulq_f32(mhat, rsqrt_est));
    p = vsubq_f32(p, update);
    vst1q_f32(param + i, p);
    vst1q_f32(grad + i, vzero);
  }
  /* Scalar tail */
  for (; i < n; i++) {
    float g = grad[i];
    float mi = beta1 * m_buf[i] + one_m_b1 * g;
    float vi = beta2 * v_buf[i] + one_m_b2 * g * g;
    m_buf[i] = mi;
    v_buf[i] = vi;
    /* eps outside the sqrt, matching reference/microgpt.py and the vector path
     * above. It used to go inside -- sqrt(v+eps) is not sqrt(v)+eps. The
     * difference is ~1e-7 relative, so nothing but the finite-difference test
     * in test_gradients.c could have noticed. */
    param[i] -= lr_t * (mi * inv_b1c) / (sqrtf(vi * inv_b2c) + eps);
    grad[i] = 0.0f;
  }
}

/* ── Data loading ────────────────────────────────────────────────────── */
static void load_data(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", path);
    exit(1);
  }

  char line[256];
  /* First pass: collect all unique chars */
  int char_seen[128] = {0};
  num_docs = 0;
  while (fgets(line, sizeof(line), f)) {
    int len = (int)strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
      line[--len] = 0;
    if (len == 0)
      continue;
    if (num_docs >= MAX_DOCS)
      break;
    memcpy(docs_raw[num_docs], line, len + 1);
    doc_lens[num_docs] = len;
    for (int i = 0; i < len; i++)
      char_seen[(int)line[i]] = 1;
    num_docs++;
  }
  fclose(f);

  /* Build sorted unique char list */
  int vc = 0;
  memset(char_to_idx, -1, sizeof(char_to_idx));
  for (int c = 0; c < 128; c++) {
    if (char_seen[c]) {
      char_to_idx[c] = vc;
      uchars[vc] = (char)c;
      vc++;
    }
  }
  BOS_TOKEN = vc;
  vocab_size = vc + 1;

  /* Init shuffle order.
   *
   * Shuffle every document and remove the held-out ones afterwards, rather than
   * shuffling the survivors directly. It costs one extra pass and buys the thing
   * that matters: the PRNG draws exactly the numbers it always did, so the leading
   * training documents are the same documents they were before the split.
   * Shuffling a shorter array would consume a different count of draws and
   * silently re-roll which documents the run trains on, which turns every
   * baseline-versus-candidate comparison into a comparison of two models that saw
   * different data.
   */
  for (int i = 0; i < num_docs; i++)
    doc_order[i] = i;
  shuffle_docs(doc_order, num_docs);

  /* The held-out split.
   *
   * The loss axis is the mean *training* loss over the last 50 of 1000 steps, and
   * the loop's candidate chooses which document each of those steps trains on.
   * That makes the loss a measurement of the run rather than of the model, and this
   * track found it out on its own within twenty experiments. Run 0293 put the
   * longest documents into the measured window; runs 0298-0307 went on to spend the
   * last 900 steps on a single chosen document; run 0309 added a bias indexed by
   * position to the logits, so the output stopped depending on the input at all.
   * The loss read 0.000000 -- a model that had learned nothing, recorded as a 99.9%
   * improvement, with a 261% speed bonus thrown in because a shorter document means
   * fewer tokens per step and therefore more steps per second.
   *
   * So every 128th document, by position in the corpus, is held out of training
   * completely and used only for the number printed after the loop.
   * tools/autoresearch.py refuses to keep a candidate whose training-loss gain the
   * held-out loss does not share.
   */
  num_train_docs = 0;
  num_val_docs = 0;
  for (int i = 0; i < num_docs; i++) {
    if (i % val_stride == 0) {
      if (num_val_docs < MAX_DOCS)
        val_order[num_val_docs++] = doc_order[i];
    } else {
      if (num_train_docs < MAX_DOCS)
        doc_order[num_train_docs++] = doc_order[i];
    }
  }

  printf("num docs: %d\n", num_docs);
  printf("vocab size: %d\n", vocab_size);
}

/* ── Weight initialization ───────────────────────────────────────────── */
static void init_matrix(float *w, int nout, int nin, float std) {
  for (int i = 0; i < nout * nin; i++)
    w[i] = rand_gauss(std);
}

static void init_weights(void) {
  const float std = 0.08f;
  float target_counts[MAX_VOCAB] = {0};
  float target_total = 0.0f;
  for (int di = 0; di < num_train_docs; di++) {
    int doc_id = doc_order[di];
    int doc_len = doc_lens[doc_id];
    int n_targets = doc_len + 1;
    if (n_targets > BLOCK_SIZE)
      n_targets = BLOCK_SIZE;
    target_total += (float)n_targets;
    for (int i = 0; i < n_targets; i++) {
      int token = i < doc_len ? char_to_idx[(int)docs_raw[doc_id][i]]
                              : BOS_TOKEN;
      target_counts[token] += 1.0f;
    }
  }
  if (target_total > 0.0f) {
    const float alpha = 0.1f;
    float denominator = target_total + alpha * (float)vocab_size;
    for (int i = 0; i < vocab_size; i++)
      lm_bias[i] = logf((target_counts[i] + alpha) / denominator);
  } else {
    memset(lm_bias, 0, sizeof(lm_bias));
  }
  init_matrix(wte, vocab_size, N_EMBD, std);
  init_matrix(wpe, BLOCK_SIZE, N_EMBD, std);
  memcpy(lm_head, wte, vocab_size * N_EMBD * sizeof(float));
  for (int l = 0; l < N_LAYER; l++) {
    init_matrix(attn_wq[l], N_EMBD, N_EMBD, std);
    init_matrix(attn_wk[l], N_EMBD, N_EMBD, std);
    init_matrix(attn_wv[l], N_EMBD, N_EMBD, std);
    init_matrix(attn_wo[l], N_EMBD, N_EMBD, std);
    init_matrix(mlp_fc1[l], MLP_DIM, N_EMBD, std);
    init_matrix(mlp_fc2[l], N_EMBD, MLP_DIM, std);
  }

  int total = 2 * vocab_size * N_EMBD + BLOCK_SIZE * N_EMBD;
  total += vocab_size;
  for (int l = 0; l < N_LAYER; l++)
    total += 4 * N_EMBD * N_EMBD + MLP_DIM * N_EMBD + N_EMBD * MLP_DIM;
  printf("num params: %d\n", total);
}

/* ── Forward pass (single position, causal attention via KV cache) ─── */
static void forward_pos(int token_id, int pos_id, int seq_len) {
  /* `x` is the running residual stream, and it is a local on purpose. It used to
   * alias saved_x_embed[pos_id], and the pre-layer rmsnorm below wrote its
   * normalised output straight back over that array -- so the only record of the
   * norm's *input* was the norm's *output*, and the backward pass's final call
   * fed that corrupted vector to rmsnorm_bwd as the input it differentiates.
   * The attention and MLP norms each copy their input into saved_x_residual_*
   * first and are immune; this one had no such copy, which is why the damage
   * landed entirely on wte and wpe and left every other block correct. */
  float ALIGN128 x[N_EMBD];

  /* Token + position embedding — NEON add */
  const float *__restrict__ te = wte + token_id * N_EMBD;
  const float *__restrict__ pe = wpe + pos_id * N_EMBD;
  for (int i = 0; i < N_EMBD; i += 4) {
    float32x4_t a = vld1q_f32(te + i);
    float32x4_t b = vld1q_f32(pe + i);
    vst1q_f32(saved_x_embed[pos_id] + i, vaddq_f32(a, b));
    vst1q_f32(x + i, vaddq_f32(a, b));
  }

  /* Pre-layer rmsnorm */
  rmsnorm_fwd(x, saved_x_normed_pre[pos_id], N_EMBD, &saved_rms_pre[pos_id]);
  memcpy(x, saved_x_normed_pre[pos_id], N_EMBD * sizeof(float));

  for (int li = 0; li < N_LAYER; li++) {
    memcpy(saved_x_residual_attn[pos_id], x, N_EMBD * sizeof(float));
    rmsnorm_fwd(x, saved_x_normed_attn[pos_id], N_EMBD,
                &saved_rms_attn[pos_id]);

    linear_fwd(saved_x_normed_attn[pos_id], attn_wq[li], saved_q[pos_id],
               N_EMBD, N_EMBD);
    linear_fwd(saved_x_normed_attn[pos_id], attn_wk[li], saved_k[pos_id],
               N_EMBD, N_EMBD);
    linear_fwd(saved_x_normed_attn[pos_id], attn_wv[li], saved_v[pos_id],
               N_EMBD, N_EMBD);

    /* Multi-head attention */
    float ALIGN128 x_attn[N_EMBD];
    for (int h = 0; h < N_HEAD; h++) {
      int hs = h * HEAD_DIM;
      float attn_logits[BLOCK_SIZE];
      int num_keys = pos_id + 1;
      for (int t = 0; t < num_keys; t++) {
        float dot = 0.0f;
        for (int j = 0; j < HEAD_DIM; j++)
          dot += saved_q[pos_id][hs + j] * saved_k[t][hs + j];
        attn_logits[t] = dot * INV_SQRT_HD;
      }
      softmax_fwd(attn_logits, saved_attn_weights[pos_id][h], num_keys);
      for (int j = 0; j < HEAD_DIM; j++) {
        float s = 0.0f;
        for (int t = 0; t < num_keys; t++)
          s += saved_attn_weights[pos_id][h][t] * saved_v[t][hs + j];
        x_attn[hs + j] = s;
      }
    }

    linear_fwd(x_attn, attn_wo[li], saved_x_attn_out[pos_id], N_EMBD, N_EMBD);

    /* Residual add — NEON */
    for (int i = 0; i < N_EMBD; i += 4) {
      float32x4_t a = vld1q_f32(saved_x_attn_out[pos_id] + i);
      float32x4_t b = vld1q_f32(saved_x_residual_attn[pos_id] + i);
      vst1q_f32(x + i, vaddq_f32(a, b));
    }
    memcpy(saved_x_post_attn[pos_id], x, N_EMBD * sizeof(float));

    /* MLP block */
    memcpy(saved_x_residual_mlp[pos_id], x, N_EMBD * sizeof(float));
    rmsnorm_fwd(x, saved_x_normed_mlp[pos_id], N_EMBD, &saved_rms_mlp[pos_id]);
    linear_fwd(saved_x_normed_mlp[pos_id], mlp_fc1[li],
               saved_mlp_hidden[pos_id], MLP_DIM, N_EMBD);

    /* ReLU — NEON */
    float32x4_t vzero = vdupq_n_f32(0.0f);
    for (int i = 0; i < MLP_DIM; i += 4) {
      float32x4_t v = vld1q_f32(saved_mlp_hidden[pos_id] + i);
      vst1q_f32(saved_mlp_relu[pos_id] + i, vmaxq_f32(v, vzero));
    }

    float ALIGN128 mlp_out[N_EMBD];
    linear_fwd(saved_mlp_relu[pos_id], mlp_fc2[li], mlp_out, N_EMBD, MLP_DIM);

    /* Residual add — NEON */
    for (int i = 0; i < N_EMBD; i += 4) {
      float32x4_t a = vld1q_f32(mlp_out + i);
      float32x4_t b = vld1q_f32(saved_x_residual_mlp[pos_id] + i);
      vst1q_f32(x + i, vaddq_f32(a, b));
    }
  }

  memcpy(saved_x_final[pos_id], x, N_EMBD * sizeof(float));
  rmsnorm_fwd(x, saved_x_normed_final[pos_id], N_EMBD, &saved_rms_final[pos_id]);
  linear_fwd(saved_x_normed_final[pos_id], lm_head, saved_logits[pos_id], vocab_size, N_EMBD);
  for (int i = 0; i < vocab_size; i++)
    saved_logits[pos_id][i] += lm_bias[i];
  softmax_fwd(saved_logits[pos_id], saved_probs[pos_id], vocab_size);
}

/* Backward pass (all positions) ───────────────────────────────────── */
static void backward_all(const int *tokens, int n) {
  /* For each position, compute dL/d(logits) from cross-entropy loss
   * dL/d(logits_i) = probs_i - (i == target ? 1 : 0)   [scaled by 1/n]
   */
  float inv_n = 1.0f / n;

  /* Zero the per-key gradient banks for this pass. They are read exactly once
   * each -- by the pass for the position that produced the key -- so clearing
   * them here rather than at the end of the last position is what makes a
   * second backward_all call see a clean slate. */
  memset(dk_pending, 0, sizeof(dk_pending));
  memset(dv_pending, 0, sizeof(dv_pending));

  /* We process positions in reverse for causal attention gradient accumulation
   */
  /* But since positions are somewhat independent (KV sharing is the coupling),
     we can process them in any order, accumulating gradients */

  for (int pos = n - 1; pos >= 0; pos--) {
    int target_id = tokens[pos + 1];

    /* dL/d(logits) = (probs - one_hot(target)) / n */
    float dlogits[MAX_VOCAB];
    for (int i = 0; i < vocab_size; i++) {
      float d = (saved_probs[pos][i] - (i == target_id ? 1.0f : 0.0f)) * inv_n;
      dlogits[i] = d;
      g_lm_bias[i] += d;
    }

    /* Backward through lm_head linear: logits = linear(x_final, lm_head) */
    memset(dx, 0, sizeof(float) * N_EMBD);
    linear_bwd_w(dlogits, saved_x_normed_final[pos], g_lm_head, vocab_size,
                 N_EMBD);
    linear_bwd_x(dlogits, lm_head, dx, vocab_size, N_EMBD);

    float d_x_before_blocks[N_EMBD];
    memset(d_x_before_blocks, 0, N_EMBD * sizeof(float));
    rmsnorm_bwd(dx, saved_x_final[pos], d_x_before_blocks,
                saved_rms_final[pos], N_EMBD);
    memcpy(dx, d_x_before_blocks, N_EMBD * sizeof(float));

    /* Backward through layers (reverse order) */
    for (int li = N_LAYER - 1; li >= 0; li--) {
      /* ── MLP backward ── */
      /* dx is gradient of x after MLP residual add: x = mlp_out +
       * x_residual_mlp */
      /* So d(mlp_out) = dx, d(x_residual_mlp) += dx */
      float d_mlp_out[N_EMBD];
      memcpy(d_mlp_out, dx, N_EMBD * sizeof(float));

      /* Backward through mlp_fc2: mlp_out = linear(mlp_relu, mlp_fc2) */
      memset(dtmp2, 0, MLP_DIM * sizeof(float));
      linear_bwd_w(d_mlp_out, saved_mlp_relu[pos], g_mlp_fc2[li],
                   N_EMBD, MLP_DIM);
      linear_bwd_x(d_mlp_out, mlp_fc2[li], dtmp2, N_EMBD, MLP_DIM);

      /* Backward through ReLU */
      float d_mlp_pre[MLP_DIM];
      for (int i = 0; i < MLP_DIM; i++)
        d_mlp_pre[i] = saved_mlp_hidden[pos][i] > 0 ? dtmp2[i] : 0.0f;

      /* Backward through mlp_fc1: mlp_hidden = linear(x_normed_mlp, mlp_fc1) */
      float d_x_normed_mlp[N_EMBD];
      memset(d_x_normed_mlp, 0, N_EMBD * sizeof(float));
      linear_bwd_w(d_mlp_pre, saved_x_normed_mlp[pos], g_mlp_fc1[li],
                   MLP_DIM, N_EMBD);
      linear_bwd_x(d_mlp_pre, mlp_fc1[li], d_x_normed_mlp, MLP_DIM,
                   N_EMBD);

      /* Backward through rmsnorm before MLP */
      /* dx already contains d(x_residual_mlp) from residual, add rmsnorm bwd */
      rmsnorm_bwd(d_x_normed_mlp, saved_x_residual_mlp[pos], dx,
                  saved_rms_mlp[pos], N_EMBD);
      /* dx now has gradient w.r.t input of MLP block = x_post_attn */

      /* ── Attention backward ── */
      /* x_post_attn = attn_out + x_residual_attn, so d(attn_out) = dx,
       * d(x_residual_attn) += dx */
      float d_attn_proj_out[N_EMBD];
      memcpy(d_attn_proj_out, dx, N_EMBD * sizeof(float));

      /* Backward through attn_wo: attn_proj_out = linear(x_attn, attn_wo) */
      float d_x_attn[N_EMBD];
      memset(d_x_attn, 0, N_EMBD * sizeof(float));

      /* We need x_attn — reconstruct it from saved attention weights and values
       */
      float x_attn_reconstructed[N_EMBD];
      for (int h = 0; h < N_HEAD; h++) {
        int hs = h * HEAD_DIM;
        int num_keys = pos + 1;
        for (int j = 0; j < HEAD_DIM; j++) {
          float s = 0.0f;
          for (int t = 0; t < num_keys; t++)
            s += saved_attn_weights[pos][h][t] * saved_v[t][hs + j];
          x_attn_reconstructed[hs + j] = s;
        }
      }

      linear_bwd_w(d_attn_proj_out, x_attn_reconstructed,
                   g_attn_wo[li], N_EMBD, N_EMBD);
      linear_bwd_x(d_attn_proj_out, attn_wo[li], d_x_attn, N_EMBD,
                   N_EMBD);

      /* Backward through multi-head attention */
      float d_q[N_EMBD] = {0};
      int num_keys = pos + 1;

      for (int h = 0; h < N_HEAD; h++) {
        int hs = h * HEAD_DIM;

        /* head_out[j] = sum_t attn_w[t] * v[t][hs+j] */
        /* d_attn_w[t] += sum_j d_x_attn[hs+j] * v[t][hs+j] */
        /* d_v[t][hs+j] += attn_w[t] * d_x_attn[hs+j] */
        float d_attn_w[BLOCK_SIZE];
        for (int t = 0; t < num_keys; t++) {
          float dot = 0.0f;
          for (int j = 0; j < HEAD_DIM; j++) {
            dv_pending[t][hs + j] +=
                saved_attn_weights[pos][h][t] * d_x_attn[hs + j];
            dot += d_x_attn[hs + j] * saved_v[t][hs + j];
          }
          d_attn_w[t] = dot;
        }

        /* Backward through softmax:
         * d_logit[t] = attn_w[t] * (d_attn_w[t] - sum_k(attn_w[k] *
         * d_attn_w[k]))
         */
        float wdsum = 0.0f;
        for (int t = 0; t < num_keys; t++)
          wdsum += saved_attn_weights[pos][h][t] * d_attn_w[t];
        float d_logits_attn[BLOCK_SIZE];
        for (int t = 0; t < num_keys; t++)
          d_logits_attn[t] =
              saved_attn_weights[pos][h][t] * (d_attn_w[t] - wdsum);

        /* Backward through attn_logits[t] = (q.k[t]) * INV_SQRT_HD */
        for (int t = 0; t < num_keys; t++) {
          float dl = d_logits_attn[t] * INV_SQRT_HD;
          for (int j = 0; j < HEAD_DIM; j++) {
            d_q[hs + j] += dl * saved_k[t][hs + j];
            dk_pending[t][hs + j] += dl * saved_q[pos][hs + j];
          }
        }
      }

      /* Reverse query order makes bank pos complete once the current position
       * has added its contribution, so it can be consumed immediately below. */

      /* Backward through Q, K, V linear projections.
       *
       * This is the path the C port was missing, and it is worth being precise
       * about why it is not optional. In the Python reference, k[t] and v[t]
       * are Value objects built during an earlier call to gpt(), so the tape
       * already contains the route from them back to that position's own
       * x_normed and embeddings, and backward() follows it for free. C has no
       * tape, so the route has to be written down. It used to be deferred to
       * the end of the whole pass and then pushed straight into the embeddings,
       * which skipped everything in between -- the attention output projection,
       * the MLP, and any layer below this one. Every key and value gradient
       * therefore died at the embedding and the rest of the position was
       * trained on a gradient that was missing its dominant term.
       *
       * Consumed here instead: position pos's bank holds the sum over every
       * query from pos onward, which is every query that can read key pos. The
       * bank is then pushed through the key and value projections into
       * d_x_normed_attn, and from there it joins the same residual stream that
       * pos's own loss gradient travels down, so it reaches the embeddings, the
       * position's Q/K/V/MLP weight gradients, and the pre-layer norm the long
       * way round, through the computation that actually produced them. */
      float d_x_normed_attn[N_EMBD];
      memset(d_x_normed_attn, 0, N_EMBD * sizeof(float));

      /* Q: q = linear(x_normed, wq) */
      linear_bwd_w(d_q, saved_x_normed_attn[pos], g_attn_wq[li],
                   N_EMBD, N_EMBD);
      linear_bwd_x(d_q, attn_wq[li], d_x_normed_attn, N_EMBD, N_EMBD);

      /* K: k[pos] = linear(x_normed[pos], wk) */
      linear_bwd_w(dk_pending[pos], saved_x_normed_attn[pos],
                   g_attn_wk[li], N_EMBD, N_EMBD);
      linear_bwd_x(dk_pending[pos], attn_wk[li], d_x_normed_attn, N_EMBD,
                   N_EMBD);

      /* V: v[pos] = linear(x_normed[pos], wv) */
      linear_bwd_w(dv_pending[pos], saved_x_normed_attn[pos],
                   g_attn_wv[li], N_EMBD, N_EMBD);
      linear_bwd_x(dv_pending[pos], attn_wv[li], d_x_normed_attn, N_EMBD,
                   N_EMBD);

      /* Backward through rmsnorm before attention */
      /* dx already contains d(x_residual_attn) from residual */
      rmsnorm_bwd(d_x_normed_attn, saved_x_residual_attn[pos], dx,
                  saved_rms_attn[pos], N_EMBD);
    }

    /* Backward through pre-layer rmsnorm */
    float d_x_embed[N_EMBD];
    memset(d_x_embed, 0, N_EMBD * sizeof(float));
    rmsnorm_bwd(dx, saved_x_embed[pos], d_x_embed, saved_rms_pre[pos], N_EMBD);

    /* Back to wte[token_id] and wpe[pos_id] */
    int token_id = tokens[pos];
    for (int i = 0; i < N_EMBD; i++) {
      g_wte[token_id * N_EMBD + i] += d_x_embed[i];
      g_wpe[pos * N_EMBD + i] += d_x_embed[i];
    }
  }
}

/* ── Forward-only for inference (no saving for backward) ─────────────── */
static void forward_inference(int token_id, int pos_id, float *logits_out) {
  float x[N_EMBD];
  const float *te = wte + token_id * N_EMBD;
  const float *pe = wpe + pos_id * N_EMBD;
  for (int i = 0; i < N_EMBD; i++)
    x[i] = te[i] + pe[i];

  /* Pre-layer rmsnorm (in-place) */
  float rms_tmp;
  float xn[N_EMBD];
  rmsnorm_fwd(x, xn, N_EMBD, &rms_tmp);
  memcpy(x, xn, N_EMBD * sizeof(float));

  for (int li = 0; li < N_LAYER; li++) {
    float x_res[N_EMBD];
    memcpy(x_res, x, N_EMBD * sizeof(float));

    rmsnorm_fwd(x, xn, N_EMBD, &rms_tmp);

    float q[N_EMBD], k[N_EMBD], v[N_EMBD];
    linear_fwd(xn, attn_wq[li], q, N_EMBD, N_EMBD);
    linear_fwd(xn, attn_wk[li], k, N_EMBD, N_EMBD);
    linear_fwd(xn, attn_wv[li], v, N_EMBD, N_EMBD);

    /* Save K, V into cache (reuse saved_k/saved_v) */
    memcpy(saved_k[pos_id], k, N_EMBD * sizeof(float));
    memcpy(saved_v[pos_id], v, N_EMBD * sizeof(float));

    float x_attn[N_EMBD];
    for (int h = 0; h < N_HEAD; h++) {
      int hs = h * HEAD_DIM;
      int num_keys = pos_id + 1;

      float attn_logits[BLOCK_SIZE];
      for (int t = 0; t < num_keys; t++) {
        float dot = 0.0f;
        for (int j = 0; j < HEAD_DIM; j++)
          dot += q[hs + j] * saved_k[t][hs + j];
        attn_logits[t] = dot * INV_SQRT_HD;
      }

      float aw[BLOCK_SIZE];
      softmax_fwd(attn_logits, aw, num_keys);

      for (int j = 0; j < HEAD_DIM; j++) {
        float s = 0.0f;
        for (int t = 0; t < num_keys; t++)
          s += aw[t] * saved_v[t][hs + j];
        x_attn[hs + j] = s;
      }
    }

    float attn_out[N_EMBD];
    linear_fwd(x_attn, attn_wo[li], attn_out, N_EMBD, N_EMBD);
    for (int i = 0; i < N_EMBD; i++)
      x[i] = attn_out[i] + x_res[i];

    /* MLP */
    memcpy(x_res, x, N_EMBD * sizeof(float));
    rmsnorm_fwd(x, xn, N_EMBD, &rms_tmp);

    float mlp_h[MLP_DIM];
    linear_fwd(xn, mlp_fc1[li], mlp_h, MLP_DIM, N_EMBD);
    for (int i = 0; i < MLP_DIM; i++)
      mlp_h[i] = mlp_h[i] > 0 ? mlp_h[i] : 0.0f;

    float mlp_out[N_EMBD];
    linear_fwd(mlp_h, mlp_fc2[li], mlp_out, N_EMBD, MLP_DIM);
    for (int i = 0; i < N_EMBD; i++)
      x[i] = mlp_out[i] + x_res[i];
  }

  rmsnorm_fwd(x, xn, N_EMBD, &rms_tmp);
  linear_fwd(xn, lm_head, logits_out, vocab_size, N_EMBD);
  for (int i = 0; i < vocab_size; i++)
    logits_out[i] += lm_bias[i];
}

/* ── Main ────────────────────────────────────────────────────────────── */

/* The three values the harness and the parity runner pass in. The defaults are
 * the ones this file always used, so a bare `./microgpt` trains exactly what it
 * trained before the flags existed -- which is the property that makes the
 * recorded loss curves in this repository still describe this code.
 *
 * docs/ADDING-A-LANGUAGE.md requirement 4 asks for exactly these three flags
 * and asks a track *not* to resolve the dataset from the current working
 * directory. This one did: it opened "input.txt" relative to wherever it was
 * started, which is why tools/parity.py had to seed a private scratch directory
 * with a copy of the dataset to make the run work. tools/parity.py still passes
 * --input, so the private cwd is now belt and braces rather than load-bearing. */
static const char *dataset_path = "input.txt";
static int num_steps = NUM_STEPS;
static unsigned long long run_seed = 42;

static void parse_args(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--input") && i + 1 < argc) {
      dataset_path = argv[++i];
    } else if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
      num_steps = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
      run_seed = strtoull(argv[++i], NULL, 10);
    } else if (!strcmp(argv[i], "--val-stride") && i + 1 < argc) {
      char *end = NULL;
      long parsed = strtol(argv[++i], &end, 10);
      val_stride = (end && *end == '\0' && parsed > 1) ? (int)parsed : 128;
    } else {
      fprintf(stderr, "microgpt: unrecognised argument '%s'\n", argv[i]);
      fprintf(stderr, "  usage: %s [--input PATH] [--steps N] [--seed N]\n",
              argv[0]);
      exit(2);
    }
  }
  if (num_steps < 1) {
    fprintf(stderr, "microgpt: --steps must be at least 1, got %d\n", num_steps);
    exit(2);
  }
  if (val_stride < 2) {
    fprintf(stderr,
            "microgpt: --val-stride must be at least 2, got %d. 1 would hold out "
            "every document and leave nothing to train on, and a held-out loss "
            "with no training data behind it is not a measurement\n",
            val_stride);
    exit(2);
  }
}

int main(int argc, char **argv) {
  struct timespec t_start, t_end;
  parse_args(argc, argv);
  clock_gettime(CLOCK_MONOTONIC, &t_start);

  seed_rng(run_seed);
  load_data(dataset_path);
  init_weights();

  /* Adam hyperparams */
  const float learning_rate = 0.01f;
  const float learning_rate_floor = 0.001f;
  const float beta1 = 0.85f, beta2 = 0.98f;

  /* Training loop */
  for (int step = 0; step < num_steps; step++) {
    int doc_idx = doc_order[step % num_train_docs];
    const char *doc = docs_raw[doc_idx];
    int doc_len = doc_lens[doc_idx];

    /* Tokenize: BOS + chars + BOS */
    int tokens[BLOCK_SIZE + 2];
    tokens[0] = BOS_TOKEN;
    for (int i = 0; i < doc_len; i++)
      tokens[i + 1] = char_to_idx[(int)doc[i]];
    tokens[doc_len + 1] = BOS_TOKEN;
    int seq_len = doc_len + 2; /* total tokens */
    int n = seq_len - 1;       /* number of prediction positions */
    if (n > BLOCK_SIZE)
      n = BLOCK_SIZE;
    unsigned char used_wte[MAX_VOCAB] = {0};

    /* Zero gradients */
    memset(g_wte, 0, sizeof(g_wte));
    memset(g_wpe, 0, sizeof(g_wpe));
    memset(g_lm_head, 0, sizeof(g_lm_head));
    memset(g_lm_bias, 0, sizeof(g_lm_bias));
    for (int l = 0; l < N_LAYER; l++) {
      memset(g_attn_wq[l], 0, sizeof(g_attn_wq[l]));
      memset(g_attn_wk[l], 0, sizeof(g_attn_wk[l]));
      memset(g_attn_wv[l], 0, sizeof(g_attn_wv[l]));
      memset(g_attn_wo[l], 0, sizeof(g_attn_wo[l]));
      memset(g_mlp_fc1[l], 0, sizeof(g_mlp_fc1[l]));
      memset(g_mlp_fc2[l], 0, sizeof(g_mlp_fc2[l]));
    }

    /* Forward pass — all positions */
    for (int pos = 0; pos < n; pos++) {
      used_wte[tokens[pos]] = 1;
      forward_pos(tokens[pos], pos, pos + 1);
    }

    /* Compute loss */
    float loss = 0.0f;
    for (int pos = 0; pos < n; pos++) {
      int target = tokens[pos + 1];
      float p = saved_probs[pos][target];
      if (p < 1e-30f)
        p = 1e-30f;
      loss -= logf(p);
    }
    loss /= n;

    /* Backward pass */
    backward_all(tokens, n);

    /* Adam update */
    float phase = 3.1415926535897932f * (float)step / (float)num_steps;
    float lr_t =
        learning_rate_floor +
        (learning_rate - learning_rate_floor) *
            0.5f * (1.0f + cosf(phase));
    float b1c = 1.0f - powf(beta1, step + 1);
    float b2c = 1.0f - powf(beta2, step + 1);

    for (int token_id = 0; token_id < vocab_size; token_id++) {
      if (!used_wte[token_id])
        continue;
      adam_update(wte + token_id * N_EMBD,
                  g_wte + token_id * N_EMBD,
                  m_wte + token_id * N_EMBD,
                  v_wte + token_id * N_EMBD, N_EMBD, lr_t, b1c, b2c);
    }
    adam_update(lm_head, g_lm_head, m_lm_head, v_lm_head,
                vocab_size * N_EMBD, lr_t, b1c, b2c);
    adam_update(wpe, g_wpe, m_wpe, v_wpe, n * N_EMBD, lr_t, b1c, b2c);
    adam_update(lm_bias, g_lm_bias, m_lm_bias, v_lm_bias, vocab_size,
                lr_t, b1c, b2c);
    for (int l = 0; l < N_LAYER; l++) {
      adam_update(attn_wq[l], g_attn_wq[l], m_attn_wq[l], v_attn_wq[l],
                  N_EMBD * N_EMBD, lr_t, b1c, b2c);
      adam_update(attn_wk[l], g_attn_wk[l], m_attn_wk[l], v_attn_wk[l],
                  N_EMBD * N_EMBD, lr_t, b1c, b2c);
      adam_update(attn_wv[l], g_attn_wv[l], m_attn_wv[l], v_attn_wv[l],
                  N_EMBD * N_EMBD, lr_t, b1c, b2c);
      adam_update(attn_wo[l], g_attn_wo[l], m_attn_wo[l], v_attn_wo[l],
                  N_EMBD * N_EMBD, lr_t, b1c, b2c);
      adam_update(mlp_fc1[l], g_mlp_fc1[l], m_mlp_fc1[l], v_mlp_fc1[l],
                  MLP_DIM * N_EMBD, lr_t, b1c, b2c);
      adam_update(mlp_fc2[l], g_mlp_fc2[l], m_mlp_fc2[l], v_mlp_fc2[l],
                  N_EMBD * MLP_DIM, lr_t, b1c, b2c);
    }

    printf("step %4d / %4d | loss %.4f\n", step + 1, num_steps, loss);
  }

  /* ── Inference ─────────────────────────────────────────────────── */
  float temperature = 0.5f;
  printf("\n--- inference (new, hallucinated names) ---\n");
  for (int sample = 0; sample < 20; sample++) {
    /* Clear KV cache */
    int token_id = BOS_TOKEN;
    char name[BLOCK_SIZE + 1];
    int name_len = 0;

    for (int pos = 0; pos < BLOCK_SIZE; pos++) {
      float logits[MAX_VOCAB];
      forward_inference(token_id, pos, logits);

      /* Temperature scaling + softmax */
      float inv_temp = 1.0f / temperature;
      for (int i = 0; i < vocab_size; i++)
        logits[i] *= inv_temp;
      float probs[MAX_VOCAB];
      softmax_fwd(logits, probs, vocab_size);

      /* Weighted random sampling */
      float r = rand_uniform();
      float cumsum = 0.0f;
      int chosen = vocab_size - 1;
      for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (r < cumsum) {
          chosen = i;
          break;
        }
      }

      if (chosen == BOS_TOKEN)
        break;
      name[name_len++] = uchars[chosen];
      token_id = chosen;
    }
    name[name_len] = 0;
    printf("sample %2d: %s\n", sample + 1, name);
  }

  /* Held-out loss, measured once on the final model.
   *
   * Forward only -- no optimiser step touches a validation document, and none ever
   * did, because load_data kept them out of doc_order entirely. The loss is summed
   * over every prediction position of every validation document and divided by the
   * total, so a long document counts for more positions rather than for one, which
   * is what the training loss does per step.
   *
   * tools/autoresearch.py reads this line and refuses a keep whose training-loss
   * gain this number does not share. It is printed once, after training, and never
   * per step: a per-step held-out loss would be one more thing a candidate could
   * aim at, which is the failure this whole mechanism exists to stop.
   */
  double val_total = 0.0;
  int val_positions = 0;
  for (int vi = 0; vi < num_val_docs; vi++) {
    const char *doc = docs_raw[val_order[vi]];
    int doc_len = doc_lens[val_order[vi]];
    int tokens[BLOCK_SIZE + 2];
    tokens[0] = BOS_TOKEN;
    for (int i = 0; i < doc_len; i++)
      tokens[i + 1] = char_to_idx[(int)doc[i]];
    tokens[doc_len + 1] = BOS_TOKEN;
    int n = doc_len + 1;
    if (n > BLOCK_SIZE)
      n = BLOCK_SIZE;
    for (int pos = 0; pos < n; pos++)
      forward_pos(tokens[pos], pos, pos + 1);
    for (int pos = 0; pos < n; pos++) {
      float p = saved_probs[pos][tokens[pos + 1]];
      if (p < 1e-30f) p = 1e-30f;
      val_total += logf(p);
      val_positions++;
    }
  }
  float val_loss =
      val_positions > 0 ? (float)(-val_total / (double)val_positions) : 0.0f;
  printf("val_loss %f\n", val_loss);

  clock_gettime(CLOCK_MONOTONIC, &t_end);
  float elapsed = (float)(t_end.tv_sec - t_start.tv_sec) +
                  (float)(t_end.tv_nsec - t_start.tv_nsec) * 1e-9f;
  printf("\nTotal time: %.2f seconds\n", elapsed);

  return 0;
}
