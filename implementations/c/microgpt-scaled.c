/*
 * microgpt-scaled.c — C port at the *scaled* config, for the throughput track
 * float32 + ARM NEON SIMD + Apple Accelerate + branch-free hot paths
 *
 * n_embd 256 / n_head 8 / n_layer 4 / block 256, against the reference's
 * 16 / 4 / 1 / 16. Two configs on purpose: it shows the speed gap *widening*
 * with model size, which is the actual lesson, and it is why the C port is worth
 * reading in the first place. See docs/BENCHMARKS.md.
 *
 * This track is excluded from the parity gate -- it is a different model, so
 * there is nothing to compare its loss against -- and exists only to be timed.
 *
 * Build (Apple Silicon, the fast path):
 *   clang -Ofast -mcpu=apple-m1 -ffast-math -ffp-contract=fast -funroll-loops \
 *         -DMICROGPT_USE_ACCELERATE -o microgpt-scaled microgpt-scaled.c \
 *         -lm -framework Accelerate
 *
 * Build (anywhere else):
 *   cc -O3 -o microgpt-scaled microgpt-scaled.c -lm
 *
 * `microgpt_simd.h` supplies the same intrinsics and BLAS calls on every target:
 * the real ones on aarch64 with NEON, small portable fallbacks elsewhere. This
 * is the same portability work as microgpt.c -- see that header for why it was
 * necessary, and `make c-test` for the check that all configurations agree.
 */

#include "microgpt_simd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Hyperparameters (compile-time constants for loop unrolling) ──────── */
#define N_EMBD 256
#define N_HEAD 8
#define N_LAYER 4
#define BLOCK_SIZE 256
#define HEAD_DIM (N_EMBD / N_HEAD) /* 64 */
#define MLP_DIM (4 * N_EMBD)       /* 2048 */
#define MAX_DOCS 40000
#define MAX_DOC_LEN 64 //256
#define NUM_STEPS 5000
#define INV_SQRT_HD 0.125f /* 1/sqrt(HEAD_DIM=64) */
#define MAX_VOCAB 65

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

/* Weight storage — flat arrays ────────────────────────────────────── */
/* Sizes */
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

/* Adam moment buffers */
static float ALIGN128 m_wte[MAX_VOCAB * N_EMBD], v_wte[MAX_VOCAB * N_EMBD];
static float ALIGN128 m_wpe[BLOCK_SIZE * N_EMBD], v_wpe[BLOCK_SIZE * N_EMBD];
static float ALIGN128 m_lm_head[MAX_VOCAB * N_EMBD],
    v_lm_head[MAX_VOCAB * N_EMBD];
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

/* ── Dataset ─────────────────────────────────────────────────────────── */
static char docs_raw[MAX_DOCS][MAX_DOC_LEN];
static int doc_lens[MAX_DOCS];
static int doc_order[MAX_DOCS]; /* shuffle order */
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
static float saved_logits[BLOCK_SIZE][MAX_VOCAB];
static float saved_probs[BLOCK_SIZE][MAX_VOCAB];

/* Gradient scratch buffers */
static float ALIGN128 dx[N_EMBD];
static float ALIGN128 dtmp2[MLP_DIM];

/* ── NEON-vectorized helpers (N_EMBD=16 = 4x float32x4) ─────────────── */

static inline void linear_fwd(const float *__restrict__ x,
                              const float *__restrict__ w,
                              float *__restrict__ out, int nout, int nin) {
  /* Use Accelerate cblas for matrix-vector multiply: out = W * x */
  cblas_sgemv(CblasRowMajor, CblasNoTrans, nout, nin, 1.0f, w, nin, x, 1, 0.0f,
              out, 1);
}

static inline void linear_bwd_w(const float *__restrict__ dout,
                                const float *__restrict__ x,
                                float *__restrict__ gw, int nout, int nin) {
  /* gw += outer(dout, x) — use cblas_sger for rank-1 update */
  cblas_sger(CblasRowMajor, nout, nin, 1.0f, dout, 1, x, 1, gw, nin);
}

static inline void linear_bwd_x(const float *__restrict__ dout,
                                const float *__restrict__ w,
                                float *__restrict__ dx_out, int nout, int nin) {
  /* dx_out += W^T * dout */
  cblas_sgemv(CblasRowMajor, CblasTrans, nout, nin, 1.0f, w, nin, dout, 1, 1.0f,
              dx_out, 1);
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
  const float beta1 = 0.85f, beta2 = 0.99f;
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
    /* Add eps BEFORE sqrt to avoid NaN */
    param[i] -= lr_t * (mi * inv_b1c) / sqrtf(vi * inv_b2c + eps);
    grad[i] = 0.0f;
  }
}

/* ── Data loading ────────────────────────────────────────────────────── */
#define INPUT_FILE "input.txt"

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

  /* Init shuffle order */
  for (int i = 0; i < num_docs; i++)
    doc_order[i] = i;
  shuffle_docs(doc_order, num_docs);

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
  init_matrix(wte, vocab_size, N_EMBD, std);
  init_matrix(wpe, BLOCK_SIZE, N_EMBD, std);
  init_matrix(lm_head, vocab_size, N_EMBD, std);
  for (int l = 0; l < N_LAYER; l++) {
    init_matrix(attn_wq[l], N_EMBD, N_EMBD, std);
    init_matrix(attn_wk[l], N_EMBD, N_EMBD, std);
    init_matrix(attn_wv[l], N_EMBD, N_EMBD, std);
    init_matrix(attn_wo[l], N_EMBD, N_EMBD, std);
    init_matrix(mlp_fc1[l], MLP_DIM, N_EMBD, std);
    init_matrix(mlp_fc2[l], N_EMBD, MLP_DIM, std);
  }

  int total = vocab_size * N_EMBD + BLOCK_SIZE * N_EMBD + vocab_size * N_EMBD;
  for (int l = 0; l < N_LAYER; l++)
    total += 4 * N_EMBD * N_EMBD + MLP_DIM * N_EMBD + N_EMBD * MLP_DIM;
  printf("num params: %d\n", total);
}

/* ── Forward pass (single position, causal attention via KV cache) ─── */
static void forward_pos(int token_id, int pos_id, int seq_len) {
  float *x = saved_x_embed[pos_id];

  /* Token + position embedding — NEON add */
  const float *__restrict__ te = wte + token_id * N_EMBD;
  const float *__restrict__ pe = wpe + pos_id * N_EMBD;
  for (int i = 0; i < N_EMBD; i += 4) {
    float32x4_t a = vld1q_f32(te + i);
    float32x4_t b = vld1q_f32(pe + i);
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
  linear_fwd(x, lm_head, saved_logits[pos_id], vocab_size, N_EMBD);
  softmax_fwd(saved_logits[pos_id], saved_probs[pos_id], vocab_size);
}

/* Backward pass (all positions) ───────────────────────────────────── */
static void backward_all(const int *tokens, int n) {
  /* For each position, compute dL/d(logits) from cross-entropy loss
   * dL/d(logits_i) = probs_i - (i == target ? 1 : 0)   [scaled by 1/n]
   */
  float inv_n = 1.0f / n;

  /* We process positions in reverse for causal attention gradient accumulation
   */
  /* But since positions are somewhat independent (KV sharing is the coupling),
     we can process them in any order, accumulating gradients */

  for (int pos = n - 1; pos >= 0; pos--) {
    int target_id = tokens[pos + 1];

    /* dL/d(logits) = (probs - one_hot(target)) / n */
    float dlogits[MAX_VOCAB];
    for (int i = 0; i < vocab_size; i++)
      dlogits[i] = (saved_probs[pos][i] - (i == target_id ? 1.0f : 0.0f)) * inv_n;

    /* Backward through lm_head linear: logits = linear(x_final, lm_head) */
    memset(dx, 0, sizeof(float) * N_EMBD);
    linear_bwd_w(dlogits, saved_x_final[pos], g_lm_head, vocab_size, N_EMBD);
    linear_bwd_x(dlogits, lm_head, dx, vocab_size, N_EMBD);

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
      /* d_k and d_v accumulate across positions — we use saved_k/saved_v arrays
       */
      /* But since we process one query position at a time, we accumulate into
       * per-position grads */
      float d_k_accum[BLOCK_SIZE][N_EMBD]; /* we'll add to these */
      float d_v_accum[BLOCK_SIZE][N_EMBD];
      /* Initialize only what we need */
      int num_keys = pos + 1;
      for (int t = 0; t < num_keys; t++) {
        memset(d_k_accum[t], 0, N_EMBD * sizeof(float));
        memset(d_v_accum[t], 0, N_EMBD * sizeof(float));
      }

      for (int h = 0; h < N_HEAD; h++) {
        int hs = h * HEAD_DIM;

        /* head_out[j] = sum_t attn_w[t] * v[t][hs+j] */
        /* d_attn_w[t] += sum_j d_x_attn[hs+j] * v[t][hs+j] */
        /* d_v[t][hs+j] += attn_w[t] * d_x_attn[hs+j] */
        float d_attn_w[BLOCK_SIZE];
        for (int t = 0; t < num_keys; t++) {
          float dot = 0.0f;
          for (int j = 0; j < HEAD_DIM; j++) {
            d_v_accum[t][hs + j] +=
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
            d_k_accum[t][hs + j] += dl * saved_q[pos][hs + j];
          }
        }
      }

      /* Backward through Q, K, V linear projections */
      float d_x_normed_attn[N_EMBD];
      memset(d_x_normed_attn, 0, N_EMBD * sizeof(float));

      /* Q: q = linear(x_normed, wq) */
      linear_bwd_w(d_q, saved_x_normed_attn[pos], g_attn_wq[li],
                   N_EMBD, N_EMBD);
      linear_bwd_x(d_q, attn_wq[li], d_x_normed_attn, N_EMBD, N_EMBD);

      /* K: k[pos] = linear(x_normed[pos], wk) — only this position's K was
       * produced here */
      linear_bwd_w(d_k_accum[pos], saved_x_normed_attn[pos],
                   g_attn_wk[li], N_EMBD, N_EMBD);
      linear_bwd_x(d_k_accum[pos], attn_wk[li], d_x_normed_attn,
                   N_EMBD, N_EMBD);

      /* V: v[pos] = linear(x_normed[pos], wv) */
      linear_bwd_w(d_v_accum[pos], saved_x_normed_attn[pos],
                   g_attn_wv[li], N_EMBD, N_EMBD);
      linear_bwd_x(d_v_accum[pos], attn_wv[li], d_x_normed_attn,
                   N_EMBD, N_EMBD);

      /* But we also need to propagate d_k and d_v back to earlier positions'
       * x_normed, which were computed in earlier forward_pos calls. Since those
       * share the same weight matrices, we accumulate weight grads, and we need
       * to propagate dx back through those positions' rmsnorm -> residual ->
       * etc.
       *
       * HOWEVER, in the Python code the gradients from k[t] and v[t] for t <
       * pos DO flow back to earlier positions' embeddings. This is handled by
       * autograd. For manual backprop, we need to handle this.
       *
       * We accumulate the weight grads for K and V from all query positions,
       * but we ALSO need to push d_k[t] and d_v[t] back through position t's
       * computation. We'll handle this after processing all query positions.
       */
      /* For now, accumulate d_k and d_v for positions OTHER than current pos
       * into separate accumulators that we'll process later. */
      /* Actually, since each position t<pos already had its own forward saved,
       * and we need to push gradients back through those, let's accumulate into
       * global arrays and process them after all positions. */

      /* Store K/V grads for other positions — accumulate into the saved arrays
       */
      /* We use static arrays for this */
      static float dk_global[BLOCK_SIZE][N_EMBD];
      static float dv_global[BLOCK_SIZE][N_EMBD];
      static int dk_dv_initialized = 0;

      /* On first call per backward pass (pos == n-1), zero out */
      if (pos == n - 1 && !dk_dv_initialized) {
        for (int t = 0; t < n; t++) {
          memset(dk_global[t], 0, N_EMBD * sizeof(float));
          memset(dv_global[t], 0, N_EMBD * sizeof(float));
        }
        dk_dv_initialized = 1;
      }

      /* Accumulate for all positions */
      for (int t = 0; t < num_keys; t++) {
        if (t == pos)
          continue; /* already handled above */
        /* Weight grads for K and V at position t */
        linear_bwd_w(d_k_accum[t], saved_x_normed_attn[t],
                     g_attn_wk[li], N_EMBD, N_EMBD);
        linear_bwd_w(d_v_accum[t], saved_x_normed_attn[t],
                     g_attn_wv[li], N_EMBD, N_EMBD);
        /* Accumulate dx for position t */
        for (int i = 0; i < N_EMBD; i++) {
          dk_global[t][i] += d_k_accum[t][i];
          dv_global[t][i] += d_v_accum[t][i];
        }
      }

      /* If this is the last (pos==0) position being processed,
       * push all accumulated dk/dv grads back through earlier positions */
      if (pos == 0) {
        for (int t = 0; t < n; t++) {
          /* dk_global[t] needs to go back through wk -> x_normed_attn[t] ->
           * rmsnorm -> ... */
          float d_xn_kv[N_EMBD];
          memset(d_xn_kv, 0, N_EMBD * sizeof(float));
          linear_bwd_x(dk_global[t], attn_wk[li], d_xn_kv, N_EMBD,
                       N_EMBD);
          linear_bwd_x(dv_global[t], attn_wv[li], d_xn_kv, N_EMBD,
                       N_EMBD);

          /* Back through rmsnorm into x_residual_attn[t] */
          float d_x_res[N_EMBD];
          memset(d_x_res, 0, N_EMBD * sizeof(float));
          rmsnorm_bwd(d_xn_kv, saved_x_residual_attn[t], d_x_res,
                      saved_rms_attn[t], N_EMBD);

          /* This goes back to x before attention = output of previous layer or
           * embedding. For simplicity with 1 layer, this goes back to the
           * embedding. Propagate to wte and wpe. */
          /* Back through pre-layer rmsnorm */
          float d_x_pre_rms[N_EMBD];
          memset(d_x_pre_rms, 0, N_EMBD * sizeof(float));
          rmsnorm_bwd(d_x_res, saved_x_embed[t], d_x_pre_rms, saved_rms_pre[t],
                      N_EMBD);

          /* Back to wte and wpe */
          int tok_t = tokens[t];
          for (int i = 0; i < N_EMBD; i++) {
            g_wte[tok_t * N_EMBD + i] += d_x_pre_rms[i];
            g_wpe[t * N_EMBD + i] += d_x_pre_rms[i];
          }
        }
        dk_dv_initialized = 0; /* reset for next backward call */
      }

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

  linear_fwd(x, lm_head, logits_out, vocab_size, N_EMBD);
}

/* ── Model Save/Load ─────────────────────────────────────────────────── */
/* Binary format:
 * [magic:4] [vocab_size:4] [BOS_TOKEN:4] [num_docs:4]
 * [uchars:MAX_VOCAB] [char_to_idx:128*4]
 * [wte] [wpe] [lm_head]
 * [layer weights for each layer]
 */
#define MODEL_MAGIC 0x4D475054 /* "MGPT" */

static int save_model(const char *path) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "Cannot open %s for writing\n", path);
    return 0;
  }

  /* Write header */
  uint32_t magic = MODEL_MAGIC;
  fwrite(&magic, sizeof(uint32_t), 1, f);
  fwrite(&vocab_size, sizeof(int), 1, f);
  fwrite(&BOS_TOKEN, sizeof(int), 1, f);
  fwrite(&num_docs, sizeof(int), 1, f);

  /* Write vocabulary */
  fwrite(uchars, sizeof(char), MAX_VOCAB, f);
  fwrite(char_to_idx, sizeof(int), 128, f);

  /* Write weights */
  fwrite(wte, sizeof(float), vocab_size * N_EMBD, f);
  fwrite(wpe, sizeof(float), BLOCK_SIZE * N_EMBD, f);
  fwrite(lm_head, sizeof(float), vocab_size * N_EMBD, f);

  for (int l = 0; l < N_LAYER; l++) {
    fwrite(attn_wq[l], sizeof(float), N_EMBD * N_EMBD, f);
    fwrite(attn_wk[l], sizeof(float), N_EMBD * N_EMBD, f);
    fwrite(attn_wv[l], sizeof(float), N_EMBD * N_EMBD, f);
    fwrite(attn_wo[l], sizeof(float), N_EMBD * N_EMBD, f);
    fwrite(mlp_fc1[l], sizeof(float), MLP_DIM * N_EMBD, f);
    fwrite(mlp_fc2[l], sizeof(float), N_EMBD * MLP_DIM, f);
  }

  fclose(f);
  printf("Model saved to %s\n", path);
  return 1;
}

static int load_model(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Cannot open %s for reading\n", path);
    return 0;
  }

  /* Read and verify header */
  uint32_t magic;
  if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != MODEL_MAGIC) {
    fprintf(stderr, "Invalid model file: bad magic number\n");
    fclose(f);
    return 0;
  }

  int saved_vocab_size, saved_bos, saved_num_docs;
  fread(&saved_vocab_size, sizeof(int), 1, f);
  fread(&saved_bos, sizeof(int), 1, f);
  fread(&saved_num_docs, sizeof(int), 1, f);

  if (saved_vocab_size > MAX_VOCAB || saved_num_docs > MAX_DOCS) {
    fprintf(stderr, "Model file exceeds compile-time limits\n");
    fclose(f);
    return 0;
  }

  /* Update globals to match saved model */
  vocab_size = saved_vocab_size;
  BOS_TOKEN = saved_bos;
  num_docs = saved_num_docs;

  /* Read vocabulary */
  fread(uchars, sizeof(char), MAX_VOCAB, f);
  fread(char_to_idx, sizeof(int), 128, f);

  /* Read weights */
  if (fread(wte, sizeof(float), vocab_size * N_EMBD, f) != (size_t)(vocab_size * N_EMBD) ||
      fread(wpe, sizeof(float), BLOCK_SIZE * N_EMBD, f) != (size_t)(BLOCK_SIZE * N_EMBD) ||
      fread(lm_head, sizeof(float), vocab_size * N_EMBD, f) !=
          (size_t)(vocab_size * N_EMBD)) {
    fprintf(stderr, "Error reading weight matrices from model file\n");
    fclose(f);
    return 0;
  }

  for (int l = 0; l < N_LAYER; l++) {
    if (fread(attn_wq[l], sizeof(float), N_EMBD * N_EMBD, f) != (size_t)(N_EMBD * N_EMBD) ||
        fread(attn_wk[l], sizeof(float), N_EMBD * N_EMBD, f) != (size_t)(N_EMBD * N_EMBD) ||
        fread(attn_wv[l], sizeof(float), N_EMBD * N_EMBD, f) != (size_t)(N_EMBD * N_EMBD) ||
        fread(attn_wo[l], sizeof(float), N_EMBD * N_EMBD, f) != (size_t)(N_EMBD * N_EMBD) ||
        fread(mlp_fc1[l], sizeof(float), MLP_DIM * N_EMBD, f) != (size_t)(MLP_DIM * N_EMBD) ||
        fread(mlp_fc2[l], sizeof(float), N_EMBD * MLP_DIM, f) != (size_t)(N_EMBD * MLP_DIM)) {
      fprintf(stderr, "Error reading layer %d weights from model file\n", l);
      fclose(f);
      return 0;
    }
  }

  fclose(f);
  printf("Model loaded from %s\n", path);
  printf("vocab size: %d\n", vocab_size);
  printf("num docs: %d\n", num_docs);
  return 1;
}

/* ── Main ────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
  struct timespec t_start, t_end;
  clock_gettime(CLOCK_MONOTONIC, &t_start);

  const char *input_file = INPUT_FILE;
  const char *model_file = "model.bin";
  int mode = 0; /* 0=train+infer (default), 1=train only, 2=infer only */

  /* Parse arguments: [--train|--infer] [input_file] [model_file] */
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--train") == 0) {
      mode = 1;
    } else if (strcmp(argv[i], "--infer") == 0) {
      mode = 2;
    } else if (argv[i][0] != '-') {
      /* First non-flag argument is input file */
      input_file = argv[i];
      /* Second non-flag argument is model file */
      if (i + 1 < argc && argv[i + 1][0] != '-')
        model_file = argv[++i];
    }
  }

  seed_rng(42);

  /* Load or initialize model */
  if (mode == 2) {
    /* Inference only mode: load pre-trained model */
    if (!load_model(model_file)) {
      fprintf(stderr, "Failed to load model. Exiting.\n");
      return 1;
    }
  } else {
    /* Training mode: load data and initialize weights */
    load_data(input_file);
    init_weights();
  }

  /* Adam hyperparams */
  const float learning_rate = 0.0001f;
  const float beta1 = 0.9f, beta2 = 0.999f;

  /* Training loop — only if not inference-only mode */
  if (mode != 2) {
    for (int step = 0; step < NUM_STEPS; step++) {
      int doc_idx = doc_order[step % num_docs];
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

      /* Zero gradients */
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

      /* Forward pass — all positions */
      for (int pos = 0; pos < n; pos++)
        forward_pos(tokens[pos], pos, pos + 1);

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
      float lr_t = learning_rate * (1.0f - (float)step / NUM_STEPS);
      float b1c = 1.0f - powf(beta1, step + 1);
      float b2c = 1.0f - powf(beta2, step + 1);

      adam_update(wte, g_wte, m_wte, v_wte, vocab_size * N_EMBD, lr_t, b1c, b2c);
      adam_update(wpe, g_wpe, m_wpe, v_wpe, BLOCK_SIZE * N_EMBD, lr_t, b1c, b2c);
      adam_update(lm_head, g_lm_head, m_lm_head, v_lm_head, vocab_size * N_EMBD,
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

      printf("step %4d / %4d | loss %.4f\n", step + 1, NUM_STEPS, loss);
    }

    /* Save model after training (unless --train was explicitly specified without
     * wanting inference) */
    if (mode == 0 || mode == 1) {
      save_model(model_file);
    }
  }

  /* ── Inference ─────────────────────────────────────────────────── */
  float temperature = 0.5f;
  printf("\n--- inference (new, hallucinated names) ---\n");
  for (int sample = 0; sample < 20; sample++) {
    /* Clear KV cache */
    int token_id = BOS_TOKEN;
    char name[2048];
    int name_len = 0;

    for (int pos = 0; pos < 2048; pos++) {
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

  clock_gettime(CLOCK_MONOTONIC, &t_end);
  float elapsed = (float)(t_end.tv_sec - t_start.tv_sec) +
                  (float)(t_end.tv_nsec - t_start.tv_nsec) * 1e-9f;
  printf("\nTotal time: %.2f seconds\n", elapsed);

  return 0;
}
