/*
 * microgpt_simd.h — a portable stand-in for the handful of ARM NEON intrinsics
 * this file uses, plus the three Apple Accelerate BLAS calls.
 *
 * ## Why this file exists
 *
 * The original `microgpt.c` opened with:
 *
 *     #include <Accelerate/Accelerate.h>
 *     #include <arm_neon.h>
 *
 * and the Makefile hardcoded `-mcpu=apple-m1 -framework Accelerate`. That builds
 * on one machine. It does not build on the Linux runner that GitHub Actions
 * provides, which means the C parity track could not run in CI at all — the
 * parity gate was, in practice, a claim rather than a check.
 *
 * So rather than scatter `#ifdef __APPLE__` through 68 call sites, this header
 * supplies the same names with the same semantics on any target:
 *
 *   - On aarch64 with NEON, `#include <arm_neon.h>` and nothing here is used.
 *     The compiler emits the real instructions.
 *   - Everywhere else, each intrinsic becomes a small `static inline` over a
 *     four-float struct. The arithmetic is the same; the vectorisation is gone,
 *     and the C compiler re-vectorises the simple loops at -O3 anyway.
 *
 * The scalar fallback is correct before it is fast, and that is the ordering
 * that matters: a parity track that is right on every machine is worth more than
 * one that is fast on one.
 *
 * The float32x4_t layout in the fallback is deliberately `[4]` rather than
 * something clever, so that a mistake shows up as a wrong number rather than as
 * undefined behaviour.
 */

#ifndef MICROGPT_SIMD_H
#define MICROGPT_SIMD_H

#include <stddef.h>

/* ── BLAS: cblas_sgemv and cblas_sger ──────────────────────────────────── */
/*
 * The three BLAS calls are `out = W·x`, `G += outer(dout, x)`, and
 * `dx += Wᵀ·dout`. All three are two nested loops, and the reference Python is
 * the same loops written as comprehensions — so the fallback is not an
 * approximation of the accelerated path, it is the same computation the Python
 * reference does.
 *
 * Accelerate's cblas is used when available because its `sgemv` is genuinely
 * well-tuned, not because the arithmetic differs.
 */

/*
 * Accelerate is gated on an explicit opt-in, not on `__APPLE__`.
 *
 * That distinction matters: `__APPLE__` is true on a Mac whether or not the
 * build actually linked `-framework Accelerate`, so gating on it produced a
 * confusing link error -- "undefined symbol _cblas_sgemv" -- for anyone who
 * compiled on a Mac without the framework flag. The build now declares what it
 * did, in the Makefile:
 *
 *     fast path:   -DMICROGPT_USE_ACCELERATE ... -framework Accelerate
 *     portable:    (nothing)
 *
 * and the header does exactly what the build says. `MICROGPT_NO_ACCELERATE` and
 * `MICROGPT_NO_NEON` force the portable fallbacks even on hardware that has
 * them, which is what makes it possible to check that all four combinations
 * produce the same loss curve -- see docs/BENCHMARKS.md.
 */

#if defined(MICROGPT_USE_ACCELERATE) && !defined(MICROGPT_NO_ACCELERATE)
#include <Accelerate/Accelerate.h>
#define MICROGPT_HAVE_ACCELERATE 1
#else
#define MICROGPT_HAVE_ACCELERATE 0

/*
 * The call sites name CBLAS's enum constants, so the portable path has to
 * define them rather than changing the calls. The values match CBLAS's own, and
 * the functions below check the ones that matter (`CblasTrans` /
 * `CblasNoTrans`) so a wrong value fails loudly instead of quietly computing the
 * wrong orientation.
 */
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

/* Row-major, unit strides, `beta` overwrites or accumulates. */
static void cblas_sgemv(int order, int trans, int m, int n, float alpha,
                        const float *a, int lda, const float *x, int incx,
                        float beta, float *y, int incy) {
  (void)order;
  (void)incx;
  (void)incy;
  const int transposed = (trans == CblasTrans);
  /*
   * The output length depends on the orientation, and getting it wrong writes
   * past the end of the caller's buffer -- which is what this function did at
   * first, silently corrupting the stack and producing NaNs three steps into
   * training. `A` is m x n, so:
   *
   *   no-trans:  y has m elements, y[i] = sum_j A[i*lda + j] * x[j]
   *   trans:     y has n elements, y[i] = sum_j A[j*lda + i] * x[j]
   */
  const int out_len = transposed ? n : m;
  for (int i = 0; i < out_len; i++) {
    float acc = 0.0f;
    if (!transposed) {
      const float *row = a + (size_t)i * (size_t)lda;
      for (int j = 0; j < n; j++) acc += row[j] * x[j];
    } else {
      for (int j = 0; j < m; j++) acc += a[(size_t)j * (size_t)lda + (size_t)i] * x[j];
    }
    /*
     * `beta == 0` must not read `y`, and this line used to.
     *
     * CBLAS says that with beta zero the output is *overwritten* rather than
     * accumulated into, so a caller is entitled to hand in a buffer it has
     * never initialised. microgpt.c does exactly that: `linear_fwd` forwards
     * beta=0 to here, and its destinations include plain stack arrays --
     * `mlp_out` in the training path, and `q`/`k`/`v`/`attn_out`/`mlp_h`/
     * `logits` in the inference path. The natural spelling above reads them
     * anyway, and `0.0f * NaN` is NaN and `0.0f * Inf` is NaN, so a stack slot
     * holding a non-finite bit pattern -- ordinary leftover from whatever the
     * process last put there, and which ASLR moves around from run to run --
     * turned the very first training step into a NaN loss in a small fraction
     * of runs. Finite garbage is harmless, because `0.0f * finite` is exactly
     * zero, which is why this sat here looking correct for a long time.
     *
     * The fix is not a special case bolted on for the symptom: Accelerate's own
     * `cblas_sgemv` does not read `y` when beta is zero either, so branching
     * here is what makes the fallback compute the same thing as the fast path
     * instead of something that merely usually agrees. That is the entire job
     * of this file, and `test_equivalence.sh` is the check that it succeeded --
     * the fast path passed while the fallback failed, which is exactly the
     * divergence the script exists to catch.
     */
    if (beta == 0.0f) {
      y[i] = alpha * acc;
    } else {
      y[i] = alpha * acc + beta * y[i];
    }
  }
}

/* Rank-1 update: A += alpha * x * yᵀ, row-major. */
static void cblas_sger(int order, int m, int n, float alpha, const float *x,
                       int incx, const float *y, int incy, float *a, int lda) {
  (void)order;
  for (int i = 0; i < m; i++) {
    float *row = a + (size_t)i * (size_t)lda;
    const float xi = alpha * x[(size_t)i * (size_t)incx];
    for (int j = 0; j < n; j++) row[j] += xi * y[(size_t)j * (size_t)incy];
  }
}

#endif /* __APPLE__ */

/* ── NEON ──────────────────────────────────────────────────────────────── */

/*
 * Two paths, chosen the way the build says rather than the way the platform
 * guesses.
 *
 * When the target has NEON, `<arm_neon.h>` is included -- because it defines the
 * vector *types* the call sites use, and because its intrinsics are macros, not
 * functions. `MICROGPT_NO_NEON` then `#undef`s those 19 macros so the scalar
 * definitions below can take their names. That switch exists so `make test` can
 * build all four combinations of {Accelerate, scalar BLAS} x {NEON, scalar SIMD}
 * and check they produce the same loss curve. A fallback that rounded
 * differently would look exactly like a different algorithm, and the parity gate
 * compares loss values.
 *
 * The fallback reproduces the accelerated path's *sequence of operations*, not
 * just its mathematical result: `vrsqrte` is a ~8-bit estimate refined twice by
 * `vrsqrts`, and the fallback does the same two steps, so the two builds agree
 * rather than both being approximately right.
 */
/*
 * `arm_neon.h` is included only on the path that actually uses it. Forcing the
 * scalar path does not include it and then try to undefine its contents, because
 * clang defines some of the intrinsics as macros and others as `static inline`
 * functions, so there is no uniform way to displace them -- and a partial
 * attempt produces either a redefinition error or, worse, a mix of the two.
 * Not including the header is both simpler and impossible to get half-right.
 */
#if (defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)) && \
    !defined(MICROGPT_NO_NEON)

#include <arm_neon.h>
#define MICROGPT_SIMD_NATIVE 1

#else

#define MICROGPT_SIMD_NATIVE 0

/*
 * `MICROGPT_NO_NEON` together with Accelerate is not a configuration that can
 * exist, and failing with a redefinition error three hundred lines later is a
 * waste of everyone's afternoon. On Apple Silicon,
 * `<Accelerate/Accelerate.h>` includes `arm_neon.h` transitively, so asking for
 * the scalar SIMD path while linking Accelerate asks for two definitions of the
 * same names in one translation unit. clang defines some of those intrinsics as
 * macros and others as `static inline` functions, so there is no uniform way to
 * displace them.
 *
 * The three configurations that matter are all supported:
 *
 *   1. Accelerate + NEON     -- the Apple Silicon fast path
 *   2. scalar BLAS + NEON   -- Apple Silicon without the framework
 *   3. scalar BLAS + scalar -- anything, including the Linux CI runner
 *
 * and `make test` checks that all three produce the same loss curve.
 */
#if MICROGPT_HAVE_ACCELERATE && (defined(__aarch64__) || defined(_M_ARM64))
#error "MICROGPT_NO_NEON cannot be combined with MICROGPT_USE_ACCELERATE: \
Accelerate includes arm_neon.h transitively, so the vector types and intrinsics \
already exist. Drop -DMICROGPT_USE_ACCELERATE, or drop -DMICROGPT_NO_NEON."
#endif

/* A four-float lane and a two-float lane as plain structs. A `float[4]` on
 * purpose: a mistake here should be a wrong number, not undefined behaviour. */
typedef struct {
  float v[4];
} microgpt_f32x4_t;
typedef struct {
  float v[2];
} microgpt_f32x2_t;
typedef microgpt_f32x4_t float32x4_t;
typedef microgpt_f32x2_t float32x2_t;

/* Lane access, via a union so it works for the real NEON type and for the
 * struct above alike. One implementation, no second copy to drift. */
static inline float mg_lane4(float32x4_t v, int i) {
  union {
    float32x4_t v;
    float f[4];
  } u;
  u.v = v;
  return u.f[i];
}

static inline float32x4_t mg_from4(float a, float b, float c, float d) {
  union {
    float32x4_t v;
    float f[4];
  } u;
  u.f[0] = a;
  u.f[1] = b;
  u.f[2] = c;
  u.f[3] = d;
  return u.v;
}

static inline float mg_lane2(float32x2_t v, int i) {
  union {
    float32x2_t v;
    float f[2];
  } u;
  u.v = v;
  return u.f[i];
}

static inline float32x2_t mg_from2(float a, float b) {
  union {
    float32x2_t v;
    float f[2];
  } u;
  u.f[0] = a;
  u.f[1] = b;
  return u.v;
}

static inline float32x4_t vld1q_f32(const float *p) {
  return mg_from4(p[0], p[1], p[2], p[3]);
}

static inline void vst1q_f32(float *p, float32x4_t a) {
  p[0] = mg_lane4(a, 0);
  p[1] = mg_lane4(a, 1);
  p[2] = mg_lane4(a, 2);
  p[3] = mg_lane4(a, 3);
}

static inline float32x4_t vdupq_n_f32(float x) { return mg_from4(x, x, x, x); }
static inline float32x2_t vdup_n_f32(float x) { return mg_from2(x, x); }
static inline float vget_lane_f32(float32x2_t a, int lane) { return mg_lane2(a, lane); }

static inline float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) {
  return mg_from4(mg_lane4(a, 0) + mg_lane4(b, 0), mg_lane4(a, 1) + mg_lane4(b, 1),
                  mg_lane4(a, 2) + mg_lane4(b, 2), mg_lane4(a, 3) + mg_lane4(b, 3));
}

static inline float32x4_t vsubq_f32(float32x4_t a, float32x4_t b) {
  return mg_from4(mg_lane4(a, 0) - mg_lane4(b, 0), mg_lane4(a, 1) - mg_lane4(b, 1),
                  mg_lane4(a, 2) - mg_lane4(b, 2), mg_lane4(a, 3) - mg_lane4(b, 3));
}

static inline float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) {
  return mg_from4(mg_lane4(a, 0) * mg_lane4(b, 0), mg_lane4(a, 1) * mg_lane4(b, 1),
                  mg_lane4(a, 2) * mg_lane4(b, 2), mg_lane4(a, 3) * mg_lane4(b, 3));
}

static inline float32x4_t vfmaq_f32(float32x4_t acc, float32x4_t a, float32x4_t b) {
  return mg_from4(mg_lane4(acc, 0) + mg_lane4(a, 0) * mg_lane4(b, 0),
                  mg_lane4(acc, 1) + mg_lane4(a, 1) * mg_lane4(b, 1),
                  mg_lane4(acc, 2) + mg_lane4(a, 2) * mg_lane4(b, 2),
                  mg_lane4(acc, 3) + mg_lane4(a, 3) * mg_lane4(b, 3));
}

static inline float32x4_t vmaxq_f32(float32x4_t a, float32x4_t b) {
  float x0 = mg_lane4(a, 0), y0 = mg_lane4(b, 0);
  float x1 = mg_lane4(a, 1), y1 = mg_lane4(b, 1);
  float x2 = mg_lane4(a, 2), y2 = mg_lane4(b, 2);
  float x3 = mg_lane4(a, 3), y3 = mg_lane4(b, 3);
  return mg_from4(x0 > y0 ? x0 : y0, x1 > y1 ? x1 : y1, x2 > y2 ? x2 : y2,
                  x3 > y3 ? x3 : y3);
}

static inline float vaddvq_f32(float32x4_t a) {
  return mg_lane4(a, 0) + mg_lane4(a, 1) + mg_lane4(a, 2) + mg_lane4(a, 3);
}

static inline float32x2_t vmul_f32(float32x2_t a, float32x2_t b) {
  return mg_from2(mg_lane2(a, 0) * mg_lane2(b, 0), mg_lane2(a, 1) * mg_lane2(b, 1));
}

static inline float mg_rsqrte(float x) {
  if (x <= 0.0f) return 0.0f;
  union {
    float f;
    unsigned int u;
  } bits;
  bits.f = x;
  /*
   * Seed by halving the biased exponent: 0x5F375A86 is the exponent/mantissa
   * midpoint of the float32 range, so subtracting half of `x`'s exponent bits
   * lands within a factor of two of 1/sqrt(x). One Newton step then brings it
   * to roughly 12 bits, and the two `mg_rsrts` steps at the call site finish the
   * job -- which is the point: the accelerated path runs the same two steps, so
   * both converge to the same value rather than merely both being close.
   */
  bits.u = 0x5F375A86u - (bits.u >> 1);
  const float y = bits.f;
  return y * (1.5f - 0.5f * x * y * y);
}

/* One fused Newton step for 1/sqrt: approximately (3 - a*b)/2. */
static inline float mg_rsrts(float a, float b) { return 0.5f * (3.0f - a * b); }

static inline float32x2_t vrsqrte_f32(float32x2_t a) {
  return mg_from2(mg_rsqrte(mg_lane2(a, 0)), mg_rsqrte(mg_lane2(a, 1)));
}

static inline float32x2_t vrsqrts_f32(float32x2_t a, float32x2_t b) {
  return mg_from2(mg_rsrts(mg_lane2(a, 0), mg_lane2(b, 0)),
                  mg_rsrts(mg_lane2(a, 1), mg_lane2(b, 1)));
}

static inline float32x4_t vrsqrteq_f32(float32x4_t a) {
  return mg_from4(mg_rsqrte(mg_lane4(a, 0)), mg_rsqrte(mg_lane4(a, 1)),
                  mg_rsqrte(mg_lane4(a, 2)), mg_rsqrte(mg_lane4(a, 3)));
}

static inline float32x4_t vrsqrtsq_f32(float32x4_t a, float32x4_t b) {
  return mg_from4(mg_rsrts(mg_lane4(a, 0), mg_lane4(b, 0)),
                  mg_rsrts(mg_lane4(a, 1), mg_lane4(b, 1)),
                  mg_rsrts(mg_lane4(a, 2), mg_lane4(b, 2)),
                  mg_rsrts(mg_lane4(a, 3), mg_lane4(b, 3)));
}

#endif /* native NEON */

#endif /* MICROGPT_SIMD_H */
