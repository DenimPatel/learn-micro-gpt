#!/bin/sh
# test_equivalence.sh — the three build configurations must agree.
#
# Why this exists: the parity gate in tools/parity.py compares *loss values* across
# implementations. If the portable SIMD fallback rounded differently from the
# NEON path, a runner that built the "slow" configuration would see a different
# loss curve and the gate would report it as a different algorithm. So the
# fallbacks have to be verified to be the same computation, not merely a
# plausible one.
#
# Three configurations, all of which must build and run:
#
#   1. Accelerate + NEON    the Apple Silicon fast path
#   2. scalar BLAS + NEON  Apple Silicon without the framework
#   3. scalar BLAS + scalar  anything, including the Linux CI runner
#
# A fourth combination (Accelerate + forced-scalar SIMD) cannot exist -- see the
# #error in microgpt_simd.h -- and is not tested.
#
# Tolerance: the configurations are compared on a *windowed mean* of the loss,
# not per step. That is deliberate. Two BLAS implementations may sum a dot
# product in different orders, so the last bits can differ, and over a few
# hundred steps that compounds. The parity gate itself uses the same reasoning:
# see tools/parity.py and docs/BENCHMARKS.md.

set -e
cd "$(dirname "$0")"

CC="${CC:-cc}"
DATA="${DATA:-../../data/input.txt}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [ ! -f "$DATA" ]; then
  echo "error: $DATA not found; run from implementations/c or set DATA=" >&2
  exit 1
fi
cp "$DATA" "$WORK/input.txt"

# Extract `step N | loss X` lines, then reduce to 10-step window means.
reduce() {
  awk '
    /^step/ {
      for (i = 1; i <= NF; i++) {
        if ($i == "loss") { loss = $(i + 1) }
        if ($i == "|") { step = $(i - 2) }
      }
      n++
      sum += loss
      if (n % 10 == 0) { printf "%.4f\n", sum / 10; sum = 0 }
    }
  ' "$1"
}

build_and_run() {
  name="$1"
  shift
  printf '  %-26s ' "$name"
  # shellcheck disable=SC2086
  if ! $CC -O3 -w "$@" -o "$WORK/mg" microgpt.c -lm 2>"$WORK/err.txt"; then
    echo "BUILD FAILED"
    sed 's/^/      /' "$WORK/err.txt" | head -6
    return 1
  fi
  ( cd "$WORK" && ./mg ) > "$WORK/out.txt" 2>/dev/null
  if grep -qi 'nan' "$WORK/out.txt"; then
    echo "RUN PRODUCED NaN"
    return 1
  fi
  reduce "$WORK/out.txt" > "$WORK/$name.txt"
  echo "$(wc -l < "$WORK/$name.txt" | tr -d ' ') window means"
}

echo "testing build equivalence (${CC})"
build_and_run accel+neon   -DMICROGPT_USE_ACCELERATE -framework Accelerate || exit 1
build_and_run scalar+neon  || exit 1
build_and_run scalar+scalar -DMICROGPT_NO_NEON || exit 1

# Tolerance: 1% *relative* on a 10-step window mean.
#
# Absolute tolerance was the first attempt and it was wrong. Measured on an
# M-series Mac, the fastest and slowest configurations differ by up to 0.0028 on
# window means near 2.5 -- 0.11% relative. That is not a bug: `cblas_sgemv` and a
# naive loop sum a 16-element dot product in different orders, the last bits
# differ, and 1,000 steps of a chaotic training process amplify that. The
# fallback is the same *computation*; it is not the same *rounding*.
#
# 1% relative is the right bar because it is far below the +/-10% band the parity
# gate uses (so a rounding difference can never be mistaken for a different
# algorithm) and far above the noise (so a real semantic difference in a
# fallback would still fail). The parity gate applies the same reasoning; see
# tools/parity.py and docs/BENCHMARKS.md.
TOLERANCE_PCT=1

echo
echo "comparing windowed means against the fast path (tolerance ${TOLERANCE_PCT}% relative):"
status=0
reference="$WORK/accel+neon.txt"
for other in "$WORK/scalar+neon.txt" "$WORK/scalar+scalar.txt"; do
  name="$(basename "$other" .txt)"
  if [ ! -f "$reference" ] || [ ! -f "$other" ]; then
    # No Accelerate on this platform (e.g. the Linux runner): the two
    # portable configurations still have to agree with each other.
    if [ "$name" = "scalar+scalar" ] && [ -f "$WORK/scalar+neon.txt" ]; then
      reference="$WORK/scalar+neon.txt"
    else
      continue
    fi
  fi
  if diff -q "$reference" "$other" >/dev/null 2>&1; then
    echo "  ok    $name is byte-identical to $(basename "$reference" .txt)"
  else
    worst=$(paste "$reference" "$other" | awk '
      { d = ($1 - $2); if (d < 0) d = -d; if ($1 != 0) { r = 100 * d / $1; if (r > m) m = r } }
      END { printf "%.4f", m + 0 }')
    if awk -v d="$worst" -v t="$TOLERANCE_PCT" 'BEGIN { exit !(d < t) }'; then
      echo "  ok    $name agrees within ${worst}% (max relative windowed difference)"
    else
      echo "  FAIL  $name differs by ${worst}%"
      status=1
    fi
  fi
done

if [ "$status" -eq 0 ]; then
  echo
  echo "equivalence ok: every build configuration learns the same thing"
else
  echo
  echo "equivalence FAILED" >&2
fi
exit "$status"
