# Portable build for the C tracks.
#
# Two configurations, one source file:
#
#   fast      -DMICROGPT_USE_ACCELERATE -framework Accelerate   (Apple Silicon)
#   portable  no SIMD, no BLAS, no extra libraries             (anything)
#
# `make portable` is what the Linux CI runner uses, and it is the reason the
# parity gate can run there at all. See microgpt_simd.h.

CC       ?= cc
CFLAGS   ?= -O3 -Wall -Wextra -Wno-unused-function
LDLIBS   := -lm
UNAME_S  := $(shell uname -s)

# Reference config (parity track). The scaled config is a separate binary.
TRACKS   := microgpt microgpt-scaled

.PHONY: all fast portable test check clean help

all: portable

help:
	@echo "make portable   build with the portable fallbacks (works everywhere)"
	@echo "make fast       build with NEON + Apple Accelerate (Apple Silicon)"
	@echo "make test       build all four combinations and check they agree"
	@echo "make check      build and run the finite-difference gradient tests"
	@echo "make clean      remove binaries"

portable: $(TRACKS)

microgpt: microgpt.c microgpt_simd.h
	$(CC) $(CFLAGS) -o $@ microgpt.c $(LDLIBS)

microgpt-scaled: microgpt-scaled.c microgpt_simd.h
	$(CC) $(CFLAGS) -o $@ microgpt-scaled.c $(LDLIBS)

# The fast path, only where Accelerate exists.
fast:
ifeq ($(UNAME_S),Darwin)
	$(CC) -Ofast -mcpu=apple-m1 -ffast-math -ffp-contract=fast -funroll-loops \
	      -Wall -Wno-unused-function -DMICROGPT_USE_ACCELERATE \
	      -o microgpt microgpt.c $(LDLIBS) -framework Accelerate
else
	@echo "note: no Apple Accelerate on $(UNAME_S); falling back to NEON-only"
	$(CC) $(CFLAGS) -o microgpt microgpt.c $(LDLIBS)
endif

# The equivalence check. All four combinations of {Accelerate, scalar BLAS} x
# {NEON, scalar SIMD} must produce the same loss curve, because the parity gate
# compares loss values and a fallback that quietly rounds differently would look
# exactly like a different algorithm.
test:
	@./test_equivalence.sh

# The real tests. The gradient check is a *known-issue* lock, not a pass:
# see docs/KNOWN-ISSUES.md issue 1.
check: portable
	@$(CC) -O2 -Wall -Wno-unused-function -o test_gradients test_gradients.c $(LDLIBS)
	@./test_gradients ../../data/input.txt

clean:
	rm -f microgpt microgpt-scaled test_gradients /tmp/mg_*
