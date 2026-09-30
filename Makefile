# =============================================================================
# learn-micro-gpt -- single entry point
#
# `make` with no arguments prints this help. Everything in this repository is
# reachable from here, on macOS, Linux, and the GitHub Actions ubuntu runner.
#
#   make help
#
# Overridable knobs (all of them are variables, so `make CC=gcc CFLAGS=-O0` works):
#   PYTHON   python interpreter           (default: python3)
#   CC       C compiler                   (default: cc, auto-detected)
#   GO       go binary                    (default: go)
#   CARGO    cargo binary                 (default: cargo)
#   NPM      npm binary                   (default: npm)
#   CFLAGS / CFLAGS_FAST / CFLAGS_NATIVE native build tuning
#   STEPS    default training step count  (default: 1000)
# =============================================================================

SHELL := /bin/sh
.DEFAULT_GOAL := help
.PHONY: help

PYTHON ?= python3
NPM    ?= npm
GO     ?= go
CARGO  ?= cargo

VIS        := visualizer
C_DIR      := implementations/c
TRACES     := traces
BENCHMARKS := benchmarks

# --- Toolchain detection ------------------------------------------------------
# The C ports accelerate on Apple Silicon and on x86 with AVX2, but they must
# *compile* everywhere. Portable defaults first, native tuning only when the
# host actually advertises it.
UNAME_S := $(shell uname -s)

# The C tracks build in one of three configurations, all of which produce the
# same loss curve (implementations/c/test_equivalence.sh checks that):
#
#   1. Accelerate + NEON    Apple Silicon fast path
#   2. scalar BLAS + NEON   Apple Silicon without the framework
#   3. scalar + scalar      anywhere, including the Linux CI runner
#
# The third is the default everywhere, because it is the one that always works.
# C_FAST=1 opts into the first on macOS.
ifeq ($(UNAME_S),Darwin)
  CC        ?= cc
  CFLAGS_NATIVE ?= -mcpu=apple-m1
  CFLAGS_FAST   ?= -Ofast -ffast-math -ffp-contract=fast -funroll-loops
  CFLAGS_PORTABLE ?= -O3
  ifeq ($(C_FAST),1)
    CFLAGS   ?= $(CFLAGS_FAST) $(CFLAGS_NATIVE) $(CFLAGS_BASE) -DMICROGPT_USE_ACCELERATE
    LDLIBS_PLATFORM ?= -framework Accelerate
  else
    CFLAGS   ?= $(CFLAGS_PORTABLE) $(CFLAGS_BASE)
  endif
else
  CC        ?= cc
  CFLAGS_NATIVE ?=
  CFLAGS_FAST   ?= -Ofast -ffast-math -ffp-contract=fast -funroll-loops
  CFLAGS_PORTABLE ?= -O3
  CFLAGS   ?= $(CFLAGS_PORTABLE) $(CFLAGS_BASE)
endif

CFLAGS_BASE    ?= -Wall -Wno-unused-function
CFLAGS_DEBUG   ?= -O0 -g -fsanitize=address,undefined
CFLAGS         ?= -O3 $(CFLAGS_NATIVE) $(CFLAGS_BASE)
CFLAGS_FAST    := $(CFLAGS_FAST) $(CFLAGS_NATIVE) $(CFLAGS_BASE)

# Every "tool is missing" case should say what to install, not just fail.
define require
	@command -v $(1) >/dev/null 2>&1 || { \
		echo "error: '$(1)' not found on PATH. $(2)" >&2; exit 127; }
endef

## ---------------------------------------------------------------- help ------

help:
	@echo "learn-micro-gpt -- a guided, visual reading of Karpathy's 199-line microgpt.py"
	@echo
	@echo "USAGE"
	@echo "  make <target>            see below; bare 'make' prints this help"
	@echo
	@echo "GETTING STARTED"
	@echo "  setup                    install toolchains and app dependencies"
	@echo "  dev                      run the visualizer on http://localhost:5173"
	@echo "  run                      train the Python reference (the 199 lines)"
	@echo "  run-c                    train the C port, micro config (parity track)"
	@echo "  run-scaled               train the C port, scaled config (benchmark track)"
	@echo
	@echo "OTHER LANGUAGE TRACKS"
	@echo "  run-go | run-rust | run-ts    run that track"
	@echo
	@echo "CONTENT AND TRUTH"
	@echo "  validate                 resolve concept anchors, validate content + schema"
	@echo "  test                     Python tool tests (unittest)"
	@echo "  test-web                 Vitest unit tests + Playwright e2e"
	@echo "  parity                   statistical loss-band check across all tracks"
	@echo "  verify-provenance        sha256 check on every pinned vendored file"
	@echo "  trace                    regenerate traces/ for every track"
	@echo "  trace-check              regenerate traces/ and fail if anything drifted"
	@echo "  bench                    regenerate benchmarks/results.json on this machine"
	@echo
	@echo "SHIP IT"
	@echo "  build                    production build of the visualizer"
	@echo "  preview                  serve the production build locally"
	@echo "  lint                     every linter this repo has"
	@echo "  c-test                   C build-config equivalence + gradient checks"
	@echo "  fmt                      every formatter, in place"
	@echo "  check                    lint + test + validate + parity (what CI runs)"
	@echo
	@echo "AUTORESEARCH  (a model improves the Rust track; see docs/AUTORESEARCH.md)"
	@echo "  autoresearch-rust        run the loop, committing and pushing every experiment"
	@echo "  autoresearch-c           the same, on the C track (TRACK=c)"
	@echo "  autoresearch-verify      what CI runs: rebuild, re-check, re-measure the loss axis"
	@echo "  autoresearch-render      regenerate autoresearch/results.json from the ledger"
	@echo
	@echo "HOUSEKEEPING"
	@echo "  provenance               regenerate docs/PROVENANCE.md from the pins"
	@echo "  clean                    remove build artifacts and generated data"
	@echo "  distclean                clean, plus node_modules and cargo target"
	@echo
	@echo "VARIABLES: PYTHON=$(PYTHON) CC=$(CC) GO=$(GO) CARGO=$(CARGO) NPM=$(NPM) STEPS=$(if $(STEPS),$(STEPS),1000)"
	@echo "            EXPERIMENTS=$(if $(EXPERIMENTS),$(EXPERIMENTS),1) RESEARCH_BRANCH=$(if $(RESEARCH_BRANCH),$(RESEARCH_BRANCH),main)"

## ------------------------------------------------------------- setup ------

setup: setup-python setup-web
	@echo "setup complete. Next: make dev   (visualizer)  or   make run   (train the reference)"

setup-python:
	@$(call require,$(PYTHON),Install Python 3.9+ (the tools are stdlib-only; no pip needed).)

setup-web:
	cd $(VIS) && $(NPM) install

## --------------------------------------------------------------- run ------

# The reference resolves input.txt from the CWD, so it runs in a throwaway
# directory seeded from data/. That keeps reference/microgpt.py byte-identical
# while still being reproducible. See docs/HOW-TO-READ.md.
run: run-python

run-python: | data/input.txt
	@$(call require,$(PYTHON),Install Python 3.9+.)
	@rm -rf .run && mkdir -p .run
	@cp data/input.txt .run/input.txt
	@echo "Running reference/microgpt.py (199 lines) in .run/ with STEPS=$(if $(STEPS),$(STEPS),1000)"
	@cd .run && $(PYTHON) $(CURDIR)/reference/microgpt.py

data/input.txt:
	@echo "error: data/input.txt is missing. It is committed; see data/README.md." >&2; exit 1

## ------------------------------------------------------- language tracks ----

run-c: $(C_DIR)/microgpt
	./$(C_DIR)/microgpt --input data/input.txt

run-scaled: $(C_DIR)/microgpt-scaled
	./$(C_DIR)/microgpt-scaled --input data/input.txt

$(C_DIR)/microgpt: $(C_DIR)/microgpt.c $(C_DIR)/microgpt_simd.h
	@$(call require,$(CC),Install clang or gcc.)
	$(CC) $(CFLAGS) -o $@ $< -lm $(LDLIBS_PLATFORM)

$(C_DIR)/microgpt-scaled: $(C_DIR)/microgpt-scaled.c $(C_DIR)/microgpt_simd.h
	@$(call require,$(CC),Install clang or gcc.)
	$(CC) $(CFLAGS) -o $@ $< -lm $(LDLIBS_PLATFORM)

run-go:
	@$(call require,$(GO),Install Go from https://go.dev/dl/)
	cd implementations/go && $(GO) run . --input ../../data/input.txt

run-rust:
	@$(call require,$(CARGO),Install Rust from https://rustup.rs)
	cd implementations/rust && $(CARGO) run --release -- --input ../../data/input.txt

run-ts:
	cd implementations/typescript && $(NPM) install && $(NPM) run start -- --input ../../data/input.txt

## ---------------------------------------------------- content and truth ----

validate:
	$(PYTHON) -m tools.gen_anchors
	$(PYTHON) -m tools.validate_concepts
	$(PYTHON) -m tools.render_content

test:
	$(PYTHON) -m unittest discover -s tools/tests -t . -v

test-web: install-web
	cd $(VIS) && $(NPM) run test:unit
	cd $(VIS) && $(NPM) run test:e2e

install-web:
	cd $(VIS) && $(NPM) install

verify-provenance:
	$(PYTHON) -m tools.check_provenance

provenance:
	$(PYTHON) -m tools.gen_provenance_doc

# Re-hash every derived entry and write the digests back to tools/provenance.py.
# Separate from `provenance` on purpose: that one is safe to run any time (it only
# re-renders a document from values already in the source), while this one edits
# the source. A derived file that changed legitimately has its digest refreshed
# here, in the same commit as the change, rather than by hand.
verify-provenance-refresh:
	$(PYTHON) -m tools.provenance --refresh-derived
	$(MAKE) provenance
	$(PYTHON) -m tools.check_provenance

trace:
	$(PYTHON) -m tools.trace --all

# The point of committed traces: they are regenerated and compared, so they can
# never quietly stop describing the code they came from.
#
# The comparison is done by tools/trace.py rather than by `git diff`, because
# `meta.json` carries a timestamp and a wall-clock time. Diffing the directory
# reports drift on every single run, and a check that always fails is one that
# everyone learns to ignore.
trace-check:
	$(PYTHON) -m tools.trace --check

parity:
	$(PYTHON) -m tools.parity

bench:
	$(PYTHON) -m tools.bench

## ---------------------------------------------------------- autoresearch ------

# The research loop. Four knobs, all overridable:
#
#   make autoresearch-rust                          # one Rust experiment, pushed
#   make autoresearch-rust EXPERIMENTS=5
#   make autoresearch-rust EXPERIMENTS=1 PUSH=0        # commit locally, do not push
#   make autoresearch-rust TRACK=go                    # or TRACK=typescript, TRACK=c
#   make autoresearch-rust RESEARCH_BRANCH=my-branch   # the branch it must be on
#
# TRACK picks which implementation to optimise. Each is measured against its own
# frozen comparator in its own candidate directory, so a Go steps-per-second says
# nothing about the Rust one and the two never compete on the same axis.
#
# C is also spelled out as its own target below, because it is the one track that
# could not be a candidate at all until docs/KNOWN-ISSUES.md issue 1 was fixed:
# its hand-written backward pass measured a gradient ratio of -0.12, which is
# outside the harness's own [0.5, 2.0] tripwire, so its baseline could never have
# been seeded. `TRACK=c` and `make autoresearch-c` are the same thing.
#
# It commits and pushes one commit per experiment, so the GitHub Page updates
# without anyone pressing anything. Note the cost of that: CI cancels
# in-progress runs on the same ref, so a burst of pushes leaves only the last one
# validated and deployed. EXPERIMENTS=5 is a sane first run; a hundred is a
# hundred full CI cycles.
#
# It needs OPENROUTER_API_KEY in the environment you run make from. The model and
# its reasoning setting are pinned in autoresearch/model.json.
autoresearch-rust:
	@$(call require,$(PYTHON),Install Python 3.9+.)
	@$(if $(shell git rev-parse --abbrev-ref HEAD),,\
		echo "error: this is not a git repository, so there is nowhere to commit to." >&2; exit 1;)
	$(PYTHON) -m tools.autoresearch seed --track $(if $(TRACK),$(TRACK),rust)
	$(PYTHON) -m tools.autoresearch loop \
		--track $(if $(TRACK),$(TRACK),rust) \
		--experiments $(if $(EXPERIMENTS),$(EXPERIMENTS),1) \
		--branch $(if $(RESEARCH_BRANCH),$(RESEARCH_BRANCH),main) \
		$(if $(PUSH_DELAY),--push-delay $(PUSH_DELAY),) \
		$(if $(filter 0,$(PUSH)),--no-push,)

autoresearch-c:
	@$(MAKE) autoresearch-rust TRACK=c

# What CI runs: no API key, no loop, no writes to the candidate. Rebuilds the
# candidate, checks the gradient probe, confirms the committed source still
# matches the run the site is quoting, and re-measures the loss axis. It
# deliberately does not check the speed axis and says so out loud.
autoresearch-verify:
	@$(call require,$(PYTHON),Install Python 3.9+.)
	$(PYTHON) -m tools.autoresearch verify

autoresearch-render:
	@$(call require,$(PYTHON),Install Python 3.9+.)
	$(PYTHON) -m tools.autoresearch render

## -------------------------------------------------------------- ship ------

dev:
	cd $(VIS) && $(NPM) run dev

build:
	cd $(VIS) && $(NPM) run build

preview:
	cd $(VIS) && $(NPM) run preview

# All three C build configurations must produce the same loss curve. A fallback
# that rounded differently would look to the parity gate like a different
# algorithm, so this is a correctness check, not a nicety.
c-test:
	cd $(C_DIR) && ./test_equivalence.sh
	cd $(C_DIR) && $(MAKE) check

lint:
	$(PYTHON) -m tools.lint

fmt:
	$(PYTHON) -m tools.fmt

check: lint test validate test-web c-test parity build
	$(PYTHON) -m tools.autoresearch verify

## ------------------------------------------------------ housekeeping ------

clean:
	rm -rf .run dist
	rm -f $(C_DIR)/microgpt $(C_DIR)/microgpt-scaled $(C_DIR)/*.o
	rm -f model.bin *.profraw *.profdata
	rm -rf $(VIS)/dist $(VIS)/playwright-report $(VIS)/test-results
	rm -rf implementations/rust/target autoresearch/candidate/target
	rm -f benchmarks/results.json.orig

distclean: clean
	rm -rf node_modules $(VIS)/node_modules implementations/typescript/node_modules
	rm -rf implementations/rust/target autoresearch/candidate/target
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

## ------------------------------------------------------------- meta ------

.PHONY: setup setup-python setup-web run run-python run-c run-scaled \
        run-go run-rust run-ts validate test test-web install-web \
        verify-provenance provenance trace trace-check parity bench c-test \
        autoresearch-rust autoresearch-c autoresearch-verify autoresearch-render \
        dev build preview lint fmt check clean distclean
