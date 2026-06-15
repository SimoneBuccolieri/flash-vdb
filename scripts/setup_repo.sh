#!/usr/bin/env bash
# ============================================================
# setup_repo.sh — FlashVDB Repository Scaffold
# Run once from the repository root to create the full layout.
# Usage: bash scripts/setup_repo.sh
# ============================================================

set -euo pipefail

echo "[flashvdb] Scaffolding repository layout..."

# ── Core library ─────────────────────────────────────────────
mkdir -p include/flashvdb
touch include/flashvdb/flashvdb.hpp          # Public API header (single-include)
touch include/flashvdb/simd.hpp             # SIMD dispatch helpers (NEON / AVX2)
touch include/flashvdb/index.hpp            # Graph index structures
touch include/flashvdb/quantize.hpp         # Scalar quantization (SQ8) primitives
touch include/flashvdb/io.hpp               # Binary persistence (dump / load)
touch include/flashvdb/distance.hpp         # Distance / similarity functions

# ── Implementation sources ────────────────────────────────────
mkdir -p src
touch src/index.cpp
touch src/quantize.cpp
touch src/io.cpp
touch src/distance.cpp

# ── Tests ────────────────────────────────────────────────────
mkdir -p tests
touch tests/test_insert.cpp
touch tests/test_search.cpp
touch tests/test_io.cpp
touch tests/test_simd.cpp
touch tests/test_quantize.cpp

# ── Benchmarks ───────────────────────────────────────────────
mkdir -p bench
touch bench/bench_search.cpp
touch bench/bench_io.cpp

# ── Documentation ────────────────────────────────────────────
mkdir -p docs
touch docs/architecture.md
touch docs/api_reference.md
touch docs/benchmarks.md

# ── CI / tooling ─────────────────────────────────────────────
mkdir -p .github/workflows
touch .github/workflows/ci.yml

# ── Build artifacts (git-ignored) ────────────────────────────
mkdir -p build

echo "[flashvdb] Done. Directory layout:"
find . \
    -not -path './.git/*' \
    -not -path './venv/*' \
    -not -path './__pycache__/*' \
    -not -path './build/*' \
    | sort
