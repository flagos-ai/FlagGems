#!/usr/bin/env bash

# =============================================================================
# FlagGems default-vs-expanded tuning benchmark workflow
# =============================================================================
#
# Purpose
# -------
# Run the complete cold-tuning and hot-latency benchmark, aggregate the two
# methods by shape, and keep every reproducibility artifact below:
#
#   <project_root>/flagtune-benchmark-output/
#
# Method semantics
# ----------------
# The similarly named FlagGems and FlagTree mechanisms must not be conflated:
#
#   default
#     USE_FLAGTUNE is removed from the child environment. FlagGems keeps the
#     kernel's default config list, while the configured LibTuner "flagtune"
#     policy runs the FlagTree XGBoost+GA proposer.
#
#   expanded
#     USE_FLAGTUNE=1 is set only for this method. FlagGems replaces the config
#     list with the expanded parameter space and bypasses the proposer, so the
#     LibTuner default policy exhaustively benchmarks the expanded configs.
#
# Workflow
# --------
# 1. Resolve this file's project root, independent of the caller's cwd.
# 2. Select a model shape file (or an explicit --shape-yaml).
# 3. Isolate default and expanded FlagGems caches below the output root.
# 4. Run cold first-call tuning on the requested visible GPUs.
# 5. Reuse each populated cache for the hot pytest latency pass.
# 6. Produce the raw CSV/JSONL, Markdown report, manifest, and the reference
#    one-row-per-shape tuning_latency_compare_by_shape.csv.
# 7. Check that every expected aggregate artifact is present and non-empty.
#
# Output layout
# -------------
#   flagtune-benchmark-output/
#   |-- cache/<run-name>/
#   |   |-- <run-name>_default/
#   |   `-- <run-name>_expanded/
#   `-- <run-name>/
#       |-- invocation.txt
#       |-- run.log
#       |-- manifest.json
#       |-- selected_shapes.yaml
#       |-- cold_default.log / cold_expanded.log
#       |-- hot_default.log / hot_expanded.log
#       |-- tuning_latency_compare.csv
#       |-- tuning_latency_compare.jsonl
#       |-- tuning_latency_compare.md
#       `-- tuning_latency_compare_by_shape.csv
#
# Requirements
# ------------
# - Run from an environment with torch, pytest, PyYAML, FlagGems, and a
#   FlagTree-enabled Triton import. The wrapper prepends <project_root>/src to
#   PYTHONPATH and verifies CUDA plus triton.flagtune before starting.
# - CUDA_VISIBLE_DEVICES controls which physical GPUs are exposed. --parallel
#   counts logical devices inside that visible set and must not exceed it.
# - The default model expects the sibling workspace path
#   ../autotune/FlagGems/FlagTune/shape-config/. Use --shape-yaml elsewhere.
# - A full expanded-space cold pass is intentionally expensive and can create
#   a large cache. The wrapper never deletes an existing result directory.
#
# Examples
# --------
# Full default-model run on four selected GPUs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     ./benchmark/flagtune/run_tuning_latency_compare.sh --parallel 4
#
# One-shape smoke run with an explicit Python environment:
#   ./benchmark/flagtune/run_tuning_latency_compare.sh \
#     --python /path/to/env/bin/python --parallel 1 --max-shapes 1
#
# Custom shape file and stable run name:
#   ./benchmark/flagtune/run_tuning_latency_compare.sh \
#     --shape-yaml /path/to/shapes.yaml --run-name my_full_run
#
# Use --help for the complete option list. Environment variables such as
# PYTHON_BIN, PARALLEL, DTYPES, HOT_WARMUP, HOT_ITER, RUN_NAME, OUTPUT_ROOT,
# SHAPE_YAML, and CUDA_VISIBLE_DEVICES may be used by automation.
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd -P)

PYTHON_BIN=${PYTHON_BIN:-${PYTHON:-python3}}
MODEL=${MODEL:-Qwen3.5-397B-A17B-p1024d1024}
SHAPE_YAML=${SHAPE_YAML:-}
PARALLEL=${PARALLEL:-}
DTYPES=${DTYPES:-bfloat16}
HOT_WARMUP=${HOT_WARMUP:-1000}
HOT_ITER=${HOT_ITER:-100}
START_SHAPE=${START_SHAPE:-0}
MAX_SHAPES=${MAX_SHAPES:-}
RUN_NAME=${RUN_NAME:-tuning_latency_compare_$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/flagtune-benchmark-output}
FAIL_FAST=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Run the complete FlagGems tuning/latency comparison and aggregate its results.

The two benchmark methods are fixed by this workflow:
  default   USE_FLAGTUNE is unset. Use default configs with XGBoost+GA.
  expanded  USE_FLAGTUNE=1. Exhaustively benchmark the expanded config space.

Usage:
  benchmark/flagtune/run_tuning_latency_compare.sh [options]

Options:
  --shape-yaml PATH   FlagTune-style shape YAML. Overrides --model.
  --model NAME        Model under the sibling shape-config directory.
  --parallel N        Number of visible GPUs/workers. Defaults to all visible GPUs.
  --dtypes LIST       Dtypes passed to the benchmark (default: bfloat16).
  --hot-warmup N      Hot-cache benchmark warmup count (default: 1000).
  --hot-iter N        Hot-cache benchmark iteration count (default: 100).
  --start-shape N     Start offset in the selected shape list (default: 0).
  --max-shapes N      Limit selected shapes; omitted means all shapes.
  --run-name NAME     Result directory name under flagtune-benchmark-output/.
  --output-root PATH  Override <project_root>/flagtune-benchmark-output/.
  --python PATH       Python executable from the benchmark environment.
  --fail-fast         Stop after the first failed benchmark pass.
  --dry-run           Print the resolved command without running it.
  -h, --help          Show this help.

Environment variables with the corresponding uppercase names can also set
defaults, for example PARALLEL=4, PYTHON_BIN=python, or
CUDA_VISIBLE_DEVICES=0,1,2,3.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --shape-yaml) (($# >= 2)) || die "$1 requires a value"; SHAPE_YAML=$2; shift 2 ;;
        --model) (($# >= 2)) || die "$1 requires a value"; MODEL=$2; shift 2 ;;
        --parallel) (($# >= 2)) || die "$1 requires a value"; PARALLEL=$2; shift 2 ;;
        --dtypes) (($# >= 2)) || die "$1 requires a value"; DTYPES=$2; shift 2 ;;
        --hot-warmup) (($# >= 2)) || die "$1 requires a value"; HOT_WARMUP=$2; shift 2 ;;
        --hot-iter) (($# >= 2)) || die "$1 requires a value"; HOT_ITER=$2; shift 2 ;;
        --start-shape) (($# >= 2)) || die "$1 requires a value"; START_SHAPE=$2; shift 2 ;;
        --max-shapes) (($# >= 2)) || die "$1 requires a value"; MAX_SHAPES=$2; shift 2 ;;
        --run-name) (($# >= 2)) || die "$1 requires a value"; RUN_NAME=$2; shift 2 ;;
        --output-root) (($# >= 2)) || die "$1 requires a value"; OUTPUT_ROOT=$2; shift 2 ;;
        --python) (($# >= 2)) || die "$1 requires a value"; PYTHON_BIN=$2; shift 2 ;;
        --fail-fast) FAIL_FAST=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ $RUN_NAME != */* ]] || die "--run-name must not contain '/'"
[[ $START_SHAPE =~ ^[0-9]+$ ]] || die "--start-shape must be a non-negative integer"
[[ -z $MAX_SHAPES || $MAX_SHAPES =~ ^[1-9][0-9]*$ ]] || die "--max-shapes must be a positive integer"
[[ $HOT_WARMUP =~ ^[0-9]+$ ]] || die "--hot-warmup must be a non-negative integer"
[[ $HOT_ITER =~ ^[1-9][0-9]*$ ]] || die "--hot-iter must be a positive integer"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset USE_FLAGTUNE
unset FLAGTUNE_INCLUDE

if [[ -z $PARALLEL ]]; then
    PARALLEL=$(
        "$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null
    ) || die "cannot query visible GPUs with $PYTHON_BIN"
fi
[[ $PARALLEL =~ ^[1-9][0-9]*$ ]] || die "--parallel must be a positive integer"

mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd -P)
RESULT_DIR="$OUTPUT_ROOT/$RUN_NAME"
CACHE_ROOT="$OUTPUT_ROOT/cache/$RUN_NAME"
[[ ! -e $RESULT_DIR ]] || die "result directory already exists: $RESULT_DIR"
mkdir -p "$RESULT_DIR" "$CACHE_ROOT"

if [[ -n $SHAPE_YAML ]]; then
    SHAPE_ARGS=(--shape-yaml "$SHAPE_YAML")
else
    SHAPE_ARGS=(--model "$MODEL")
fi

COMMAND=(
    "$PYTHON_BIN" "$SCRIPT_DIR/tuning_latency_compare.py"
    "${SHAPE_ARGS[@]}"
    --methods default,expanded
    --cache-root "$CACHE_ROOT"
    --output-dir "$RESULT_DIR"
    --run-name "$RUN_NAME"
    --cold-warmup 0
    --cold-iter 1
    --hot-warmup "$HOT_WARMUP"
    --hot-iter "$HOT_ITER"
    --parallel "$PARALLEL"
    --dtypes "$DTYPES"
    --start-shape "$START_SHAPE"
)
if [[ -n $MAX_SHAPES ]]; then
    COMMAND+=(--max-shapes "$MAX_SHAPES")
fi
if ((FAIL_FAST)); then
    COMMAND+=(--fail-fast)
fi

{
    printf 'project_root=%s\n' "$PROJECT_ROOT"
    printf 'result_dir=%s\n' "$RESULT_DIR"
    printf 'cache_root=%s\n' "$CACHE_ROOT"
    printf 'default_method=%s\n' 'USE_FLAGTUNE unset; default configs with XGBoost+GA'
    printf 'expanded_method=%s\n' 'USE_FLAGTUNE=1; exhaustive expanded config search'
    printf 'command='
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
} > "$RESULT_DIR/invocation.txt"

printf 'Project root: %s\n' "$PROJECT_ROOT"
printf 'Result directory: %s\n' "$RESULT_DIR"
printf 'Cache root: %s\n' "$CACHE_ROOT"
printf 'Parallel workers: %s\n' "$PARALLEL"
printf 'Command: '
printf '%q ' "${COMMAND[@]}"
printf '\n'

if ((DRY_RUN)); then
    printf 'Dry run only; benchmark was not started.\n'
    exit 0
fi

"$PYTHON_BIN" - "$PARALLEL" <<'PY'
import sys

import torch
import flag_gems
import triton.flagtune

parallel = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
visible = torch.cuda.device_count()
if parallel > visible:
    raise SystemExit(
        f"parallel workers ({parallel}) exceed visible CUDA devices ({visible})"
    )
print(f"Preflight OK: flag_gems={flag_gems.__file__}")
print(f"Preflight OK: triton.flagtune={triton.flagtune.__file__}")
print(f"Preflight OK: visible_cuda_devices={visible}")
PY

set +e
"${COMMAND[@]}" 2>&1 | tee "$RESULT_DIR/run.log"
BENCHMARK_STATUS=${PIPESTATUS[0]}
set -e

if ((BENCHMARK_STATUS != 0)); then
    printf 'Benchmark failed with exit code %s. Partial outputs: %s\n' \
        "$BENCHMARK_STATUS" "$RESULT_DIR" >&2
    exit "$BENCHMARK_STATUS"
fi

EXPECTED_FILES=(
    tuning_latency_compare.csv
    tuning_latency_compare.jsonl
    tuning_latency_compare.md
    tuning_latency_compare_by_shape.csv
    manifest.json
)
for name in "${EXPECTED_FILES[@]}"; do
    [[ -s $RESULT_DIR/$name ]] || die "missing or empty output: $RESULT_DIR/$name"
done

printf 'Benchmark and aggregation completed: %s\n' "$RESULT_DIR"
