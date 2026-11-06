#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE="${1:-both}"
WARMUP="${WARMUP:-10}"
ITER="${ITER:-10}"
RESULTS="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS}"

run_one() {
  local state="$1"
  local native_flag="0"
  if [[ "${state}" == "on" ]]; then native_flag="1"; fi
  FLAGGEMS_MTHREADS_NATIVE_INDEX_PUT="${native_flag}" \
    python "${SCRIPT_DIR}/run_benchmark.py" "${state}" \
      --warmup "${WARMUP}" --iter "${ITER}" \
      --output "${RESULTS}/official_${state}.json" \
    2>&1 | tee "${RESULTS}/benchmark_${state}.log"
}

case "${STATE}" in
  on|off) run_one "${STATE}" ;;
  both)
    run_one off
    run_one on
    python "${SCRIPT_DIR}/trace_official_hits.py" \
      --output "${RESULTS}/native_trace.json"
    python "${SCRIPT_DIR}/audit_csv.py" \
      --off "${RESULTS}/official_off.json" \
      --on "${RESULTS}/official_on.json" \
      --out-dir "${RESULTS}" \
      --flat-csv-dir "${SCRIPT_DIR}"
    ;;
  *) echo "usage: $0 [off|on|both]" >&2; exit 2 ;;
esac
