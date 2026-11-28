#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="${1:-both}"
mkdir -p "${SCRIPT_DIR}/results"

run_one() {
  local state="$1"
  local native_flag=0
  if [[ "${state}" == "on" ]]; then native_flag=1; fi
  FLAGGEMS_MTHREADS_NATIVE_INDEX_PUT="${native_flag}" python "${SCRIPT_DIR}/run_correctness.py" "${state}" \
    2>&1 | tee "${SCRIPT_DIR}/results/correctness_${state}.log"
}

case "${STATE}" in
  on|off) run_one "${STATE}" ;;
  both) run_one off; run_one on ;;
  *) echo "usage: $0 [off|on|both]" >&2; exit 2 ;;
esac
