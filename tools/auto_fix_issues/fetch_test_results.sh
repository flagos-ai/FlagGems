#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# fetch_test_results.sh - Fetch test results for FlagGems issues
#
# Usage:
#   ./fetch_test_results.sh [OPTIONS]
#
# Options:
#   --issue-id ID           Fetch test context for a specific issue
#   --test-run-id ID        Fetch all results for a test run
#   --failed-only           Show only failed operators (with --test-run-id)
#   --format FORMAT         Output format: table, json (default: table)
#   --help                  Show this help
# ============================================================

BASE_URL="${ISSUES_URL:-http://10.1.4.213:31080}"
USERNAME="${ISSUES_USER:-admin}"
PASSWORD="${ISSUES_PASS:-admin123}"

ISSUE_ID=""
TEST_RUN_ID=""
FAILED_ONLY=false
FORMAT="table"

while [[ $# -gt 0 ]]; do
    case ${1:-} in
        --issue-id) ISSUE_ID="$2"; shift 2 ;;
        --test-run-id) TEST_RUN_ID="$2"; shift 2 ;;
        --failed-only) FAILED_ONLY=true; shift ;;
        --format) FORMAT="$2"; shift 2 ;;
        --help) head -14 "$0" | tail -10; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Login and get token
TOKEN=$(curl -sf -X POST "${BASE_URL}/api/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"${USERNAME}\",\"password\":\"${PASSWORD}\"}" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")

if [[ -z "$TOKEN" ]]; then
    echo "Error: Login failed" >&2
    exit 1
fi

AUTH="Authorization: Bearer ${TOKEN}"

# Fetch test context for a specific issue
if [[ -n "$ISSUE_ID" ]]; then
    RESULT=$(curl -sf "${BASE_URL}/api/issues/${ISSUE_ID}/test-context" -H "$AUTH")

    if [[ "$FORMAT" == "json" ]]; then
        echo "$RESULT" | python3 -m json.tool
        exit 0
    fi

    echo "$RESULT" | python3 -c "
import json, sys
d = json.load(sys.stdin)[0]
tr = d['test_result']
run = d['test_run']
env = run.get('test_env', {})

print('=== Test Context for Issue #${ISSUE_ID} ===')
print()
print('--- Test Result ---')
print(f\"  Status: {'PASS' if tr['acc_status'] else 'FAIL'}\")
print(f\"  Accuracy: {tr['acc_passed']}/{tr['acc_total']} passed, {tr['acc_failed']} failed\")
print(f\"  Speedup: {tr.get('avg_speedup', 'N/A')}\")
print()
print('--- Test Environment ---')
torch = env.get('torch', {})
print(f\"  PyTorch: {torch.get('version', 'N/A')}\")
print(f\"  Device: {torch.get('device_name', 'N/A')} x{torch.get('device_count', 'N/A')}\")
triton = env.get('triton', {})
print(f\"  Triton: {triton.get('version', 'N/A')}\")
print(f\"  Python: {env.get('python', 'N/A')}\")
fg = env.get('flag_gems', {})
print(f\"  FlagGems: {fg.get('version', 'N/A')}\")
print(f\"  OS: {env.get('os_name', 'N/A')} {env.get('os_release', 'N/A')}\")
print()
print('--- Perf Results ---')
for f in tr.get('perf_funcs', []):
    print(f\"  {f['func_name']}: avg_speedup={f.get('avg_speedup', 'N/A')}\")
    for dtype, speed in f.get('dtype_speedups', {}).items():
        print(f\"    {dtype}: {speed:.4f}\")
"
    exit 0
fi

# Fetch all results for a test run
if [[ -n "$TEST_RUN_ID" ]]; then
    RESULT=$(curl -sf "${BASE_URL}/api/reports/test-run/${TEST_RUN_ID}/results" -H "$AUTH")

    if [[ "$FORMAT" == "json" ]]; then
        echo "$RESULT" | python3 -m json.tool
        exit 0
    fi

    FILTER_FLAG=$FAILED_ONLY python3 -c "
import json, sys, os
d = json.load(sys.stdin)
results = d['results']
failed_only = os.environ.get('FILTER_FLAG', 'false') == 'true'

if failed_only:
    results = [r for r in results if not r.get('acc_status', True)]

print(f'=== Test Run #${TEST_RUN_ID} Results ({len(results)} operators) ===')
print()
print(f'{\"Operator\":<35} | {\"Status\":<6} | {\"Passed\":>6} | {\"Failed\":>6} | {\"Total\":>6} | {\"Speedup\":>8}')
print('-' * 80)
for r in results:
    name = r['operator_name']
    if len(name) > 33:
        name = name[:30] + '...'
    status = 'PASS' if r.get('acc_status') else 'FAIL'
    passed = r.get('acc_passed', 0)
    failed = r.get('acc_failed', 0)
    total = r.get('acc_total', 0)
    speedup = f\"{r['avg_speedup']:.4f}\" if r.get('avg_speedup') else '-'
    print(f\"{name:<35} | {status:<6} | {passed:>6} | {failed:>6} | {total:>6} | {speedup:>8}\")
" <<< "$RESULT"
    exit 0
fi

echo "Error: Must specify --issue-id or --test-run-id" >&2
exit 1
