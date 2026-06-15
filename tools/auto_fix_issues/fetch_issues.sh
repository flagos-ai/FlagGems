#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# fetch_issues.sh - Query FlagGems internal issue tracking system
#
# Usage:
#   ./fetch_issues.sh [OPTIONS]
#
# Options:
#   --assigned-to USER_ID   Filter by assignee (default: 22)
#   --status STATUS         open, in_progress, resolved, closed (default: open)
#   --page-size N           Results per page (default: 100)
#   --stats                 Show statistics summary
#   --all                   Show all issues (no assignee filter)
#   --format FORMAT         Output format: table, json (default: table)
#   --help                  Show this help
# ============================================================

BASE_URL="${ISSUES_URL:-http://10.1.4.213:31080}"
USERNAME="${ISSUES_USER:-admin}"
PASSWORD="${ISSUES_PASS:-admin123}"

ASSIGNED_TO="22"
STATUS="open"
PAGE_SIZE=100
SHOW_STATS=false
SHOW_ALL=false
FORMAT="table"

while [[ $# -gt 0 ]]; do
    case ${1:-} in
        --assigned-to) ASSIGNED_TO="$2"; shift 2 ;;
        --status) STATUS="$2"; shift 2 ;;
        --page-size) PAGE_SIZE="$2"; shift 2 ;;
        --stats) SHOW_STATS=true; shift ;;
        --all) SHOW_ALL=true; shift ;;
        --format) FORMAT="$2"; shift 2 ;;
        --help) head -16 "$0" | tail -12; exit 0 ;;
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

# Fetch statistics
if [[ "$SHOW_STATS" == "true" ]]; then
    curl -sf "${BASE_URL}/api/issues/stats" -H "$AUTH" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('=== Issue Statistics ===')
print(f'Open: {d[\"open_count\"]}  |  Closed: {d.get(\"by_status\", {}).get(\"closed\", 0)}')
print(f'New this week: {d[\"new_this_week\"]}')
print(f'Avg age: {d[\"avg_age_days\"]:.1f} days')
print(f'Overdue (>14d): {d[\"overdue_count\"]}')
print()
print('By assignee:')
for a in d.get('by_assignee', []):
    print(f'  {a[\"name\"]}: open={a.get(\"open\",0)} in_progress={a.get(\"in_progress\",0)} resolved={a.get(\"resolved\",0)}')
"
    exit 0
fi

# Build query params
PARAMS="page_size=${PAGE_SIZE}"
if [[ "$SHOW_ALL" == "false" ]]; then
    PARAMS="${PARAMS}&assigned_to=${ASSIGNED_TO}"
fi
if [[ -n "$STATUS" ]]; then
    PARAMS="${PARAMS}&status=${STATUS}"
fi

# Fetch issues
RESULT=$(curl -sf "${BASE_URL}/api/issues/?${PARAMS}" -H "$AUTH")

if [[ "$FORMAT" == "json" ]]; then
    echo "$RESULT" | python3 -m json.tool
    exit 0
fi

# Table format
echo "$RESULT" | python3 -c "
import json, sys
d = json.load(sys.stdin)
total = d['total']
items = d['items']

print(f'=== Issues ({total} total) ===')
print(f'{\"ID\":>5} | {\"Operator\":<35} | {\"Type\":<16} | {\"Severity\":<8} | {\"Backend\":<8} | {\"Age\":>5} | Status')
print('-' * 100)
for i in items:
    op = i.get('operator_name') or '-'
    if len(op) > 33:
        op = op[:30] + '...'
    it = i.get('issue_type') or '-'
    sev = i.get('severity') or '-'
    be = i.get('backend_name') or '-'
    age = f\"{i['age_days']}d\"
    st = i['status']
    print(f\"{i['id']:>5} | {op:<35} | {it:<16} | {sev:<8} | {be:<8} | {age:>5} | {st}\")
"
