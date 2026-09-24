#!/usr/bin/env bash
#
# Trigger a GitHub Actions workflow through workflow_dispatch.
#
# Why this script exists:
#   GitHub Actions' built-in schedule event is best effort. In this repository,
#   scheduled workflow creation has been observed to be delayed by 1-5 hours
#   during load, and GitHub may also drop a scheduled event. Therefore the
#   built-in schedule cannot provide a reliable execution time for GPU tests.
#
#   The workflow is intentionally triggered by crontab on an always-on
#   control-plane server instead. The server's crontab creates the workflow run
#   at the required wall-clock time, while GitHub still allocates the runners
#   and executes the workflow.
#
# Configuration and installation:
#   1. Create a fine-grained GitHub token with repository access to the target
#      repository and "Actions: Read and write" permission.
#
#   2. Install this script on an always-on server. The server does not need a
#      GPU and should not be one of the self-hosted test runners.
#
#      sudo install -d -m 700 /opt/flaggems /etc/flaggems
#      sudo install -m 755 .github/scripts/trigger_workflow.sh \
#        /opt/flaggems/trigger_workflow.sh
#
#   3. Create the protected environment file below. Use "export" because the
#      crontab sources this file before starting the script.
#
#      sudo tee /etc/flaggems/ops-test-trigger.env >/dev/null <<'EOF'
#      export GITHUB_TOKEN=REPLACE_WITH_FINE_GRAINED_TOKEN
#      export GITHUB_ORG=flagos-ai
#      export GITHUB_REPO=FlagGems
#      export GITHUB_REF=master
#      export GITHUB_WORKFLOW=ops-test.yaml
#      export GITHUB_PROXY=socks5h://127.0.0.1:1080
#      EOF
#      sudo chmod 600 /etc/flaggems/ops-test-trigger.env
#
#      GITHUB_ORG defaults to "flagos-ai".
#      GITHUB_REPO defaults to "FlagGems".
#      GITHUB_REF defaults to "master"; it is never looked up remotely.
#      GITHUB_WORKFLOW defaults to "ops-test.yaml".
#      GITHUB_PROXY is optional. Leave it empty for direct access. It can be
#      an HTTP(S) or SOCKS proxy accepted by curl, for example:
#      socks5h://127.0.0.1:1080.
#
#   4. Install the crontab entry below. CRON_TZ is supported by Vixie cron and
#      cronie. If the server uses a cron implementation without CRON_TZ,
#      configure the server timezone as Asia/Shanghai, or use 13:30 on a UTC
#      server.
#
#      sudo crontab -e
#
#      SHELL=/bin/bash
#      MAILTO=
#      CRON_TZ=Asia/Shanghai
#      30 21 * * 3,6 . /etc/flaggems/ops-test-trigger.env && \
#        /opt/flaggems/trigger_workflow.sh \
#        >> /var/log/flaggems-ops-test-trigger.log 2>&1
#
#      The entry means Wednesday and Saturday at 21:30 Beijing time.
#
# Execution flow:
#   1. Parse and individually validate org, repo, workflow, ref, proxy, and
#      workflow inputs locally.
#   2. Build the workflow_dispatch API URL. No repository/default-branch or
#      workflow-existence lookup is performed, so there is no unnecessary
#      validation traffic. GitHub validates the actual repository/workflow when
#      the final dispatch request is posted.
#   3. POST the workflow_dispatch payload.
#   4. Treat HTTP 204 as success.
#   5. If the request fails, sleep 60 seconds and query recent
#      workflow_dispatch runs before retrying. If a run matching the attempted
#      ref was created, stop successfully to avoid duplicate execution.
#   6. If no matching run was created, retry. The initial request plus five
#      retries are allowed. If the run check itself fails, stop instead of
#      risking a duplicate workflow.
#
# Retry behavior:
#   A failed dispatch is retried up to five times after the initial attempt.
#   Each retry waits 60 seconds and checks that the previous attempt did not
#   create a workflow run. This protects against an accepted GitHub request
#   followed by a client-side timeout.
#
# Supported command-line options:
#   --org VALUE             GitHub organization/user. Default: flagos-ai.
#   --repo VALUE            Repository name. Default: FlagGems.
#   --ref VALUE             Dispatch ref (branch or tag). Default: master.
#   --workflow VALUE        Workflow file name (.yaml or .yml). Default:
#                           ops-test.yaml.
#   --branch VALUE          Add workflow_dispatch input branch=VALUE.
#   --vendors VALUE         Add workflow_dispatch input vendors=VALUE.
#   --ops VALUE             Add workflow_dispatch input ops=VALUE.
#   --upload-log VALUE      Add workflow_dispatch input upload_log=VALUE.
#   --send-feishu VALUE     Add workflow_dispatch input send_feishu=VALUE.
#   --input KEY=VALUE       Add any additional workflow_dispatch input. Repeat
#                           this option for multiple custom inputs.
#   -f KEY=VALUE            Alias for --input, matching "gh workflow run".
#   -h, --help              Show this documentation.
#
# Environment variables are accepted as defaults and are overridden by command
# line options:
#   GITHUB_TOKEN             Required. Fine-grained token with Actions: write.
#   GITHUB_ORG               Default organization/user: flagos-ai.
#   GITHUB_REPO              Default repository: FlagGems.
#   GITHUB_REF               Default ref: master.
#   GITHUB_WORKFLOW          Default workflow file: ops-test.yaml.
#   GITHUB_PROXY             Optional curl proxy for GitHub API requests.
#
# ops-test.yaml example:
#   .github/scripts/trigger_workflow.sh \
#     --org flagos-ai \
#     --repo FlagGems \
#     --workflow ops-test.yaml \
#     --ref update_kunlunxin \
#     --branch update_kunlunxin \
#     --vendors Nvidia,MThreads,KunLunXin,Iluvatar,Ascend,Hygon,Metax,THead \
#     --ops abs,add,sum \
#     --upload-log skip \
#     --send-feishu skip
#
# Equivalent gh CLI command:
#   gh workflow run ops-test.yaml \
#     --repo flagos-ai/FlagGems \
#     --ref update_kunlunxin \
#     -f branch=update_kunlunxin \
#     -f vendors=Nvidia,MThreads,KunLunXin,Iluvatar,Ascend,Hygon,Metax,THead \
#     -f ops=abs,add,sum \
#     -f upload_log=skip \
#     -f send_feishu=skip
#
# The same inputs can be passed through the generic -f/--input option:
#   .github/scripts/trigger_workflow.sh \
#     --workflow ops-test.yaml \
#     --ref update_kunlunxin \
#     -f branch=update_kunlunxin \
#     -f vendors=Nvidia,MThreads,KunLunXin,Iluvatar,Ascend,Hygon,Metax,THead \
#     -f ops=abs,add,sum \
#     -f upload_log=skip \
#     -f send_feishu=skip
#
# Generic custom-input example:
#   .github/scripts/trigger_workflow.sh \
#     --workflow another-test.yaml \
#     --input environment=staging \
#     --input test_level=full
#
# The workflow must declare workflow_dispatch. If a workflow declares different
# inputs, pass them with repeated --input/-f options without changing this
# script.

set -euo pipefail

usage() {
  sed -n '1,/^set -euo pipefail$/p' "$0" | sed '$d'
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_option_value() {
  local option="$1"
  if (($# < 2)) || [[ -z "${2:-}" ]]; then
    die "${option} requires a value"
  fi
}

validate_identifier() {
  local name="$1"
  local value="$2"
  local pattern="$3"
  if [[ ! "${value}" =~ ${pattern} ]]; then
    die "invalid ${name}: ${value}"
  fi
}

validate_input_key() {
  local key="$1"
  validate_identifier "workflow input name" "${key}" '^[A-Za-z_][A-Za-z0-9_-]*$'
}

parse_input_pair() {
  local pair="$1"
  local key="${pair%%=*}"
  local value="${pair#*=}"

  if [[ "${pair}" != *=* ]] || [[ -z "${key}" ]]; then
    die "workflow input must use KEY=VALUE: ${pair}"
  fi
  validate_input_key "${key}"

  local existing
  for existing in "${INPUT_PAIRS[@]:-}"; do
    if [[ "${existing%%=*}" == "${key}" ]]; then
      die "duplicate workflow input: ${key}"
    fi
  done
  INPUT_PAIRS+=("${key}=${value}")
}

ORG="${GITHUB_ORG:-flagos-ai}"
REPO="${GITHUB_REPO:-FlagGems}"
REF="${GITHUB_REF:-master}"
WORKFLOW="${GITHUB_WORKFLOW:-ops-test.yaml}"
GITHUB_PROXY="${GITHUB_PROXY:-}"
INPUT_PAIRS=()

while (($# > 0)); do
  case "$1" in
    --org)
      require_option_value "$@"
      ORG="$2"
      shift 2
      ;;
    --repo)
      require_option_value "$@"
      REPO="$2"
      shift 2
      ;;
    --ref)
      require_option_value "$@"
      REF="$2"
      shift 2
      ;;
    --workflow)
      require_option_value "$@"
      WORKFLOW="$2"
      shift 2
      ;;
    --branch)
      require_option_value "$@"
      parse_input_pair "branch=$2"
      shift 2
      ;;
    --vendors)
      require_option_value "$@"
      parse_input_pair "vendors=$2"
      shift 2
      ;;
    --ops)
      require_option_value "$@"
      parse_input_pair "ops=$2"
      shift 2
      ;;
    --upload-log)
      require_option_value "$@"
      parse_input_pair "upload_log=$2"
      shift 2
      ;;
    --send-feishu)
      require_option_value "$@"
      parse_input_pair "send_feishu=$2"
      shift 2
      ;;
    --input|-f)
      require_option_value "$@"
      parse_input_pair "$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help for usage)"
      ;;
  esac
done

validate_identifier "org" "${ORG}" '^[A-Za-z0-9][A-Za-z0-9_.-]{0,38}$'
validate_identifier "repo" "${REPO}" '^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$'
validate_identifier "workflow" "${WORKFLOW}" '^[A-Za-z0-9][A-Za-z0-9_.-]*\.(yaml|yml)$'
validate_identifier "ref" "${REF}" '^[A-Za-z0-9._/-]+$'

if [[ -n "${GITHUB_PROXY}" ]]; then
  validate_identifier "GITHUB_PROXY" "${GITHUB_PROXY}" '^[A-Za-z][A-Za-z0-9+.-]*://[^[:space:]]+$'
fi
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  die "GITHUB_TOKEN must be set"
fi
if ! command -v curl >/dev/null 2>&1; then
  die "curl is required"
fi
if ! command -v python3 >/dev/null 2>&1; then
  die "python3 is required to build JSON and inspect workflow runs"
fi

api_headers=(
  --header 'Accept: application/vnd.github+json'
  --header "Authorization: Bearer ${GITHUB_TOKEN}"
  --header 'X-GitHub-Api-Version: 2022-11-28'
)
curl_common_args=(
  --silent
  --show-error
  --connect-timeout 10
  --max-time 60
  "${api_headers[@]}"
)
if [[ -n "${GITHUB_PROXY}" ]]; then
  curl_common_args+=(--proxy "${GITHUB_PROXY}")
fi

payload="$(
  python3 - "${REF}" "${INPUT_PAIRS[@]}" <<'PY'
import json
import sys

ref = sys.argv[1]
inputs = {}
for pair in sys.argv[2:]:
    key, value = pair.split("=", 1)
    inputs[key] = value

print(json.dumps({"ref": ref, "inputs": inputs}, ensure_ascii=True, separators=(",", ":")))
PY
)" || die "failed to build workflow_dispatch JSON payload"

workflow_url="https://api.github.com/repos/${ORG}/${REPO}/actions/workflows/${WORKFLOW}"
dispatch_url="${workflow_url}/dispatches"
runs_url="${workflow_url}/runs?event=workflow_dispatch&per_page=20"
max_retries=5
retry_sleep_seconds=60

find_recent_dispatch() {
  local attempt_started_at="$1"
  local response
  local curl_rc
  local status
  local body

  response="$(
    curl \
      "${curl_common_args[@]}" \
      --write-out $'\n%{http_code}' \
      "${runs_url}"
  )" || curl_rc=$?
  curl_rc="${curl_rc:-0}"

  if ((curl_rc != 0)) || [[ "${response}" != *$'\n'* ]]; then
    return 2
  fi

  status="${response##*$'\n'}"
  body="${response%$'\n'*}"
  if [[ "${status}" != "200" ]]; then
    return 2
  fi

  printf '%s' "${body}" | python3 -c '
import json
import sys
from datetime import datetime

ref = sys.argv[1]
started_at = datetime.fromisoformat(sys.argv[2].replace("Z", "+00:00"))
data = json.load(sys.stdin)

for run in data.get("workflow_runs", []):
    if run.get("event") != "workflow_dispatch":
        continue
    created_at = run.get("created_at")
    if not created_at:
        continue
    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    if created < started_at:
        continue
    if run.get("head_branch") == ref or run.get("head_sha") == ref:
        print(run.get("id", "unknown"))
        break
' "${REF}" "${attempt_started_at}"
}

for ((retry=0; retry<=max_retries; retry++)); do
  attempt_started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  attempt=$((retry + 1))
  printf 'Dispatch attempt %d/%d: %s/%s workflow %s, ref %s\n' \
    "${attempt}" "$((max_retries + 1))" "${ORG}" "${REPO}" "${WORKFLOW}" "${REF}"

  unset curl_rc
  response="$(
    curl \
      "${curl_common_args[@]}" \
      --request POST \
      --header 'Content-Type: application/json' \
      --data "${payload}" \
      --write-out $'\n%{http_code}' \
      "${dispatch_url}"
  )" || curl_rc=$?
  curl_rc="${curl_rc:-0}"

  if ((curl_rc == 0)) && [[ "${response}" == *$'\n'* ]]; then
    dispatch_status="${response##*$'\n'}"
    dispatch_body="${response%$'\n'*}"
  else
    dispatch_status=""
    dispatch_body="${response:-}"
  fi

  if [[ "${dispatch_status}" == "204" ]]; then
    printf 'Triggered %s/%s workflow %s at %s for ref %s\n' \
      "${ORG}" \
      "${REPO}" \
      "${WORKFLOW}" \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      "${REF}"
    exit 0
  fi

  printf 'Dispatch failed (curl_rc=%s, http_status=%s).\n' \
    "${curl_rc}" "${dispatch_status:-unknown}" >&2
  if [[ -n "${dispatch_body}" ]]; then
    printf '%s\n' "${dispatch_body}" >&2
  fi

  if ((retry == max_retries)); then
    die "workflow_dispatch failed after ${max_retries} retries"
  fi

  printf 'Waiting %ss before checking for an accepted run.\n' \
    "${retry_sleep_seconds}" >&2
  sleep "${retry_sleep_seconds}"

  recent_run_id="$(find_recent_dispatch "${attempt_started_at}")" || check_rc=$?
  check_rc="${check_rc:-0}"
  if ((check_rc != 0)); then
    die "cannot verify whether the previous dispatch created a run; refusing to retry"
  fi
  if [[ -n "${recent_run_id}" ]]; then
    printf 'Previous dispatch created workflow run %s; refusing duplicate retry.\n' \
      "${recent_run_id}"
    exit 0
  fi

  unset curl_rc check_rc recent_run_id
done
