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
#      sudo install -d -m 755 /opt/github-actions
#      sudo install -d -m 700 /etc/github-actions
#      sudo install -d -m 755 /var/log/github-actions
#      sudo install -m 755 .github/scripts/trigger_workflow.sh \
#        /opt/github-actions/trigger_workflow.sh
#
#   3. Create the protected environment file below. Use "export" because the
#      crontab sources this file before starting the script.
#
#      sudo tee /etc/github-actions/ops-test-trigger.env >/dev/null <<'EOF'
#      export GITHUB_TOKEN=REPLACE_WITH_FINE_GRAINED_TOKEN
#      export GITHUB_ORG=flagos-ai
#      export GITHUB_REPO=FlagGems
#      export GITHUB_REF=master
#      export GITHUB_WORKFLOW=ops-test.yaml
#      # The GitHub API URL is HTTPS, so one explicit HTTPS proxy is enough.
#      # Choose one proxy example, not both. No http_proxy or no_proxy is
#      # required for this script when only api.github.com is accessed.
#      export GITHUB_HTTPS_PROXY=socks5h://127.0.0.1:1080
#      # export GITHUB_HTTPS_PROXY=http://proxy.example.com:3128
#      export GITHUB_ACTIONS_LOG_FILE=/var/log/github-actions/trigger_workflow.log
#      EOF
#      sudo chmod 600 /etc/github-actions/ops-test-trigger.env
#
#      GITHUB_ORG defaults to "flagos-ai".
#      GITHUB_REPO defaults to "FlagGems".
#      GITHUB_REF defaults to "master"; it is never looked up remotely.
#      GITHUB_WORKFLOW defaults to "ops-test.yaml".
#      GITHUB_HTTPS_PROXY is optional. Leave it empty for direct access. It
#      applies only to this script's HTTPS requests to api.github.com. It can
#      be either a SOCKS5 proxy or an HTTP CONNECT proxy:
#      socks5h://127.0.0.1:1080
#      http://proxy.example.com:3128
#      GITHUB_PROXY is still accepted as a backward-compatible alias, but
#      GITHUB_HTTPS_PROXY takes precedence when both are set.
#      GITHUB_ACTIONS_LOG_FILE defaults to
#      /var/log/github-actions/trigger_workflow.log. For a non-root manual run,
#      set it to a writable path such as /tmp/trigger_workflow.log.
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
#      30 21 * * 3,6 . /etc/github-actions/ops-test-trigger.env && \
#        /opt/github-actions/trigger_workflow.sh
#
#      The script timestamps and appends stdout, stderr, curl errors, HTTP
#      status codes, and non-empty GitHub response bodies to
#      GITHUB_ACTIONS_LOG_FILE, so cron redirection is not required.
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
# Required parameters:
#   GITHUB_TOKEN          Required environment variable. Fine-grained token
#                         with repository "Actions: Read and write".
#   No command-line option is mandatory because org, repo, workflow, and ref
#   all have defaults. Direct network access or GITHUB_HTTPS_PROXY is also
#   required to reach api.github.com.
#
# Optional script parameters:
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
# Parameter roles:
#   --org, --repo, --workflow, and --ref are script/API parameters. They build
#   the workflow_dispatch URL and top-level "ref" field.
#   --branch, --vendors, --ops, --upload-log, --send-feishu, and --input/-f
#   are workflow_dispatch input parameters. They are serialized into the
#   payload's "inputs" object and must be declared by the target workflow.
#   For ops-test.yaml specifically, --branch controls the source ref checked
#   out by the workflow. It is different from the API --ref.
#
# Environment variables are accepted as defaults and are overridden by command
# line options:
#   GITHUB_TOKEN             Required. Fine-grained token with Actions: write.
#   GITHUB_ORG               Default organization/user: flagos-ai.
#   GITHUB_REPO              Default repository: FlagGems.
#   GITHUB_REF               Default ref: master.
#   GITHUB_WORKFLOW          Default workflow file: ops-test.yaml.
#   GITHUB_HTTPS_PROXY       Optional proxy for HTTPS GitHub API requests.
#   GITHUB_PROXY              Backward-compatible alias for GITHUB_HTTPS_PROXY.
#   GITHUB_ACTIONS_LOG_FILE  Log file. Default:
#                            /var/log/github-actions/trigger_workflow.log.
#
# Example 1: scheduled/default run (required values only):
#   export GITHUB_TOKEN=REPLACE_WITH_FINE_GRAINED_TOKEN
#   export GITHUB_ACTIONS_LOG_FILE=/tmp/trigger_workflow.log
#   /opt/github-actions/trigger_workflow.sh
#
# Example 2: use a proxy (choose one):
#   # SOCKS5 proxy with remote DNS resolution:
#   export GITHUB_HTTPS_PROXY=socks5h://127.0.0.1:1080
#
#   # HTTP CONNECT proxy:
#   export GITHUB_HTTPS_PROXY=http://proxy.example.com:3128
#
#   /opt/github-actions/trigger_workflow.sh
#
#   Alternatively, without the script-specific variable, standard curl
#   environment configuration is sufficient:
#   unset GITHUB_HTTPS_PROXY GITHUB_PROXY
#   export https_proxy=socks5h://127.0.0.1:1080
#   # or: export https_proxy=http://proxy.example.com:3128
#   # http_proxy and no_proxy are not required because every request is to
#   # https://api.github.com.
#   /opt/github-actions/trigger_workflow.sh
#
# Example 3: ops-test.yaml, run source from the same branch:
#   .github/scripts/trigger_workflow.sh \
#     --org flagos-ai \
#     --repo FlagGems \
#     --workflow ops-test.yaml \
#     --ref example_branch \
#     --branch example_branch \
#     --vendors Nvidia,MThreads,KunLunXin,Iluvatar,Ascend,Hygon,Metax,THead \
#     --ops abs,add,sum \
#     --upload-log skip \
#     --send-feishu skip
#
#      --ref example_branch selects the branch containing the workflow file.
#      --branch example_branch is an ops-test.yaml input and controls the
#      source ref checked out for the test. Both are appropriate when the
#      workflow definition and test source are on the same branch.
#
# Example 4: workflow definition on master, test source from a tag or commit:
#   .github/scripts/trigger_workflow.sh \
#     --workflow ops-test.yaml \
#     --ref master \
#     --branch example_tag \
#     --vendors Nvidia,MThreads,KunLunXin,Iluvatar,Ascend,Hygon,Metax,THead \
#     --ops abs,add,sum \
#     --upload-log skip \
#     --send-feishu skip
#
#      In this case --ref and --branch intentionally differ. If --branch is
#      omitted for ops-test.yaml, the workflow defaults its checkout to
#      master, even when --ref points to another branch/tag. For a generic
#      workflow without a "branch" input, only --ref is needed.
#      GitHub workflow_dispatch --ref accepts a branch or tag. The ops-test
#      workflow's --branch input additionally accepts a commit SHA, for example
#      --branch example_commit_id.
#
# Equivalent gh CLI command for Example 3:
#   gh workflow run ops-test.yaml \
#     --repo flagos-ai/FlagGems \
#     --ref example_branch \
#     -f branch=example_branch \
#     -f vendors=Nvidia,MThreads,KunLunXin,Iluvatar,Ascend,Hygon,Metax,THead \
#     -f ops=abs,add,sum \
#     -f upload_log=skip \
#     -f send_feishu=skip
#
# Generic workflow input parameters:
#   --input KEY=VALUE and -f KEY=VALUE are script options that add
#   {"KEY":"VALUE"} under the GitHub workflow_dispatch "inputs" object.
#   They are equivalent to the "-f key=value" options accepted by
#   "gh workflow run", and can be repeated for arbitrary inputs declared by
#   the target workflow. They are not shell arguments inside the workflow.
#
# Example 5: the same ops-test inputs using generic -f/--input:
#   .github/scripts/trigger_workflow.sh \
#     --workflow ops-test.yaml \
#     --ref example_branch \
#     -f branch=example_branch \
#     -f vendors=Nvidia,MThreads,KunLunXin,Iluvatar,Ascend,Hygon,Metax,THead \
#     -f ops=abs,add,sum \
#     -f upload_log=skip \
#     -f send_feishu=skip
#
# Example 6: arbitrary custom workflow inputs:
#   .github/scripts/trigger_workflow.sh \
#     --workflow another-test.yaml \
#     --ref example_tag \
#     --input environment=staging \
#     --input test_level=full
#
# The workflow must declare workflow_dispatch. If a workflow declares different
# inputs, pass them with repeated --input/-f options without changing this
# script.

set -euo pipefail

LOG_FILE="${GITHUB_ACTIONS_LOG_FILE:-/var/log/github-actions/trigger_workflow.log}"
LOG_READY=0
LOG_WARNED=0

usage() {
  sed -n '1,/^set -euo pipefail$/p' "$0" | sed '$d'
}

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

setup_log_file() {
  if ((LOG_READY == 1)); then
    return 0
  fi

  local log_dir
  log_dir="$(dirname -- "${LOG_FILE}")"
  if mkdir -p "${log_dir}" 2>/dev/null && : >>"${LOG_FILE}" 2>/dev/null; then
    LOG_READY=1
    return 0
  fi

  if ((LOG_WARNED == 0)); then
    printf '[%s] WARN: cannot write log file %s; continuing with console output only\n' \
      "$(timestamp)" "${LOG_FILE}" >&2
    LOG_WARNED=1
  fi
  return 1
}

log() {
  local line
  line="[$(timestamp)] INFO: $*"
  printf '%s\n' "${line}"
  if setup_log_file; then
    printf '%s\n' "${line}" >>"${LOG_FILE}" || true
  fi
}

log_error() {
  local line
  line="[$(timestamp)] ERROR: $*"
  printf '%s\n' "${line}" >&2
  if setup_log_file; then
    printf '%s\n' "${line}" >>"${LOG_FILE}" || true
  fi
}

log_multiline() {
  local level="$1"
  local prefix="$2"
  local text="$3"
  local line

  while IFS= read -r line || [[ -n "${line}" ]]; do
    if [[ "${level}" == "ERROR" ]]; then
      log_error "${prefix}${line}"
    else
      log "${prefix}${line}"
    fi
  done <<<"${text}"
}

die() {
  log_error "$*"
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
GITHUB_HTTPS_PROXY="${GITHUB_HTTPS_PROXY:-${GITHUB_PROXY:-}}"
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

if [[ -n "${GITHUB_HTTPS_PROXY}" ]]; then
  validate_identifier "GITHUB_HTTPS_PROXY" "${GITHUB_HTTPS_PROXY}" '^[A-Za-z][A-Za-z0-9+.-]*://[^[:space:]]+$'
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
if [[ -n "${GITHUB_HTTPS_PROXY}" ]]; then
  curl_common_args+=(--proxy "${GITHUB_HTTPS_PROXY}")
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

curl_api() {
  local body_file="$1"
  local stderr_file="$2"
  shift 2

  local status
  local curl_rc
  status="$(
    curl \
      "${curl_common_args[@]}" \
      --output "${body_file}" \
      --write-out '%{http_code}' \
      "$@" \
      2>"${stderr_file}"
  )" || curl_rc=$?
  curl_rc="${curl_rc:-0}"

  printf '%s\t%s\n' "${curl_rc}" "${status}"
}

find_recent_dispatch() {
  local attempt_started_at="$1"
  local body_file
  local stderr_file
  local curl_result
  local curl_rc
  local status
  local body
  local curl_stderr
  local recent_run_id

  body_file="$(mktemp)"
  stderr_file="$(mktemp)"

  curl_result="$(curl_api "${body_file}" "${stderr_file}" "${runs_url}")"
  body="$(cat "${body_file}")"
  curl_stderr="$(cat "${stderr_file}")"
  rm -f "${body_file}" "${stderr_file}"

  IFS=$'\t' read -r curl_rc status <<<"${curl_result}"
  if ((curl_rc != 0)); then
    log_error "workflow run check request failed (curl_rc=${curl_rc}, http_status=${status:-unknown})"
    if [[ -n "${curl_stderr}" ]]; then
      log_multiline ERROR "curl stderr: " "${curl_stderr}"
    fi
    return 2
  fi

  if [[ "${status}" != "200" ]]; then
    log_error "workflow run check returned http_status=${status}"
    if [[ -n "${body}" ]]; then
      log_multiline ERROR "GitHub response body: " "${body}"
    fi
    return 2
  fi

  if recent_run_id="$(
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
  )"; then
    printf '%s\n' "${recent_run_id}"
  else
    log_error "failed to parse workflow run response"
    if [[ -n "${body}" ]]; then
      log_multiline ERROR "GitHub response body: " "${body}"
    fi
    return 2
  fi
}

log "Starting workflow dispatch trigger"
log "Target repository: ${ORG}/${REPO}"
log "Workflow file: ${WORKFLOW}"
log "Dispatch ref: ${REF}"
log "Dispatch URL: ${dispatch_url}"
if ((${#INPUT_PAIRS[@]} > 0)); then
  input_keys=()
  for input_pair in "${INPUT_PAIRS[@]}"; do
    input_keys+=("${input_pair%%=*}")
  done
  log "Workflow input keys: ${input_keys[*]}"
else
  log "Workflow input keys: none"
fi
if [[ -n "${GITHUB_HTTPS_PROXY}" ]]; then
  log "Using explicit GitHub HTTPS proxy configured by GITHUB_HTTPS_PROXY"
elif [[ -n "${https_proxy:-}${HTTPS_PROXY:-}" ]]; then
  log "No GITHUB_HTTPS_PROXY configured; curl may use HTTPS_PROXY/https_proxy from the environment"
else
  log "No explicit GitHub HTTPS proxy configured"
fi
log "Log file: ${LOG_FILE}"

for ((retry=0; retry<=max_retries; retry++)); do
  attempt_started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  attempt=$((retry + 1))
  log "Dispatch attempt ${attempt}/$((max_retries + 1)): ${ORG}/${REPO} workflow ${WORKFLOW}, ref ${REF}"

  body_file="$(mktemp)"
  stderr_file="$(mktemp)"
  curl_result="$(
    curl_api \
      "${body_file}" \
      "${stderr_file}" \
      --request POST \
      --header 'Content-Type: application/json' \
      --data "${payload}" \
      "${dispatch_url}"
  )"
  dispatch_body="$(cat "${body_file}")"
  curl_stderr="$(cat "${stderr_file}")"
  rm -f "${body_file}" "${stderr_file}"

  IFS=$'\t' read -r curl_rc dispatch_status <<<"${curl_result}"
  log "Dispatch attempt ${attempt}/$((max_retries + 1)) returned curl_rc=${curl_rc}, http_status=${dispatch_status:-unknown}"

  if [[ -n "${curl_stderr}" ]]; then
    log_multiline ERROR "curl stderr: " "${curl_stderr}"
  fi
  if [[ -n "${dispatch_body}" ]]; then
    if [[ "${dispatch_status}" == "204" ]]; then
      log_multiline INFO "GitHub response body: " "${dispatch_body}"
    else
      log_multiline ERROR "GitHub response body: " "${dispatch_body}"
    fi
  fi

  if [[ "${dispatch_status}" == "204" ]]; then
    log "Triggered ${ORG}/${REPO} workflow ${WORKFLOW} at $(date -u '+%Y-%m-%dT%H:%M:%SZ') for ref ${REF}"
    exit 0
  fi

  log_error "Dispatch failed (curl_rc=${curl_rc}, http_status=${dispatch_status:-unknown})"

  if ((retry == max_retries)); then
    die "workflow_dispatch failed after ${max_retries} retries"
  fi

  log "Waiting ${retry_sleep_seconds}s before checking for an accepted run"
  sleep "${retry_sleep_seconds}"

  log "Checking whether the previous dispatch created a workflow run after ${attempt_started_at}"
  recent_run_id="$(find_recent_dispatch "${attempt_started_at}")" || check_rc=$?
  check_rc="${check_rc:-0}"
  if ((check_rc != 0)); then
    die "cannot verify whether the previous dispatch created a run; refusing to retry"
  fi
  if [[ -n "${recent_run_id}" ]]; then
    log "Previous dispatch created workflow run ${recent_run_id}; refusing duplicate retry"
    exit 0
  fi

  log "No matching workflow run was found; retrying dispatch"
  unset curl_rc check_rc recent_run_id
done
