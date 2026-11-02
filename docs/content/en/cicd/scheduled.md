---
title: Scheduled and On-Demand Testing
weight: 20
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Scheduled and On-Demand Testing

Beyond per-PR checks, FlagGems runs broader test sweeps on a schedule to
catch regressions that only show up across the full operator suite or across
vendors that aren't touched by a given PR.

## `daily.yaml` — full-suite regression on NVIDIA

**Trigger:** cron `3 16 * * *` (00:03 Beijing time, daily); also manual
`workflow_dispatch`.

Runs the entire operator test suite (not just changed operators) on the
NVIDIA runner (hosted on an **NVIDIA A100-40G** GPU server with the official upstream **Triton** compiler installed, rather than FlagTree):

- `cpp-op` — the full C++ extension test suite.
- `python-op` — `tools/test-op.sh` with `CHANGED_FILES=__ALL__`, which
  expands to every `tests/test*.py` file to test baseline compatibility on official Triton and Ampere architecture. Also collects coverage and
  uploads it as the `op-ut-coverage` artifact.
- `examples` — `tools/test-examples.sh` against all example/model tests.

## `coverage.yaml` — publish coverage to the docs site

**Trigger:** `workflow_run` after `daily` completes; also manual
`workflow_dispatch` with a `run_id` input.

Downloads the `op-ut-coverage` artifact produced by `daily.yaml`, unpacks
the HTML coverage report and summary markdown, commits them under
`docs/static/coverage/<date>` and `docs/content/en/references/test/unit/`,
and pushes directly to the `gh-pages` branch. `hugo-site.yaml` then rebuilds
and republishes the site in response to that push.

## `ops-test.yaml` — multi-vendor QA acceptance testing

**Trigger:** `workflow_dispatch` only, with optional `branch`, `vendors`,
`ops`, `upload_log`, and `send_feishu` inputs. GitHub Actions' native
`schedule` trigger isn't used — it was observed to delay or drop runs by
1-5 hours under load, which isn't acceptable for GPU test scheduling.
Instead, `.github/scripts/trigger_workflow.sh` is installed as a crontab job
on an always-on control-plane server; by default it fires `30 21 * * 3,6`
(Beijing time, Wednesday and Saturday) and calls the GitHub API to dispatch
this workflow, so GitHub still allocates the runners and executes it. The
script retries on failure and checks for a duplicate run before retrying, to
avoid double-triggering.

Runs the full (or a filtered) operator suite across every enabled vendor
backend for QA acceptance testing — of a scheduled sweep via the external
dispatcher, or of a specific branch/vendor/operator set via a manual run
before a release. The `prepare` job resolves the checkout ref (defaulting
to `master`) and builds the backend matrix from
`.github/configs/backends/config.yaml` plus one YAML file per backend,
filtered by the `vendors` input (case-insensitive, empty = all enabled
vendors); the `ops` input likewise narrows the run to specific operators
(empty = every stage via `--stages all`).

Backends are split into two execution modes:

- **Container-based** (`test-container`) — runs inside the backend's Docker
  image, for vendors whose SDK is distributed as a container. Job timeout
  is 1440 minutes (24h) to accommodate the longest vendor runs (e.g. Ascend
  910B).
- **Native** (`test-native`) — runs directly on the self-hosted runner
  without a container, with an 1080-minute (18h) job timeout.

Each matrix job retries checkout up to three times (with 30s backoff),
installs FlagGems, checks GPU availability, runs `tools/run_tests.py`
scoped by `--ops` or `--stages all`, summarizes results with
`add_labels`/`psum_text`/`psum_html`, packages them into a per-vendor
archive, and — unless disabled via the `upload_log`/`send_feishu` inputs —
uploads the archive to the internal op-monitor service and posts a Feishu
notification with the outcome.

## Manually dispatched utility workflows

The following workflows only have a `workflow_dispatch` trigger. They are run
by hand from the **Actions** tab (**Run workflow**) or with `gh workflow run`.

### `random-test.yaml` — run selected operators on a chosen runner

**Trigger:** `workflow_dispatch` with three required inputs:

| Input | Meaning | Example |
|---|---|---|
| `runner` | Runner label to run on | `a100`, `h100`, `h20`, `ascend`, `cann9`, `metax`, `maca381`, `enflame`, `cambricon` |
| `operators` | Comma-separated operator names | `add,mul,softmax` |
| `gpus` | Comma-separated GPU indices | `0,1` |

```bash
gh workflow run random-test.yaml \
  -f runner=a100 -f operators=add,mul,softmax -f gpus=0,1
```

The `check-permission` job first verifies that the triggering user belongs
to the `FlagGems` team; otherwise the test job is skipped. The `test-operator`
job then maps the runner label to a vendor and backend profile (for example
`a100`/`h100`/`h20` → `nvidia-cuda133`, `ascend`/`cann9` → `ascend-cann900`;
an unknown label is used as both values), checks out the code, runs the
vendor's GPU check script and `setup-flaggems`, and executes
`tools/run_tests.py --ops <operators> --gpus <gpus>`. The `results/`
directory is uploaded as the `random-test` artifact.

Do not put spaces in `operators` or `gpus`, since they are passed straight
onto the command line. To reproduce locally, run `tools/run_tests.py`
directly.

### `probe-node.yaml` — inspect a self-hosted runner's environment

**Trigger:** `workflow_dispatch` with one required input, `runner` (runner
label, default `h100`).

Collects environment information from a self-hosted runner so you can choose
the right backend profile for it (for example `nvidia-cuda133` vs
`nvidia-cuda128`). It needs only bash on the runner and does not check out
the repository. It reports:

- OS and Python: `uname -a`, `python3 --version`
- Container / virtualization: `/.dockerenv`, `/proc/1/cgroup`,
  `systemd-detect-virt`, PID 1
- Memory: `free -h` and the cgroup memory limit
- GPU: `nvidia-smi`, plus per-GPU total/used/free memory
- CUDA toolchain: `nvcc --version` and `/usr/local/cuda*`

The output goes to the job log and to the run's **Summary** page
(`$GITHUB_STEP_SUMMARY`). The probe is read-only and has no team-membership
check. The GPU and CUDA probes are NVIDIA-specific; on other vendors they
print "not found" while the remaining sections still apply.

### `review-stats.yaml` — pull request analytics report

**Trigger:** `workflow_dispatch` with two optional inputs:
`report_date_start` and `report_date_end`, both in `dd/MM/YYYY` format.

Runs on `ubuntu-latest` and uses
[`AlexSim93/pull-request-analytics-action`](https://github.com/AlexSim93/pull-request-analytics-action)
to analyze `flagos-ai/FlagGems` pull requests (review turnaround, activity,
and similar metrics) over the given date range. Working hours are configured
as 9:00–23:00 in the `Asia/Shanghai` time zone. The report is created as an
issue in the `flagos-ai/FlagGems` repository, so the job needs
`issues: write` (plus read access to contents, pull requests and statuses).

```bash
gh workflow run review-stats.yaml \
  -f report_date_start=01/09/2026 -f report_date_end=30/09/2026
```

## Comment-triggered PR commands

`command.yaml` (`/test <operator>:<runner>`) and `fix-sort.yaml` (`/fix-sort`)
are started by commenting on a pull request rather than by `workflow_dispatch`
or a schedule, so they are documented in
[Pull request checks](pull-request.md) together with the checks
they complement.

## Feishu and monitoring integrations

- `ci-report-feishu.yaml` reports every `rule-check` completion to a Feishu
  Bitable for tracking pass/fail trends over time. It is triggered by
  `workflow_run` (not `pull_request`), so it executes in the base repository
  and its secrets are available even for fork PRs; the run itself only needs
  `tools/ci_report_feishu.py` plus a token, so it uses a sparse checkout of
  that single path. Because the GitHub API can lag behind a finished run, the
  script sleeps 5s and then retries up to 3 times with 10s/20s backoff. Its
  per-PR role is described under `rule-check` in
  [Pull request checks](pull-request.md).
- `ops-test.yaml` and `command.yaml`'s failure paths send Feishu chat
  notifications via `.github/scripts/notify_feishu.py`.
- `ops-test.yaml` also uploads results to an internal "op-monitor" HTTP
  service for longer-term dashboards.

These are observability workflows; they do not gate merges.
