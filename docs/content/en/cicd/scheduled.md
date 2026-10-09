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
NVIDIA runner:

- `cpp-op` — the full C++ extension test suite.
- `python-op` — `tools/test-op.sh` with `CHANGED_FILES=__ALL__`, which
  expands to every `tests/test*.py` file. Also collects coverage and
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

## `weekly.yaml` — multi-vendor full-suite testing

**Trigger:** cron `30 13 * * 3,6` (21:30 Beijing time, Wednesday and
Saturday); also manual `workflow_dispatch` with optional `branch`,
`vendors`, `ops`, `upload_log`, and `send_feishu` inputs.

Runs the full (or a filtered) operator suite across every enabled vendor
backend, configured in `.github/configs/weekly/weekly-test.yaml` and one
YAML file per backend. Backends are split into two execution modes:

- **Container-based** (`test-container`) — runs inside the backend's Docker
  image, for vendors whose SDK is distributed as a container.
  Ascend (910B) gets a 24h job timeout; others get standard limits, since
  the whole-vendor run can take many hours.
- **Native** (`test-native`) — runs directly on the self-hosted runner
  without a container, for vendors set up that way.

Each run installs FlagGems, checks GPU availability, runs
`tools/run_tests.py` (scoped by `--ops` or `--stages all`), summarizes
results with `psum_text`/`psum_html`, and (unless disabled via input)
uploads the packaged results to the internal op-monitor service and posts a
Feishu notification with the outcome.

## `ops-test.yaml` — QA acceptance testing

**Trigger:** manual `workflow_dispatch` only, with optional `branch`,
`vendors`, `ops`, `upload_log`, and `send_feishu` inputs.

The on-demand counterpart to `weekly.yaml`, used by QA for acceptance testing
of a specific branch, vendor set, or operator list before a release. The
`prepare` job resolves the checkout ref (defaulting to `master`) and builds
the backend matrix from `.github/configs/backends/config.yaml` plus one YAML
file per backend, filtered by the `vendors` input (case-insensitive, empty =
all enabled vendors); the `ops` input likewise narrows the run to specific
operators (empty = every stage via `--stages all`).

Backends are split into the same two execution modes as `weekly.yaml`:

- **Container-based** (`test-container`) — runs inside the backend's Docker
  image. Job timeout is 1440 minutes (24h) to accommodate the longest vendor
  runs (e.g. Ascend 910B).
- **Native** (`test-native`) — runs directly on the self-hosted runner,
  with an 1080-minute (18h) job timeout.

Each matrix job retries checkout up to three times (with 30s backoff),
installs FlagGems, checks GPU availability, runs `tools/run_tests.py`
scoped by `--ops` or `--stages all`, summarizes results with
`add_labels`/`psum_text`/`psum_html`, packages them into a per-vendor
archive, and — unless disabled via the `upload_log`/`send_feishu` inputs —
uploads the archive to the internal op-monitor service and posts a Feishu
notification with the outcome.

## `command.yaml` — on-demand `/test` command

**Trigger:** `issue_comment` created on a PR, matching `/test
<operator>:<runner>`.

Lets any PR participant with the right permissions request an ad-hoc test
of a single operator on a specific runner (e.g. `/test
constant_pad_nd:mthreads`) without waiting for the operator to have a
`vendor/*` label or for a full CI run. For operators that already have
history, it runs the test before and after the PR's changes and posts a
comparison; for brand-new operators, it posts a single-run report. Results
(and full logs) are attached as a PR comment and an uploaded artifact.

## Feishu and monitoring integrations

- `ci-report-feishu.yaml` reports every `rule-check` completion to a Feishu
  Bitable for tracking pass/fail trends over time.
- `weekly.yaml`, `ops-test.yaml`, and `command.yaml`'s failure paths send
  Feishu chat notifications via `.github/scripts/notify_feishu.py`.
- `weekly.yaml` and `ops-test.yaml` also upload results to an internal
  "op-monitor" HTTP service for longer-term dashboards.

These are observability workflows; they do not gate merges.
