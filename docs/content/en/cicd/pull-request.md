---
title: Pull Request Checks
weight: 10
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

# Pull Request Checks

All workflows below live under `.github/workflows/`. Unless noted otherwise,
they target pull requests against the `master` branch.

## 1. `triage.yaml` — PR labeling

**Trigger:** `pull_request_target` on `opened`, `synchronize`, `reopened`,
`ready_for_review`.

`pull_request_target` runs in the context of the base repository, so it has
write access to labels even for PRs from forks. It:

- Applies path-based labels using `actions/labeler` and `.github/labeler.yml`
  (e.g. `vendor/NVIDIA`, `tests`, `ops/cpp`, `documentation`, `examples`,
  `ops/alpha` — the label set that other workflows key off of).
- Applies a size label (`size/XSmall` … `size/XLarge`) based on diff size.
- Adds a `competition` or `KernelGen` label when the PR title matches those
  programs.

`unittest.yaml` waits for `triage.yaml` to finish before reading labels, so
labels are always in place before label-gated jobs decide whether to run.

## 2. `rule-check.yaml` — static rule checks

**Trigger:** `pull_request` on `opened`, `reopened`, `synchronize`.

Runs a set of lightweight Python scripts (under `tools/ci_checks/`) against
the operators changed in the PR, derived by diffing `conf/operators.yaml`
and related source files between the PR base and head:

| Job | Enforces |
|---|---|
| `derive-changed-operators` | Computes the operator/file diff used by all jobs below |
| `check-operators-yaml` | New/changed operators are registered correctly in `conf/operators.yaml` |
| `check-init-exports` | Operators exported in `__init__.py` are registered and vice versa |
| `check-kernelgen-tests` | Test files don't call `use_gems()` (bypasses reference comparison) |
| `check-operator-markers` | Each operator has a test file and a matching `@pytest.mark.<id>` |
| `check-aten-operators` | `for:` entries in `operators.yaml` match valid ATen operator names |
| `check-api-logs` | *(warning-only, always exits 0)* operators follow the API logging convention |
| `check-benchmark-coverage` | *(warning-only, always exits 0)* changed operators have a benchmark file |

Most jobs are gated with `if: needs.derive-changed-operators.outputs.has_changes
== 'true'` and are **skipped** for PRs that don't touch operator-related
files (e.g. a docs-only or CI-only PR).

A `rule-check-required` job aggregates the results of all jobs above and
always runs (`if: always()`), treating skipped upstream jobs as passing.
Branch protection should require this aggregate job rather than the
individual jobs — requiring a job that gets skipped on some PRs leaves the
check permanently pending and blocks merges.

`ci-report-feishu.yaml` listens for `rule-check` completion (`workflow_run`)
and posts a summary to a Feishu Bitable, including for PRs from forks (since
`workflow_run` also executes in the base repo context with access to
secrets).

## 3. `linter.yml` — code style

**Trigger:** `pull_request` on `opened`, `edited`, `reopened`, `synchronize`;
also `push` to `master`.

Runs `pre-commit` (the same hooks contributors run locally via
`pre-commit install`) to check Python formatting and basic style rules.

## 4. `unittest.yaml` — operator and backend tests

**Trigger:** `pull_request` on `opened`, `synchronize`, `reopened`,
`labeled`, `unlabeled`; also `push` to `master` and manual
`workflow_dispatch`.

This is the main test entrypoint. A `preprocess` job first waits for
`triage.yaml`, then computes the PR's changed files, PR number, current
labels, and a per-backend test matrix (from `.github/backends.json`,
filtered to backends whose `vendor/*` label is present on the PR). All
other jobs are **label-gated** — a job only runs if the PR carries the
matching label:

| Job | Runs when label is present | What it does |
|---|---|---|
| `build-doc` | `documentation` (and PR merged) | Syncs docs via `sync-docs.yaml` |
| `cpp-op` | `ops/cpp` | Builds and runs the C++ extension test suite (`cpp-op-test.yaml`) |
| `python-op` | `tests` | Runs `tools/test-op.sh` on the changed operators (NVIDIA runner) |
| `examples` | `examples` | Runs `tools/test-examples.sh` (end-to-end model tests) |
| `backend-tests` | any `vendor/*` label | Runs `backend-test.yaml` once per matched backend/runner |
| `alpha-ops` | `ops/alpha` | Runs the experimental-ops test script on NVIDIA |

`backend-test.yaml` (a reusable `workflow_call` workflow) checks out the PR,
runs the vendor's GPU-availability check, sets up the environment via the
`setup-flaggems` composite action, then runs `tools/test-op.sh` (or an
alternate `test_script` for `alpha-ops`) scoped to the PR's changed files.

Because these jobs are skipped when the corresponding label is absent, a
`unittest-required` gate job aggregates `preprocess`, `cpp-op`, `python-op`,
`examples`, `backend-tests`, and `alpha-ops` with `if: always()`, so branch
protection can require a single, always-present check instead of the
individual label-gated jobs.

## 5. On-demand and comment-triggered workflows

These respond to `issue_comment` events on a PR, so anyone with write access
can trigger them by commenting:

- **`command.yaml`** (`/test <operator>:<runner>`) — checks out the PR
  (including forks, via `github/command` with `allow_forks: true`), runs the
  named operator's tests on the named runner, and posts the result (and a
  before/after comparison for existing operators) as a PR comment.
- **`fix-sort.yaml`** (`/fix-sort`) — runs `tools/ci_checks/sort_exports.py
  --fix` from the trusted `master` copy against the PR branch, commits and
  pushes the fix if anything changed, and comments the outcome.

## Setting required status checks

When configuring branch protection or a repository ruleset for `master`,
prefer the aggregate/gate jobs over individual sub-jobs wherever a workflow
has conditional (`if:`) or label-gated jobs:

- `rule-check / rule-check-required`
- `unittest / unittest-required`
- `linter / code-style` (not gated, safe to require directly)

Requiring a job that GitHub Actions may legitimately skip for some PRs
leaves that check in a permanent "Expected" state and blocks auto-merge
indefinitely, since GitHub's required-check logic does not treat "skipped"
as equivalent to "passed" for that purpose.
