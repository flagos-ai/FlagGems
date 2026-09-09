---
title: 拉取请求检查
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

<!--
# Pull Request Checks

All workflows below live under `.github/workflows/`. Unless noted otherwise,
they target pull requests against the `master` branch.
-->
# 拉取请求检查

以下工作流均位于 `.github/workflows/` 目录下。除特别说明外，
均针对目标分支为 `master` 的拉取请求触发。

<!--
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
-->
## 1. `triage.yaml` — PR 打标签

**触发条件：** `pull_request_target` 事件，类型为 `opened`、`synchronize`、
`reopened`、`ready_for_review`。

`pull_request_target` 运行在基础仓库（base repository）的上下文中，因此即使是来自
fork 的 PR，它也有权限写入标签。该工作流会：

- 使用 `actions/labeler` 结合 `.github/labeler.yml` 配置，根据改动路径打上标签
  （例如 `vendor/NVIDIA`、`tests`、`ops/cpp`、`documentation`、`examples`、
  `ops/alpha` —— 其他工作流会依据这些标签决定是否运行）。
- 根据 diff 大小打上尺寸标签（`size/XSmall` … `size/XLarge`）。
- 当 PR 标题匹配特定活动时，添加 `competition` 或 `KernelGen` 标签。

`unittest.yaml` 会先等待 `triage.yaml` 执行完毕才读取标签，从而确保按标签
门控的作业在判断是否运行之前，标签已经打好。

<!--
## 2. `rule-check.yaml` — static rule checks

**Trigger:** `pull_request` on `opened`, `reopened`, `synchronize`.

Runs a set of lightweight Python scripts (under `tools/ci_checks/`) against
the operators changed in the PR, derived by diffing `conf/operators.yaml`
and related source files between the PR base and head:
-->
## 2. `rule-check.yaml` — 静态规则检查

**触发条件：** `pull_request` 事件，类型为 `opened`、`reopened`、`synchronize`。

针对 PR 中变更的算子（通过对比 PR 基线与头部提交中 `conf/operators.yaml`
及相关源码文件的差异得出），运行一系列轻量级 Python 脚本（位于
`tools/ci_checks/` 目录下）：

<!--
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
-->
| 作业 | 检查内容 |
|---|---|
| `derive-changed-operators` | 计算变更的算子/文件列表，供以下所有作业使用 |
| `check-operators-yaml` | 新增/变更的算子是否在 `conf/operators.yaml` 中正确注册 |
| `check-init-exports` | `__init__.py` 中导出的算子是否已注册，以及反向的一致性检查 |
| `check-kernelgen-tests` | 测试文件是否调用了 `use_gems()`（会绕过与参考实现的比较） |
| `check-operator-markers` | 每个算子是否都有对应的测试文件和匹配的 `@pytest.mark.<id>` 标记 |
| `check-aten-operators` | `operators.yaml` 中的 `for:` 字段是否为合法的 ATen 算子名称 |
| `check-api-logs` | （仅警告，始终以状态码 0 退出）算子是否遵循 API 日志规范 |
| `check-benchmark-coverage` | （仅警告，始终以状态码 0 退出）变更的算子是否有对应的性能测试文件 |

<!--
Most jobs are gated with `if: needs.derive-changed-operators.outputs.has_changes
== 'true'` and are **skipped** for PRs that don't touch operator-related
files (e.g. a docs-only or CI-only PR).

A `rule-check-required` job aggregates the results of all jobs above and
always runs (`if: always()`), treating skipped upstream jobs as passing.
Branch protection should require this aggregate job rather than the
individual jobs — requiring a job that gets skipped on some PRs leaves the
check permanently pending and blocks merges.
-->
上述大部分作业都通过 `if: needs.derive-changed-operators.outputs.has_changes
== 'true'` 进行门控，对于不涉及算子相关文件改动的 PR（例如纯文档或纯 CI
脚本改动），这些作业会被**跳过**。

`rule-check-required` 作业会汇总以上所有作业的结果，并始终运行
（`if: always()`），将被跳过的上游作业视为通过。分支保护规则应当要求这个
汇总作业，而不是逐个要求单独的子作业 —— 如果某个检查在部分 PR 上会被跳过，
将其设为 required 会导致该检查永久停留在 pending 状态，从而阻塞合并。

<!--
`ci-report-feishu.yaml` listens for `rule-check` completion (`workflow_run`)
and posts a summary to a Feishu Bitable, including for PRs from forks (since
`workflow_run` also executes in the base repo context with access to
secrets).
-->
`ci-report-feishu.yaml` 会监听 `rule-check` 的完成事件（`workflow_run`），
将结果汇总上报到飞书多维表格（Bitable），即使 PR 来自 fork 仓库也能正常上报
（因为 `workflow_run` 同样运行在基础仓库上下文中，具备访问 secrets 的权限）。

<!--
## 3. `linter.yml` — code style

**Trigger:** `pull_request` on `opened`, `edited`, `reopened`, `synchronize`;
also `push` to `master`.

Runs `pre-commit` (the same hooks contributors run locally via
`pre-commit install`) to check Python formatting and basic style rules.
-->
## 3. `linter.yml` — 代码风格检查

**触发条件：** `pull_request` 事件，类型为 `opened`、`edited`、`reopened`、
`synchronize`；此外推送到 `master` 分支时也会触发。

运行 `pre-commit`（与贡献者通过 `pre-commit install` 在本地执行的钩子相同），
检查 Python 代码格式和基本风格规范。

<!--
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
-->
## 4. `unittest.yaml` — 算子与后端测试

**触发条件：** `pull_request` 事件，类型为 `opened`、`synchronize`、
`reopened`、`labeled`、`unlabeled`；此外推送到 `master` 分支和手动触发
（`workflow_dispatch`）也会执行。

这是主要的测试入口。`preprocess` 作业首先等待 `triage.yaml` 执行完毕，
然后计算 PR 的变更文件列表、PR 编号、当前标签，以及按后端划分的测试矩阵
（来自 `.github/backends.json`，只保留 PR 上带有对应 `vendor/*` 标签的后端）。
其余所有作业均按**标签门控** —— 只有当 PR 带有匹配标签时才会运行：

<!--
| Job | Runs when label is present | What it does |
|---|---|---|
| `build-doc` | `documentation` (and PR merged) | Syncs docs via `sync-docs.yaml` |
| `cpp-op` | `ops/cpp` | Builds and runs the C++ extension test suite (`cpp-op-test.yaml`) |
| `python-op` | `tests` | Runs `tools/test-op.sh` on the changed operators (NVIDIA runner) |
| `examples` | `examples` | Runs `tools/test-examples.sh` (end-to-end model tests) |
| `backend-tests` | any `vendor/*` label | Runs `backend-test.yaml` once per matched backend/runner |
| `alpha-ops` | `ops/alpha` | Runs the experimental-ops test script on NVIDIA |
-->
| 作业 | 触发所需标签 | 功能说明 |
|---|---|---|
| `build-doc` | `documentation`（且 PR 已合并） | 通过 `sync-docs.yaml` 同步文档 |
| `cpp-op` | `ops/cpp` | 构建并运行 C++ 扩展测试套件（`cpp-op-test.yaml`） |
| `python-op` | `tests` | 对变更算子运行 `tools/test-op.sh`（NVIDIA runner） |
| `examples` | `examples` | 运行 `tools/test-examples.sh`（端到端模型测试） |
| `backend-tests` | 任意 `vendor/*` 标签 | 针对每个匹配的后端/runner 运行一次 `backend-test.yaml` |
| `alpha-ops` | `ops/alpha` | 在 NVIDIA 上运行实验性算子的测试脚本 |

<!--
`backend-test.yaml` (a reusable `workflow_call` workflow) checks out the PR,
runs the vendor's GPU-availability check, sets up the environment via the
`setup-flaggems` composite action, then runs `tools/test-op.sh` (or an
alternate `test_script` for `alpha-ops`) scoped to the PR's changed files.

Because these jobs are skipped when the corresponding label is absent, a
`unittest-required` gate job aggregates `preprocess`, `cpp-op`, `python-op`,
`examples`, `backend-tests`, and `alpha-ops` with `if: always()`, so branch
protection can require a single, always-present check instead of the
individual label-gated jobs.
-->
`backend-test.yaml`（一个可复用的 `workflow_call` 工作流）会检出 PR 代码、
运行对应厂商的 GPU 可用性检查、通过 `setup-flaggems` 组合动作搭建环境，
然后针对 PR 变更的文件运行 `tools/test-op.sh`（`alpha-ops` 场景下使用
另一个 `test_script`）。

由于这些作业在对应标签缺失时会被跳过，`unittest-required` 汇总作业会以
`if: always()` 聚合 `preprocess`、`cpp-op`、`python-op`、`examples`、
`backend-tests`、`alpha-ops` 的结果，使分支保护规则可以要求这一个始终存在
的检查，而不必逐个要求那些按标签门控的作业。

<!--
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
-->
## 5. 按需触发与评论触发的工作流

以下工作流响应 PR 下的 `issue_comment` 事件，任何具有写权限的用户都可以
通过评论触发：

- **`command.yaml`**（`/test <算子名>:<runner>`） —— 检出 PR 代码（包括来自
  fork 的 PR，通过 `github/command` 设置 `allow_forks: true` 实现），在指定
  runner 上运行指定算子的测试，并将结果（对于已有算子，还包括改动前后的
  对比）以 PR 评论的形式发布。
- **`fix-sort.yaml`**（`/fix-sort`） —— 使用来自可信 `master` 分支的
  `tools/ci_checks/sort_exports.py --fix` 脚本对 PR 分支执行修复，
  如有变更则自动提交并推送，同时评论说明处理结果。

<!--
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
-->
## 设置 Required 状态检查

在为 `master` 配置分支保护规则或仓库 Ruleset 时，对于存在条件判断
（`if:`）或按标签门控作业的工作流，应优先选择汇总（gate）作业，而不是
单独的子作业：

- `rule-check / rule-check-required`
- `unittest / unittest-required`
- `linter / code-style`（该作业未被门控，可以直接安全地设为 required）

如果将某个在部分 PR 上会被 GitHub Actions 合理跳过的作业设为 required，
该检查会永久停留在"Expected"状态，从而无限期阻塞 auto-merge —— 因为在
required-check 的判定逻辑中，"跳过（skipped）"并不等同于"通过（passed）"。
