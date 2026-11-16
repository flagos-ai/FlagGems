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

**Trigger:** `pull_request` on `opened`, `reopened`, `synchronize`; also
`push` to `master`.

A single job, `rule-check-required`, runs on `ubuntu-latest` (15-minute
timeout) and executes the steps below in order. It uses a plain
`actions/checkout` with full history, because the changed-operator step diffs
the PR base against its head.

First it runs `pre-commit`, the same hooks contributors run locally via
`pre-commit install`. On a PR only the files the PR changed are linted
(`--from-ref`/`--to-ref`); on a push to `master` the whole tree is linted. A
hook failure fails the job immediately.

On pull requests it then runs lightweight Python scripts from
`tools/ci_checks/`. `derive_changed_operators.py` maps the files changed
between the PR base and head to operator IDs using `conf/operators.yaml` and
`conf/deprecated_operators.yaml`, and the operator-scoped steps only run when
it finds changes:

| Step | Runs when | Enforces |
|---|---|---|
| `check-operators-yaml` | operators changed | New/changed operators are registered correctly in `conf/operators.yaml` |
| `check-init-exports` | every PR | Operators exported in `__init__.py` are registered and vice versa |
| `check-backend-imports` | every PR | Vendor backend code still imports names that exist in shared code (`utils/`, `runtime/`, `ops/`), so a rename can't break a vendor whose own CI never runs |
| `check-kernelgen-tests` | operators changed | Test files don't call `use_gems()` (bypasses reference comparison) |
| `check-operator-markers` | operators changed | Each operator has a test file and a matching `@pytest.mark.<id>` |
| `check-aten-operators` | operators changed | `for:` entries in `operators.yaml` match valid ATen operator names |
| `check-no-torch-fallback` | every PR | Lines added by the PR in operator code don't call PyTorch compute APIs (see below) |
| `check-api-logs` | operators changed | *(warning-only)* operators follow the API logging convention |
| `check-benchmark-coverage` | operators changed | *(warning-only)* changed operators have a benchmark file |

The blocking checks above (all except the last two) run with
`continue-on-error`, so one failure doesn't hide the others. A final
`Verify required checks passed` step then fails the job if any of them failed.
The two warning-only scripts always exit 0 and emit `::warning` annotations
instead.

`check-no-torch-fallback` checks only the lines a PR adds under `ops/`,
`fused/`, `experimental_ops/` and the vendor `ops/`, `fused/`, `fla/`
directories, so existing violations are not reported. It flags calls to
`torch.<op>()`, `torch.nn.functional.*` and `torch.ops.aten.*`. Allocation and
metadata helpers (`torch.empty`, `torch.zeros`, `torch.cuda.*`, ...) are
allowed. Append `# allow-torch-compute` to a line to exempt it; reviewers
should question every use. The same check runs locally as a `pre-commit` hook
that compares the working tree with `HEAD`; in CI that comparison is empty, so
the dedicated step above is what enforces it.

Steps that don't apply to a PR are skipped, not failed, and the job itself
always exists. Branch protection can therefore require
`rule-check / rule-check-required` directly.

`ci-report-feishu.yaml` listens for `rule-check` completion (`workflow_run`)
and posts a summary to a Feishu Bitable, including for PRs from forks (since
`workflow_run` also executes in the base repo context with access to
secrets).
-->
## 2. `rule-check.yaml` — 静态规则检查

**触发条件：** `pull_request` 事件，类型为 `opened`、`reopened`、`synchronize`；
此外推送到 `master` 分支时也会触发。

单个作业 `rule-check-required` 在 `ubuntu-latest` 上运行（超时 15 分钟），
按顺序执行下列步骤。它使用带完整历史的普通 `actions/checkout`，因为“变更算子”
步骤需要对比 PR 基线与头部提交。

首先运行 `pre-commit`，即贡献者通过 `pre-commit install` 在本地执行的同一组钩子。
PR 上只检查该 PR 改动的文件（`--from-ref`/`--to-ref`）；推送到 `master` 时检查
整个代码树。钩子失败会让作业立即失败。

在 PR 上，随后运行 `tools/ci_checks/` 下的轻量级 Python 脚本。
`derive_changed_operators.py` 依据 `conf/operators.yaml` 和
`conf/deprecated_operators.yaml`，把 PR 基线与头部提交之间变更的文件映射为算子
ID；与算子相关的步骤只有在检测到变更时才会运行：

| 步骤 | 运行时机 | 检查内容 |
|---|---|---|
| `check-operators-yaml` | 有算子变更 | 新增/变更的算子是否在 `conf/operators.yaml` 中正确注册 |
| `check-init-exports` | 每个 PR | `__init__.py` 中导出的算子是否已注册，以及反向的一致性检查 |
| `check-backend-imports` | 每个 PR | 厂商后端代码引用的名称在共享代码（`utils/`、`runtime/`、`ops/`）中仍然存在，避免重命名破坏自身 CI 不会运行的厂商后端 |
| `check-kernelgen-tests` | 有算子变更 | 测试文件是否调用了 `use_gems()`（会绕过与参考实现的比较） |
| `check-operator-markers` | 有算子变更 | 每个算子是否都有对应的测试文件和匹配的 `@pytest.mark.<id>` 标记 |
| `check-aten-operators` | 有算子变更 | `operators.yaml` 中的 `for:` 字段是否为合法的 ATen 算子名称 |
| `check-no-torch-fallback` | 每个 PR | PR 在算子代码中新增的行不得调用 PyTorch 计算接口（详见下文） |
| `check-api-logs` | 有算子变更 | （仅警告）算子是否遵循 API 日志规范 |
| `check-benchmark-coverage` | 有算子变更 | （仅警告）变更的算子是否有对应的性能测试文件 |

上述阻塞型检查（除最后两项外）都以 `continue-on-error` 运行，因此一项失败不会
掩盖其他项。随后由最后的 `Verify required checks passed` 步骤在其中任何一项
失败时让作业失败。两个仅警告的脚本始终以状态码 0 退出，只输出 `::warning` 注解。

`check-no-torch-fallback` 只检查 PR 在 `ops/`、`fused/`、`experimental_ops/`
以及各厂商后端 `ops/`、`fused/`、`fla/` 目录中新增的行，存量违规不会被报告。
它会拦截 `torch.<op>()`、`torch.nn.functional.*` 和 `torch.ops.aten.*` 调用；
`torch.empty`、`torch.zeros`、`torch.cuda.*` 等内存分配和元数据接口允许使用。
在行末添加 `# allow-torch-compute` 可豁免该行，评审时应对每处豁免提出质疑。
本地可通过 `pre-commit` 钩子运行同一检查，它将工作区与 `HEAD` 比较；在 CI 中该
比较结果为空，真正起拦截作用的是上面的独立步骤。

不适用于某个 PR 的步骤会被跳过而不是失败，且该作业本身始终存在，因此分支保护
规则可以直接要求 `rule-check / rule-check-required`。

`ci-report-feishu.yaml` 会监听 `rule-check` 的完成事件（`workflow_run`），
将结果汇总上报到飞书多维表格（Bitable），即使 PR 来自 fork 仓库也能正常上报
（因为 `workflow_run` 同样运行在基础仓库上下文中，具备访问 secrets 的权限）。

<!--
## 3. `unittest.yaml` — operator and backend tests

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
## 3. `unittest.yaml` — 算子与后端测试

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
| `python-op` | `tests` | 运行在 NVIDIA A100-40G 服务器上（安装官方 upstream triton 编译器而非 flagtree），对变更算子运行 `tools/test-op.sh` 进行单元测试 |
| `examples` | `examples` | 运行 `tools/test-examples.sh`（端到端模型测试） |
| `backend-tests` | 任意 `vendor/*` 标签 | 针对每个匹配的后端/runner 运行一次 `backend-test.yaml` |
| `alpha-ops` | `ops/alpha` | 在 NVIDIA 上运行实验性算子的测试脚本 |

#### `python-op` 运行环境与编译器选型

`python-op` 是验证公版算子正确性与兼容性的核心门禁作业：
- **硬件基础设施**：该作业部署并运行在标签为 `jiuding` 的专用自托管 Runner 上，底层硬件为配备 **NVIDIA A100-40G** GPU 的高性能服务器；
- **编译器选型**：在 `python-op` 测试环境中，**默认安装并使用的是官方上游原生 Triton 编译器，而非 FlagTree 编译器**。这一策略用于严格检验所有公版算子实现（位于 `src/flag_gems/ops`、`src/flag_gems/fused` 以及 `src/flag_gems/experimental_ops`）与官方社区 Triton 语义及 NVIDIA Ampere 架构的基线兼容性，防止依赖特定编译器衍生分支的代码合入公版目录。

<!--
`backend-test.yaml` (a reusable `workflow_call` workflow) checks out the PR
with `checkout-retry`, runs the vendor's GPU-availability check, installs
FlagGems (with `flagos-ai/FlagOps`'s `setup-project` action on native runners,
or a vendor-specific install step in the containerized jobs), then runs `tools/test-op.sh` (or an
alternate `test_script` for `alpha-ops`) scoped to the PR's changed files.

Seven vendors (Enflame, Hygon, Iluvatar, Kunlunxin, MetaX, MooreThreads,
T-Head) only have their driver/runtime available inside a vendor container, so
each has its own `test-<vendor>` job that runs inside that container. Those
jobs install FlagGems on top of the image's existing dependencies instead of
running `setup.sh`, which would rebuild a venv and reinstall torch. The container image, `docker run` options, volume mounts, and
runner labels all come from the `container` block of each backend's entry in
`.github/backends.json`: `unittest.yaml` reads them into the test matrix and
passes them through to `backend-test.yaml` as `workflow_call` inputs. This
keeps the container configuration in a single source of truth
(`.github/backends.json`) shared with the on-demand `command.yaml` workflow
below.

Because these jobs are skipped when the corresponding label is absent, a
`unittest-required` gate job aggregates `preprocess`, `cpp-op`, `python-op`,
`examples`, `backend-tests`, and `alpha-ops` with `if: always()`, so branch
protection can require a single, always-present check instead of the
individual label-gated jobs.

The `python-op` and `examples` jobs in `unittest.yaml`, plus the jobs in
`cpp-op-test.yaml` and `backend-test.yaml`, first run an independent
`actions/checkout` with `continue-on-error`, then call `checkout-retry`.
The composite action makes at most three attempts (one initial attempt and
two retries, waiting 30s before each retry), so these PR test jobs can make
up to four checkout attempts in total. `rule-check.yaml` does not retry; it
uses a plain `actions/checkout`. `ci-report-feishu.yaml` instead uses
`robust-checkout` from `flagos-ai/FlagOps` with `rounds: 5` (the action
defaults to 3). Each round tries SSH when a key is available, then HTTPS, and
stops after a successful fetch. Each link has a 10s connection probe and a 180s
hard fetch timeout. HTTP fetch also aborts after 20s below 20 KiB/s; SSH uses
keepalives every 5s and aborts after two unanswered probes. There is no fixed wait
between rounds. The round count counts link-rotation cycles, not checkout
retries; the four-attempt limit above does not apply to that workflow.
-->
`backend-test.yaml`（一个可复用的 `workflow_call` 工作流）会通过 `checkout-retry`
检出 PR 代码、运行对应厂商的 GPU 可用性检查、安装 FlagGems（原生 runner 上使用
`flagos-ai/FlagOps` 的 `setup-project` 动作，容器化作业中使用各厂商专用的安装步骤），
然后针对 PR 变更的文件运行 `tools/test-op.sh`（`alpha-ops` 场景下使用
另一个 `test_script`）。

其中七个厂商（Enflame、Hygon、Iluvatar、Kunlunxin、MetaX、MooreThreads、T-Head）
的驱动/运行时只在厂商容器内可用，因此每个厂商都有各自的 `test-<vendor>` 作业，
在对应容器中运行。这些作业在镜像已有依赖的基础上安装 FlagGems，而不是运行
`setup.sh`（后者会重建 venv 并重新安装 torch）。容器镜像、`docker run`
选项、卷挂载以及 runner 标签，全部来自 `.github/backends.json` 中各后端条目下的
`container` 配置块：`unittest.yaml` 将其读入测试矩阵，并作为 `workflow_call`
入参透传给 `backend-test.yaml`。这样容器配置就保存在单一数据源
（`.github/backends.json`）中，并与下文的按需工作流 `command.yaml` 共用。

由于这些作业在对应标签缺失时会被跳过，`unittest-required` 汇总作业会以
`if: always()` 聚合 `preprocess`、`cpp-op`、`python-op`、`examples`、
`backend-tests`、`alpha-ops` 的结果，使分支保护规则可以要求这一个始终存在
的检查，而不必逐个要求那些按标签门控的作业。

`unittest.yaml` 的 `python-op`、`examples`，以及 `cpp-op-test.yaml`、
`backend-test.yaml` 中的作业，会先独立执行一次允许失败的
`actions/checkout`，再调用 `checkout-retry`。该组合动作内部最多尝试检出
3 次（首次检出加最多 2 次重试，每次重试前等待 30 秒），因此这类 PR 测试
作业合计最多可尝试检出 4 次。

`rule-check.yaml` 不做重试，使用普通的 `actions/checkout`。
`ci-report-feishu.yaml` 则使用 `flagos-ai/FlagOps` 的 `robust-checkout`，并将
`rounds` 设为 5（动作默认值为 3）。每轮先尝试 SSH（有密钥时），再尝试 HTTPS；
任一链路成功就结束，密钥为空时跳过 SSH。每条链路的连接探测最多 10 秒，
fetch 的硬超时为 180 秒。HTTP fetch 若连续 20 秒低于 20 KiB/s 也会中止；
SSH 每隔 5 秒发送 keepalive，连续 2 次无响应则中止。
轮次之间没有固定等待。“5 轮”指链路轮转次数，并非重试检出 5 次；
上述 4 次检出上限不适用于该工作流。

<!--
## 4. On-demand and comment-triggered workflows

These respond to `issue_comment` events on a PR, so anyone with write access
can trigger them by commenting:

- **`command.yaml`** (`/test <operator>:<runner>`) — checks out the PR
  (including forks, via `github/command` with `allow_forks: true`), runs the
  named operator's tests on the named runner, and posts the result (and a
  before/after comparison for existing operators) as a PR comment.
- **`fix-sort.yaml`** (`/fix-sort`) — runs `tools/ci_checks/sort_exports.py
  --fix` from the trusted `master` copy against the PR branch, commits and
  pushes the fix if anything changed, and comments the outcome.

`command.yaml` has one generic job plus one job per containerized vendor
(Enflame, Hygon, MetaX, MooreThreads, Iluvatar, Kunlunxin, T-Head). Because the workflow is
triggered by `issue_comment` rather than `pull_request`, the default checkout
doesn't land on the PR branch, so each job checks it out explicitly via the
`gh pr checkout` in the `checkout-pr` action. The per-vendor jobs
resolve their container image, options, and runner labels from the same
`container` block in `.github/backends.json` (looked up in `preprocess`), so
the container configuration is not duplicated between `command.yaml` and
`backend-test.yaml`. Most jobs use the shared actions from
`flagos-ai/FlagOps` (`checkout-retry`, `checkout-pr`, `setup-project`,
`run-ondemand-test`). Only the Enflame job uses this repository's own
composite actions under `.github/actions/`:

| Composite action | Used for |
|---|---|
| `checkout-retry` | Check out the repository with up to 3 attempts, sleeping 30s between failures |
| `checkout-pr` | Install the `gh` CLI if missing (picking the binary matching the runner's CPU arch), then `gh pr checkout` the PR branch |
| `setup-flaggems` | Run `setup.sh` for the backend and the GPU-availability check |
| `ondemand-test` | Run the single-operator accuracy/perf comparison, upload the result artifact, and post the PR comment |
-->
## 4. 按需触发与评论触发的工作流

以下工作流响应 PR 下的 `issue_comment` 事件，任何具有写权限的用户都可以
通过评论触发：

- **`command.yaml`**（`/test <算子名>:<runner>`） —— 检出 PR 代码（包括来自
  fork 的 PR，通过 `github/command` 设置 `allow_forks: true` 实现），在指定
  runner 上运行指定算子的测试，并将结果（对于已有算子，还包括改动前后的
  对比）以 PR 评论的形式发布。
- **`fix-sort.yaml`**（`/fix-sort`） —— 使用来自可信 `master` 分支的
  `tools/ci_checks/sort_exports.py --fix` 脚本对 PR 分支执行修复，
  如有变更则自动提交并推送，同时评论说明处理结果。

`command.yaml` 包含一个通用作业，外加每个容器化厂商各一个作业（Enflame、Hygon、MetaX、
MooreThreads、Iluvatar、Kunlunxin、T-Head）。由于该工作流由 `issue_comment` 而非
`pull_request` 触发，默认的检出不会落在 PR 分支上，因此每个作业都通过
`checkout-pr` 动作里的 `gh pr checkout` 显式检出 PR 分支。各厂商作业从
`.github/backends.json` 中相同的 `container` 配置块解析容器镜像、选项和 runner
标签（在 `preprocess` 中查表），因而容器配置不会在 `command.yaml` 和
`backend-test.yaml` 之间重复。大多数作业使用 `flagos-ai/FlagOps` 提供的共享动作
（`checkout-retry`、`checkout-pr`、`setup-project`、`run-ondemand-test`）；只有
Enflame 作业使用本仓库 `.github/actions/` 下的组合动作：

| 组合动作 | 用途 |
|---|---|
| `checkout-retry` | 最多尝试 3 次检出仓库，每次失败后等待 30 秒 |
| `checkout-pr` | 若缺失则安装 `gh` CLI（按 runner 的 CPU 架构选择对应二进制），再 `gh pr checkout` 检出 PR 分支 |
| `setup-flaggems` | 针对该后端运行 `setup.sh` 及 GPU 可用性检查 |
| `ondemand-test` | 运行单算子精度/性能对比、上传结果产物并发布 PR 评论 |

<!--
## Setting required status checks

When configuring branch protection or a repository ruleset for `master`,
prefer the aggregate/gate jobs over individual sub-jobs wherever a workflow
has conditional (`if:`) or label-gated jobs:

- `rule-check / rule-check-required`
- `unit-test / unittest-required`

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
- `unit-test / unittest-required`

如果将某个在部分 PR 上会被 GitHub Actions 合理跳过的作业设为 required，
该检查会永久停留在"Expected"状态，从而无限期阻塞 auto-merge —— 因为在
required-check 的判定逻辑中，"跳过（skipped）"并不等同于"通过（passed）"。
