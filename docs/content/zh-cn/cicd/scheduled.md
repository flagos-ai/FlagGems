---
title: 定时与按需测试
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

<!--
# Scheduled and On-Demand Testing

Beyond per-PR checks, FlagGems runs broader test sweeps on a schedule to
catch regressions that only show up across the full operator suite or across
vendors that aren't touched by a given PR.
-->
# 定时与按需测试

除了针对每个 PR 的检查外，FlagGems 还会按计划运行更大范围的测试，
以捕获那些只在全量算子测试或某个 PR 未涉及的厂商后端上才会暴露的回归问题。

<!--
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
-->
## `daily.yaml` — NVIDIA 平台全量回归测试

**触发条件：** cron 表达式 `3 16 * * *`（每天北京时间 00:03）；此外支持手动
触发（`workflow_dispatch`）。

在 NVIDIA runner 上运行完整的算子测试套件（而非仅测试变更的算子）：

- `cpp-op` —— 完整的 C++ 扩展测试套件。
- `python-op` —— 以 `CHANGED_FILES=__ALL__` 运行 `tools/test-op.sh`，
  该设置会展开为 `tests/test*.py` 下的所有测试文件。同时收集覆盖率数据
  并作为 `op-ut-coverage` 构件（artifact）上传。
- `examples` —— 对所有示例/模型测试运行 `tools/test-examples.sh`。

<!--
## `coverage.yaml` — publish coverage to the docs site

**Trigger:** `workflow_run` after `daily` completes; also manual
`workflow_dispatch` with a `run_id` input.

Downloads the `op-ut-coverage` artifact produced by `daily.yaml`, unpacks
the HTML coverage report and summary markdown, commits them under
`docs/static/coverage/<date>` and `docs/content/en/references/test/unit/`,
and pushes directly to the `gh-pages` branch. `hugo-site.yaml` then rebuilds
and republishes the site in response to that push.
-->
## `coverage.yaml` — 将覆盖率发布到文档站点

**触发条件：** `daily` 工作流执行完成后触发（`workflow_run`）；此外支持手动
触发（`workflow_dispatch`），需提供 `run_id` 参数。

下载 `daily.yaml` 生成的 `op-ut-coverage` 构件，解压 HTML 覆盖率报告和
汇总 markdown 文件，将其提交到 `docs/static/coverage/<date>` 和
`docs/content/en/references/test/unit/` 目录下，并直接推送到 `gh-pages`
分支。`hugo-site.yaml` 随后会响应该推送，重新构建并发布站点。

<!--
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
-->
## `weekly.yaml` — 多厂商全量测试

**触发条件：** cron 表达式 `30 13 * * 3,6`（每周三、周六北京时间 21:30）；
此外支持手动触发（`workflow_dispatch`），可选参数包括 `branch`、`vendors`、
`ops`、`upload_log`、`send_feishu`。

针对所有已启用的厂商后端运行完整（或经过筛选）的算子测试套件，配置来自
`.github/configs/weekly/weekly-test.yaml` 以及每个后端对应的独立 YAML
文件。各后端按两种执行方式划分：

- **基于容器**（`test-container`） —— 在后端对应的 Docker 镜像内运行，
  适用于以容器形式分发 SDK 的厂商。由于全量测试可能耗时较长，Ascend（910B）
  的作业超时时间设置为 24 小时，其他厂商使用标准超时限制。
- **原生运行**（`test-native`） —— 直接在自托管 runner 上运行，不使用容器，
  适用于以这种方式配置的厂商。

每次运行都会安装 FlagGems、检查 GPU 可用性、运行 `tools/run_tests.py`
（通过 `--ops` 或 `--stages all` 参数控制测试范围）、使用
`psum_text`/`psum_html` 汇总结果，并（除非通过输入参数禁用）将打包后的
结果上传到内部 op-monitor 服务，同时发送包含结果的飞书通知。

<!--
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
-->
## `command.yaml` — 按需 `/test` 命令

**触发条件：** PR 下创建的 `issue_comment` 事件，且评论内容匹配
`/test <算子名>:<runner>` 格式。

允许拥有相应权限的任何 PR 参与者，针对单个算子在指定 runner 上发起临时测试
（例如 `/test constant_pad_nd:mthreads`），而无需等待该算子被打上 `vendor/*`
标签或等待完整的 CI 运行。对于已有测试历史的算子，会在 PR 改动前后分别运行
测试并发布对比结果；对于全新算子，则只发布单次运行的报告。测试结果（及完整
日志）会以 PR 评论和上传构件的形式提供。

<!--
## Feishu and monitoring integrations

- `ci-report-feishu.yaml` reports every `rule-check` completion to a Feishu
  Bitable for tracking pass/fail trends over time.
- `weekly.yaml` and `command.yaml`'s failure paths send Feishu chat
  notifications via `.github/scripts/notify_feishu.py`.
- `weekly.yaml` also uploads results to an internal "op-monitor" HTTP
  service for longer-term dashboards.

These are observability workflows; they do not gate merges.
-->
## 飞书与监控集成

- `ci-report-feishu.yaml` 会将每一次 `rule-check` 的执行结果上报到飞书
  多维表格（Bitable），用于跟踪长期的通过/失败趋势。
- `weekly.yaml` 和 `command.yaml` 的失败处理路径会通过
  `.github/scripts/notify_feishu.py` 发送飞书群消息通知。
- `weekly.yaml` 还会将结果上传到内部的 "op-monitor" HTTP 服务，用于生成
  长期监控看板。

这些均属于可观测性（observability）相关的工作流，并不会阻塞 PR 的合并。
