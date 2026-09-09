---
title: 发布与同步
weight: 30
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
# Release and Sync
-->
# 发布与同步

<!--
## `release.yaml` — build and publish wheels

**Trigger:** `push` of a tag matching `v[0-9]+.[0-9]+.[0-9]+` (stable
releases only; dev, rc, and other pre-release tags do not match).

- `build-wheels` builds the Python wheel inside the `manylinux_2_28_x86_64`
  container and uploads it as a workflow artifact.
- `publish` downloads that artifact and publishes it to PyPI using
  [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC via
  `id-token: write`, no stored PyPI token).

See [Versioning](/FlagGems/zh-cn/release/versioning/) for the tag-naming rules
that determine what triggers this workflow, and
[Packaging](/FlagGems/zh-cn/release/packaging/) for build details.
-->
## `release.yaml` — 构建并发布 wheel 包

**触发条件：** 推送匹配 `v[0-9]+.[0-9]+.[0-9]+` 格式的标签（仅稳定版发布；
dev、rc 等预发布标签不会匹配该规则）。

- `build-wheels` 在 `manylinux_2_28_x86_64` 容器内构建 Python wheel 包，
  并作为工作流构件（artifact）上传。
- `publish` 下载该构件，并使用
  [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) 机制
  （通过 `id-token: write` 实现 OIDC 认证，无需存储 PyPI token）将其发布到
  PyPI。

标签命名规则决定了哪些标签会触发该工作流，详见[版本管理](/FlagGems/zh-cn/release/versioning/)；
构建细节请参阅[打包](/FlagGems/zh-cn/release/packaging/)。

<!--
## `hugo-site.yaml` — build and publish documentation

**Trigger:** `push` to `gh-pages`; `workflow_run` after `coverage`
completes; also `workflow_call` (invoked by `unittest.yaml`'s `build-doc`
job when a merged PR carries the `documentation` label) and manual
`workflow_dispatch`.

Builds the Hugo site from `docs/` and publishes it to GitHub Pages.
`concurrency` is set to serialize deployments (`cancel-in-progress: false`)
so publishes don't race each other.
-->
## `hugo-site.yaml` — 构建并发布文档站点

**触发条件：** 推送到 `gh-pages` 分支；`coverage` 工作流执行完成后触发
（`workflow_run`）；此外也可通过 `workflow_call` 被调用（当一个带有
`documentation` 标签的 PR 被合并时，由 `unittest.yaml` 中的 `build-doc`
作业触发），以及手动触发（`workflow_dispatch`）。

从 `docs/` 目录构建 Hugo 站点并发布到 GitHub Pages。`concurrency`
配置为串行化部署（`cancel-in-progress: false`），避免多次发布相互抢占、
产生竞态。

<!--
## `sync-docs.yaml` — rebase docs branch onto master

**Trigger:** `workflow_call` only (invoked by `unittest.yaml`).

Rebases the `gh-pages` branch onto the latest `master` and force-pushes,
keeping documentation content in sync with code changes without waiting
for a full site rebuild.
-->
## `sync-docs.yaml` — 将文档分支 rebase 到 master

**触发条件：** 仅支持 `workflow_call`（由 `unittest.yaml` 调用）。

将 `gh-pages` 分支 rebase 到最新的 `master` 分支并强制推送，使文档内容
与代码变更保持同步，而不必等待完整的站点重新构建。

<!--
## `auto-sync.yaml` — mirror to other remotes

**Trigger:** `push` to `master`; also manual `workflow_dispatch`.

Mirrors the repository to `gitcode.com` and `code.gitlink.org.cn` using a
shared reusable workflow
(`flagos-ai/build-infra/.github/workflows/sync-to-remote.yaml`), so the
project stays available on domestic Git hosting mirrors.
-->
## `auto-sync.yaml` — 镜像同步到其他远端

**触发条件：** 推送到 `master` 分支；此外支持手动触发
（`workflow_dispatch`）。

使用共享的可复用工作流
（`flagos-ai/build-infra/.github/workflows/sync-to-remote.yaml`），将仓库
镜像同步到 `gitcode.com` 和 `code.gitlink.org.cn`，确保项目在国内 Git
托管平台上同样保持可用。

<!--
## Other operational workflows

- **`probe-node.yaml`** — manual `workflow_dispatch` to check a self-hosted
  runner (e.g. `h100`, `a100`) is online and healthy.
- **`random-test.yaml`** — manual `workflow_dispatch` to run an arbitrary
  set of operators against arbitrary GPUs on a chosen runner, for ad-hoc
  investigation.
- **`review-stats.yaml`** — manual `workflow_dispatch` to generate PR review
  statistics for a date range.
-->
## 其他运维类工作流

- **`probe-node.yaml`** —— 手动触发（`workflow_dispatch`），用于检查某个
  自托管 runner（例如 `h100`、`a100`）是否在线且状态正常。
- **`random-test.yaml`** —— 手动触发（`workflow_dispatch`），可在指定 runner
  上针对任意选定的算子和 GPU 组合运行测试，用于临时排查问题。
- **`review-stats.yaml`** —— 手动触发（`workflow_dispatch`），用于生成
  指定日期范围内的 PR 评审统计数据。
