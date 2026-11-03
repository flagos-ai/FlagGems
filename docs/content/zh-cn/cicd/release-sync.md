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

See [Versioning](../release/versioning.md) for the tag-naming rules
that determine what triggers this workflow, and
[Packaging](../release/packaging.md) for build details.
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

标签命名规则决定了哪些标签会触发该工作流，详见[版本管理](../release/versioning.md)；
构建细节请参阅[打包](../release/packaging.md)。

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
## `build-deb.yml` and `build-rpm.yml` — distribution packages

**Trigger:** `pull_request` against `master` that touches
`packaging/debian/**` (or `packaging/rpm/**`) or
`packaging/verify-installed-package.py`; also manual `workflow_dispatch`.

These workflows only verify that the packaging definition still builds and
installs; they are not a release path. Packages published to the FlagOS
repositories are built and uploaded by `flagos-ai/build-infra` from a pinned
ref, so no tag triggers a build here. The `pull_request` path filter is
deliberate: a source-only PR has no consumer for a `.deb` or `.rpm`, so
those changes are covered by the repository's own tests instead.

Each workflow builds its package in a `manylinux_2_28_x86_64` container,
then smoke-tests the result by installing it into a clean `ubuntu:22.04`
(`.deb`) or `fedora:43` (`.rpm`) image and running
`packaging/verify-installed-package.py`. That step exists because a package
can install cleanly and still be broken: a bare `Depends:`/`Requires:` can
resolve to a too-old distro version, and a subpackage missing an
`__init__.py` is silently dropped by `find_packages()`. Both only surface at
import time, long after `dpkg -l` or `rpm -q` reports success.

Artifacts are uploaded as `flag-gems-amd64-packages` and
`flag-gems-amd64-rpm-packages` (7-day retention).

## Other operational workflows

The manually dispatched tools that do not touch releases — `probe-node.yaml`,
`random-test.yaml`, `review-stats.yaml`, plus the comment-triggered
`fix-sort.yaml` — are documented in
[Scheduled and on-demand testing](scheduled.md), which
owns everything you start by hand.
-->
## `build-deb.yml` 与 `build-rpm.yml` —— 发行版安装包

**触发条件：** 针对 `master` 且改动了 `packaging/debian/**`
（或 `packaging/rpm/**`）、`packaging/verify-installed-package.py` 的
`pull_request`；此外支持手动触发（`workflow_dispatch`）。

这两个工作流只验证打包定义仍能构建并安装，不是发布路径：发布到 FlagOS
软件仓库的安装包由 `flagos-ai/build-infra` 按固定的 ref 统一构建并上传，
因此这里不再由标签触发构建。`pull_request` 的路径过滤是有意为之：纯粹的
源码 PR 没有 `.deb` 或 `.rpm` 的使用方，这类改动由仓库自身的测试覆盖即可。

两个工作流均在 `manylinux_2_28_x86_64` 容器内构建安装包，随后把产物安装到
干净的 `ubuntu:22.04`（`.deb`）或 `fedora:43`（`.rpm`）镜像中运行
`packaging/verify-installed-package.py` 做冒烟测试。这一步很有必要，因为
安装成功并不代表可用：未加版本约束的 `Depends:`/`Requires:` 可能解析到发行版
中过旧的版本，而缺少 `__init__.py` 的子包会被 `find_packages()` 静默丢弃。
这两种问题都要到导入时才暴露，而那时 `dpkg -l` 或 `rpm -q` 早已显示安装成功。

构件分别以 `flag-gems-amd64-packages` 和 `flag-gems-amd64-rpm-packages`
上传，保留 7 天。

## 其他运维类工作流

不涉及发布、需要手动触发的工具类工作流 —— `probe-node.yaml`、
`random-test.yaml`、`review-stats.yaml`，以及由评论触发的
`fix-sort.yaml` —— 统一记录在
[定时与按需测试](scheduled.md)，凡是需要手动启动的
工作流都归那一页。
