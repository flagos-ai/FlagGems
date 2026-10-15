---
title: Release and Sync
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

# Release and Sync

## `release.yaml` — build and publish wheels

**Trigger:** `push` of a tag matching `v[0-9]+.[0-9]+.[0-9]+` (stable
releases only; dev, rc, and other pre-release tags do not match).

- `build-wheels` builds the Python wheel inside the `manylinux_2_28_x86_64`
  container and uploads it as a workflow artifact.
- `publish` downloads that artifact and publishes it to PyPI using
  [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC via
  `id-token: write`, no stored PyPI token).

See [Versioning](/FlagGems/release/versioning/) for the tag-naming rules
that determine what triggers this workflow, and
[Packaging](/FlagGems/release/packaging/) for build details.

## `hugo-site.yaml` — build and publish documentation

**Trigger:** `push` to `gh-pages`; `workflow_run` after `coverage`
completes; also `workflow_call` (invoked by `unittest.yaml`'s `build-doc`
job when a merged PR carries the `documentation` label) and manual
`workflow_dispatch`.

Builds the Hugo site from `docs/` and publishes it to GitHub Pages.
`concurrency` is set to serialize deployments (`cancel-in-progress: false`)
so publishes don't race each other.

## `sync-docs.yaml` — rebase docs branch onto master

**Trigger:** `workflow_call` only (invoked by `unittest.yaml`).

Rebases the `gh-pages` branch onto the latest `master` and force-pushes,
keeping documentation content in sync with code changes without waiting
for a full site rebuild.

## `auto-sync.yaml` — mirror to other remotes

**Trigger:** `push` to `master`; also manual `workflow_dispatch`.

Mirrors the repository to `gitcode.com` and `code.gitlink.org.cn` using a
shared reusable workflow
(`flagos-ai/build-infra/.github/workflows/sync-to-remote.yaml`), so the
project stays available on domestic Git hosting mirrors.

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
[Scheduled and on-demand testing](/FlagGems/cicd/scheduled/), which owns
everything you start by hand.
