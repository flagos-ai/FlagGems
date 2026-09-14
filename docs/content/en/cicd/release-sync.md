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

## Other operational workflows

- **`probe-node.yaml`** — manual `workflow_dispatch` to check a self-hosted
  runner (e.g. `h100`, `a100`) is online and healthy.
- **`random-test.yaml`** — manual `workflow_dispatch` to run an arbitrary
  set of operators against arbitrary GPUs on a chosen runner, for ad-hoc
  investigation.
- **`review-stats.yaml`** — manual `workflow_dispatch` to generate PR review
  statistics for a date range.
