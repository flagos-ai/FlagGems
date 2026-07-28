---
title: Versioning
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


# Versioning

FlagGems uses [setuptools-scm](https://github.com/pypa/setuptools-scm) to
auto-generate version numbers from git tags following
[PEP 440](https://peps.python.org/pep-0440/).

## Version Format

| Scenario | Example version |
|----------|-----------------|
| On a release tag `v5.4.0` | `5.4.0` |
| N commits after dev tag `v5.4.0.dev0` | `5.4.0.devN+g<hash>` |
| On a dev tag `v5.4.0.dev0` | `5.4.0.dev0` |

**No version is hardcoded in any source file.** The version is derived entirely
from git tags at build time.

The `+g<hash>` suffix (local version label) is included in local and editable
installs, allowing developers to `git checkout <hash>` for debugging. This
suffix is automatically stripped when publishing to PyPI.

## Tag Naming Rules

| Tag pattern | Purpose | Triggers release CI? |
|-------------|---------|---------------------|
| `v5.4.0` | Stable release | ✅ Yes |
| `v5.4.0.dev0` | Start of dev cycle | ❌ No |
| `v5.4.0rc1` | Release candidate | ❌ No |
| `v5.4.0.post1` | Post-release fix | ✅ Yes |

## Development Cycle

After each stable release, a dev tag is created to mark the start of the next
version cycle:

```
v5.3.0          ← stable release
  │
  ├── commit 1  ← 5.4.0.dev1+gabcdef0
  ├── commit 2  ← 5.4.0.dev2+g1234567
  ├── ...
  ├── commit N  ← 5.4.0.devN+g<hash>
  │
v5.4.0          ← next stable release
  │
v5.5.0.dev0     ← start of next dev cycle
```

## Release Process

### Releasing a new version (e.g. v5.4.0)

1. **Ensure all PRs for the release are merged into `master`.**

2. **Tag the release:**
   ```bash
   git checkout master
   git pull origin master
   git tag v5.4.0
   git push origin v5.4.0
   ```

3. **CI builds release artifacts automatically.**
   The `release.yaml` workflow triggers on stable version tags
   (`v<major>.<minor>.<patch>`) and builds the wheel, then publishes to PyPI.

4. **Start the next dev cycle:**
   ```bash
   git tag v5.5.0.dev0
   git push origin v5.5.0.dev0
   ```
   From this point, all commits on `master` produce versions like
   `5.5.0.dev1+gabcdef0`.

### Releasing a patch (e.g. v5.4.1)

1. Create a release branch if needed:
   ```bash
   git checkout -b release/5.4 v5.4.0
   ```
2. Cherry-pick or merge fixes.
3. Tag and push:
   ```bash
   git tag v5.4.1
   git push origin v5.4.1
   ```

### Release candidates

Tag as `v5.4.0rc1`, `v5.4.0rc2`, etc. These are PEP 440 pre-releases and
will **not** trigger the release workflow (it only matches stable tags). To
build RC wheels, trigger the workflow manually or adjust the tag filter
temporarily.

## Checking the Current Version

```bash
# From a git checkout (diagnostic, not required for install):
python -m setuptools_scm

# From an installed package:
python -c "import flag_gems; print(flag_gems.__version__)"
```
