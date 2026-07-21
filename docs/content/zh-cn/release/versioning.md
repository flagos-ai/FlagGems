---
title: 版本管理
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
# Versioning

FlagGems uses [setuptools-scm](https://github.com/pypa/setuptools-scm) to
auto-generate version numbers from git tags following
[PEP 440](https://peps.python.org/pep-0440/).
-->
# 版本管理

FlagGems 使用 [setuptools-scm](https://github.com/pypa/setuptools-scm)
从 git 标签自动生成符合 [PEP 440](https://peps.python.org/pep-0440/) 规范的版本号。

<!--
## Version Format
-->
## 版本格式

<!--
| Scenario | Example version |
|----------|-----------------|
| On a release tag `v5.4.0` | `5.4.0` |
| N commits after dev tag `v5.4.0.dev0` | `5.4.0.devN+g<hash>` |
| On a dev tag `v5.4.0.dev0` | `5.4.0.dev0` |
-->
| 场景 | 版本号示例 |
|------|-----------|
| 位于发布标签 `v5.4.0` | `5.4.0` |
| 在开发标签 `v5.4.0.dev0` 之后的第 N 次提交 | `5.4.0.devN+g<hash>` |
| 位于开发标签 `v5.4.0.dev0` | `5.4.0.dev0` |

<!--
**No version is hardcoded in any source file.** The version is derived entirely
from git tags at build time.

The `+g<hash>` suffix (local version label) is included in local and editable
installs, allowing developers to `git checkout <hash>` for debugging. This
suffix is automatically stripped when publishing to PyPI.
-->
**源码中不硬编码任何版本号。** 版本号完全在构建时从 git 标签派生。

`+g<hash>` 后缀（本地版本标签）会包含在本地安装和可编辑安装中，
开发者可以通过 `git checkout <hash>` 定位到对应的提交进行调试。
发布到 PyPI 时，该后缀会被自动去除。

<!--
## Tag Naming Rules
-->
## 标签命名规则

<!--
| Tag pattern | Purpose | Triggers release CI? |
|-------------|---------|---------------------|
| `v5.4.0` | Stable release | ✅ Yes |
| `v5.4.0.dev0` | Start of dev cycle | ❌ No |
| `v5.4.0rc1` | Release candidate | ❌ No |
| `v5.4.0.post1` | Post-release fix | ✅ Yes |
-->
| 标签格式 | 用途 | 是否触发发布 CI？ |
|----------|------|-------------------|
| `v5.4.0` | 稳定版发布 | ✅ 是 |
| `v5.4.0.dev0` | 开发周期开始 | ❌ 否 |
| `v5.4.0rc1` | 候选版本 | ❌ 否 |
| `v5.4.0.post1` | 修订版发布 | ✅ 是 |

<!--
## Development Cycle

After each stable release, a dev tag is created to mark the start of the next
version cycle:
-->
## 开发周期

每次稳定版发布后，创建一个开发标签来标记下一个版本周期的开始：

```
v5.3.0          ← 稳定版发布
  │
  ├── commit 1  ← 5.4.0.dev1+gabcdef0
  ├── commit 2  ← 5.4.0.dev2+g1234567
  ├── ...
  ├── commit N  ← 5.4.0.devN+g<hash>
  │
v5.4.0          ← 下一个稳定版发布
  │
v5.5.0.dev0     ← 新的开发周期开始
```

<!--
## Release Process
-->
## 发布流程

<!--
### Releasing a new version (e.g. v5.4.0)

1. **Ensure all PRs for the release are merged into `master`.**

2. **Tag the release:**

3. **CI builds release artifacts automatically.**
   The `release.yaml` workflow triggers on stable version tags
   (`v<major>.<minor>.<patch>`) and builds the wheel, then publishes to PyPI.

4. **Start the next dev cycle:**
   From this point, all commits on `master` produce versions like
   `5.5.0.dev1+gabcdef0`.
-->
### 发布新版本（例如 v5.4.0）

1. **确保该版本的所有 PR 已合并到 `master` 分支。**

2. **打标签：**
   ```bash
   git checkout master
   git pull origin master
   git tag v5.4.0
   git push origin v5.4.0
   ```

3. **CI 自动构建发布产物。**
   `release.yaml` 工作流会在推送稳定版标签
   （`v<major>.<minor>.<patch>`）时触发，自动构建 wheel 并发布到 PyPI。

4. **启动下一个开发周期：**
   ```bash
   git tag v5.5.0.dev0
   git push origin v5.5.0.dev0
   ```
   此后 `master` 上的所有提交将生成 `5.5.0.dev1+gabcdef0` 格式的版本号。

<!--
### Releasing a patch (e.g. v5.4.1)

1. Create a release branch if needed.
2. Cherry-pick or merge fixes.
3. Tag and push.
-->
### 发布补丁版本（例如 v5.4.1）

1. 如需要，创建发布分支：
   ```bash
   git checkout -b release/5.4 v5.4.0
   ```
2. Cherry-pick 或合并修复。
3. 打标签并推送：
   ```bash
   git tag v5.4.1
   git push origin v5.4.1
   ```

<!--
### Release candidates

Tag as `v5.4.0rc1`, `v5.4.0rc2`, etc. These are PEP 440 pre-releases and
will **not** trigger the release workflow (it only matches stable tags). To
build RC wheels, trigger the workflow manually or adjust the tag filter
temporarily.
-->
### 候选版本

使用 `v5.4.0rc1`、`v5.4.0rc2` 等格式打标签。这些属于 PEP 440
预发布版本，**不会**触发发布工作流（仅匹配稳定版标签）。如需构建候选版本的
wheel，可手动触发工作流或临时调整标签过滤规则。

<!--
## Checking the Current Version
-->
## 查看当前版本

```bash
# 从 git 检出目录查看（诊断用途，安装时不需要）：
python -m setuptools_scm

# 从已安装的包查看：
python -c "import flag_gems; print(flag_gems.__version__)"
```
