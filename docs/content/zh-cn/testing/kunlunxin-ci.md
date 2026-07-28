---
title: 昆仑芯 CI 流水线
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

# 昆仑芯 CI 流水线

本文说明昆仑芯 P800 上的 PR 测试、按需测试、随机测试和 weekly 测试，
以及它们共同使用的环境和依赖解析规则。

## 1. 依赖兼容策略

昆仑芯后端在 `src/flag_gems/backends.yaml` 中固定使用 Python 3.10。
NumPy 2.3 及更高版本要求 Python 3.11 或更高版本，因此不能在项目依赖中将
NumPy 精确固定到 2.3.x。`pyproject.toml` 只声明 `numpy`，由安装器根据当前
Python 和其他后端包的约束选择兼容版本。当前 Python 3.10 会解析到 2.2.x
系列，Python 3.11 及以上环境可以解析到 2.3.x 或更新版本。

不要在昆仑芯 extra 中重复固定 NumPy。FlagGems 的基础依赖由所有后端共享，
其他 Python 3.10 后端可能还要求 NumPy 1.x；无版本固定可以让解析器同时考虑
这些后端包的约束。

## 2. PR 测试

PR 指向 `master` 后，流水线按下列顺序执行：

1. `.github/workflows/triage.yaml` 根据 `.github/labeler.yml` 给昆仑芯代码变更添加
   `vendor/Kunlunxin` 标签。
2. `.github/workflows/unittest.yaml` 的 `preprocess` 等待 triage 完成，收集变更文件
   和标签，并从 `.github/backends.json` 生成后端矩阵。昆仑芯矩阵项使用
   `backend=kunlunxin` 和 `runner_label=kunlunxin`。
3. `backend-tests` 调用 `.github/workflows/backend-test.yaml`，在昆仑芯自托管
   runner 上检出代码，并调用 `.github/actions/setup-flaggems`。
4. setup action 执行 `./setup.sh kunlunxin`。脚本读取
   `src/flag_gems/backends.yaml`，通过 uv 安装 Python 3.10、创建 `.venv`、加载
   `tools/env.sh`、安装构建依赖、`.[kunlunxin]`、FlagTree/Triton 和测试依赖。
5. `tools/gpu_check_kunlunxin.sh` 检查 P800 和可用显存，将可用卡号写入
   `AVAILABLE_GPUS`。
6. `tools/test-op.sh` 根据 `CHANGED_FILES` 执行受影响的 `tests/test*.py`、CPU quick
   对照测试和 `benchmark/test*`。PR 模式使用 `pytest -x`，首个失败即退出。

`linter.yml` 是独立的必跑路径：它在 GitHub 托管 runner 的 Python 3.11 环境中
执行 pre-commit，不使用昆仑芯依赖环境。

## 3. 其他触发方式

- PR 评论 `/test <op>:kunlunxin` 触发 `command.yaml`。它解析 runner 后，同样调用
  setup action 创建 Python 3.10 环境，再执行指定算子的精度和性能测试。
- 手动触发 `random-test.yaml` 并选择 `runner=kunlunxin` 时，也调用同一个 setup
  action，最后由 `tools/run_tests.py` 执行输入的算子和卡号。
- `daily.yaml` 当前只在 NVIDIA runner 上运行，不覆盖昆仑芯。

## 4. Weekly 测试

`.github/workflows/weekly.yaml` 可手动触发，也按 cron 定时触发。`prepare` 读取
`.github/configs/weekly/weekly-test.yaml` 和每个设备的配置文件，生成容器及原生
runner 矩阵。P800 配置位于 `.github/configs/weekly/P800.yml`，使用
`kunlunxin-flaggems-test-p800:20260526` 容器和 8 张 P800。

P800 属于容器矩阵，其执行链为：

1. 最多重试三次检出代码，并恢复容器内的 conda/login-shell 环境。
2. 直接执行 `pip install .`。这条路径不调用 `setup.sh`，因此 Numpy 兼容性必须
   在 `pyproject.toml` 的共享依赖中保证。
3. 执行 `tools/gpu_check_kunlunxin.sh`。
4. 清理 Triton/FlagGems 缓存，通过 `tools/run_tests.py` 执行全部 stage，或执行
   workflow 输入指定的算子。
5. 生成 CSV/HTML 汇总，打包结果并上传到 op-monitor，最后发送飞书通知。

## 5. 本地等价检查

在没有 P800 的机器上，可以验证配置、包元数据和 Python 3.10 依赖解析：

```shell
python -m build --wheel --no-isolation
python -m pip install --dry-run --ignore-installed --report /tmp/flaggems-py310.json .
python -m pip check
pre-commit run --all-files
```

在 P800 runner 或同版本容器中执行完整 PR 路径：

```shell
./setup.sh kunlunxin
source .venv/bin/activate
tools/gpu_check_kunlunxin.sh
CHANGED_FILES="tests/test_<op>.py benchmark/test_<op>.py" \
  tools/test-op.sh local
```

执行 weekly 等价路径时，应使用 P800 配置中指定的容器，然后运行：

```shell
pip install .
tools/gpu_check_kunlunxin.sh
python3 tools/run_tests.py --gpus 0,1,2,3,4,5,6,7 \
  --stages all --dump-output --output /tmp/flaggems-weekly
```

提交前应确认 Python 3.10 环境中的 `python -c "import numpy; print(numpy.__version__)"`
输出低于 2.3，并同时保留 Python 3.11 及以上环境对 Numpy 2.3+ 的解析能力。
