---
title: QA 验收测试
weight: 60
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
# QA Acceptance Testing

`.github/workflows/ops-test.yaml` is *FlagGems*' full-range operator QA
acceptance workflow. It runs the accuracy and performance test suite across
every supported hardware backend and uploads the results to the op-monitor
dashboard. This page describes the test environment, the configuration
files that drive the workflow, and how to reproduce a run locally.
-->
# QA 验收测试

`.github/workflows/ops-test.yaml` 是 *FlagGems* 的全量算子 QA 验收测试工作流。
它会在所有已支持的硬件后端上运行精度与性能测试套件，并将结果上传到
op-monitor 看板。本文介绍该工作流的测试环境、测试配置，以及本地复现方法。

<!--
## 1. Test environment

Each backend runs inside a dedicated container on a self-hosted runner with
real accelerator hardware attached. The container image, mounted volumes,
and runtime options are all backend-specific so that each vendor's driver
stack and SDK are available inside the container.
-->
## 1. 测试环境

每个后端都在挂载了真实加速卡的自托管 runner 上，通过专属容器运行。
容器镜像、挂载的目录以及运行时选项均按后端单独配置，以保证容器内可以
正常使用各厂商自己的驱动栈与 SDK。

<!--
The backends currently enabled in `.github/configs/backends/config.yaml`
are:
-->
`.github/configs/backends/config.yaml` 中当前启用的后端如下：

| 配置文件 (`name`) | 厂商 (`vendor`) | 容器镜像 |
| --- | --- | --- |
| `H800` | Nvidia | `flagos-runtime-nvidia-cuda13.3` |
| `910B` | Ascend | `flagos-runtime-ascend-cann9.0.0` |
| `C550` | Metax | `flagos-runtime-metax-maca3.8.1.3` |
| `BW1000` | Hygon | `flagos-runtime-hygon-dtk26.04` |
| `MTT-S5000` | MThreads | `flagos-runtime-mthreads-musa5.2.0` |
| `BI-V150` | Iluvatar | `flaggems-test-iluvatar-bi-v150` |
| `PPU-ZW810E` | THead | `flaggems-test-thead-ppu-zw810e` |
| `MLU590-M9DE` | Cambricon | `flaggems-test-cambricon-mlu590-m9de` |
| `P800` | KunLunXin | `flaggems-test-kunlunxin-p800` |
| `W7900D` | AMD | `flaggems-test-amd-w7900d` |
| `S60` | Enflame | `vllm0.24.0-enflame-tops1.10.6` |
| `TX8110`（默认关闭） | TsingMicro | `flaggems-test-tsingmicro-tx8110` |

<!--
Every backend config also sets:
- `runner_labels`: the self-hosted runner label the job is pinned to.
- `container_volumes` / `container_options`: host mounts (`/data`, `/home`,
  `/tmp`) and container flags such as `--privileged`, `--ipc=host`, and the
  vendor-specific device/accelerator flag (e.g. `--gpus all` on Nvidia).
- `test_gpus` / `force_test_gpus`: which device IDs to test; some vendors
  (e.g. KunLunXin) skip device 0, and some force a fixed device list instead
  of relying on the availability probe.
- `gpu_check_script`: a per-vendor script (e.g. `tools/gpu_check_nvidia.sh`)
  that waits for a device with enough free memory before the run starts.
- `output_base_dir`: where result archives are written on the runner host.
-->
每个后端配置还会设置：
- `runner_labels`：该任务绑定的自托管 runner 标签。
- `container_volumes` / `container_options`：主机挂载目录（`/data`、`/home`、
  `/tmp`）以及容器参数，例如 `--privileged`、`--ipc=host`，以及厂商专属的
  设备/加速卡参数（如 Nvidia 上的 `--gpus all`）。
- `test_gpus` / `force_test_gpus`：要测试的设备编号；部分厂商（如 KunLunXin）
  会跳过编号为 0 的设备，部分厂商会强制使用固定的设备列表而不依赖可用性探测。
- `gpu_check_script`：各厂商专属的脚本（如 `tools/gpu_check_nvidia.sh`），
  会在测试开始前等待出现满足空闲显存要求的设备。
- `output_base_dir`：测试结果归档在 runner 主机上的存放位置。

<!--
Inside the container, the workflow installs *FlagGems* with the `test`
extra (`pip install .[test]`), which brings in `pytest`, `coverage`,
`pandas`, `openpyxl`, and other reporting dependencies declared in
`pyproject.toml`.
-->
在容器内，工作流会以 `test` extra 安装 *FlagGems*（`pip install .[test]`），
对应 `pyproject.toml` 中声明的 `pytest`、`coverage`、`pandas`、`openpyxl`
等用于测试与报告生成的依赖。

<!--
## 2. Test configuration

### Backend selection

`.github/configs/backends/config.yaml` is the master switch: it lists every
backend and whether it is `enabled`. The workflow's `prepare` job reads this
file, loads the matching `.github/configs/backends/<NAME>.yaml` for each
enabled backend, and builds the job matrix from it. A `workflow_dispatch`
input (`vendors`) can further narrow the matrix to a comma-separated subset
of vendor names (case-insensitive), e.g. `Ascend,Nvidia`.

### Operator selection

Operators come from `conf/operators.yaml`, where each operator entry
declares the `stages` (`alpha`, `beta`, `stable`, `removed`) it has reached.
`tools/run_tests.py` filters by `--stages` (default `stable`); the workflow
itself exposes this as the `ops` input:
- empty (default on scheduled/dispatch runs) -> `--stages all`, i.e. every
  operator stage.
- a comma-separated list of operator IDs (e.g. `abs,sum,add`) -> `--ops
  <list>`, testing only those operators.

Enflame (`S60`) is the one exception: instead of `--stages all` it reads a
fixed `--op-list-file` from `/data/flaggems/ops_list.txt` on the runner.

### Result publishing

Two `workflow_dispatch` inputs gate the side effects of a run:
- `upload_log`: leave empty to publish the result archive to op-monitor via
  `secrets.OPMON_URL` / `secrets.OPMON_UPLOAD_KEY`; set any value to skip
  the "Publish results to op-monitor" step.
- `send_feishu`: leave empty to send a Feishu notification via
  `.github/scripts/notify_feishu.py`; set any value to skip it.
-->
## 2. 测试配置

### 后端选择

`.github/configs/backends/config.yaml` 是总开关：它列出了全部后端以及
各自是否 `enabled`。工作流的 `prepare` job 会读取该文件，为每个启用的
后端加载对应的 `.github/configs/backends/<NAME>.yaml`，并据此构建 job
矩阵。`workflow_dispatch` 的 `vendors` 输入可以进一步将矩阵缩小到指定的
厂商子集（逗号分隔、大小写不敏感），例如 `Ascend,Nvidia`。

### 算子选择

算子列表来自 `conf/operators.yaml`，其中每个算子条目都声明了自己所处的
`stages`（`alpha`、`beta`、`stable`、`removed`）。`tools/run_tests.py`
通过 `--stages` 过滤（默认 `stable`）；工作流将其封装为 `ops` 输入：
- 留空（定时任务或手动触发不填写时的默认值）-> 对应 `--stages all`，
  即测试所有阶段的算子。
- 填写以逗号分隔的算子 ID 列表（如 `abs,sum,add`）-> 对应
  `--ops <list>`，仅测试指定算子。

Enflame（`S60`）是个例外：它不使用 `--stages all`，而是从 runner 上固定的
`/data/flaggems/ops_list.txt` 文件读取 `--op-list-file`。

### 结果发布

两个 `workflow_dispatch` 输入控制一次运行的副作用：
- `upload_log`：留空时会通过 `secrets.OPMON_URL` / `secrets.OPMON_UPLOAD_KEY`
  将结果归档发布到 op-monitor；填写任意值则跳过“Publish results to
  op-monitor”步骤。
- `send_feishu`：留空时会通过 `.github/scripts/notify_feishu.py` 发送飞书
  通知；填写任意值则跳过该步骤。

<!--
## 3. Reproducing a run locally

The workflow is just a thin wrapper around `tools/run_tests.py` plus a few
report-generation scripts, so the same steps can be run directly on any
machine with the target accelerator and driver stack installed.
-->
## 3. 本地复现方法

该工作流本质上是对 `tools/run_tests.py` 以及几个报告生成脚本的封装，
因此只要本机已安装好目标加速卡及其驱动/SDK，就可以直接执行相同的步骤
来复现测试。

<!--
1. Install *FlagGems* with the test extras:
-->
1. 以 `test` extra 安装 *FlagGems*：

```shell
pip install .[test]
```

<!--
2. Make sure the accelerator is idle and has enough free memory (the
   vendor's `gpu_check_script`, e.g. `tools/gpu_check_nvidia.sh`, waits for
   this automatically in CI). Identify which device IDs you want to test,
   e.g. `0,1,2,3`.

3. Clear any stale Triton/FlagGems cache, then run the test scheduler. This
   mirrors the "Run ops tests" step in the workflow:
-->
2. 确认加速卡处于空闲状态且有足够的空闲显存（CI 中由各厂商的
   `gpu_check_script`，例如 `tools/gpu_check_nvidia.sh`，自动等待）。
   确定要测试的设备编号，例如 `0,1,2,3`。

3. 清理可能存在的 Triton/FlagGems 缓存，然后运行测试调度器。以下命令
   与工作流中“Run ops tests”步骤一致：

```shell
rm -rf ~/.triton/cache && rm -rf ~/.flaggems

# Run every stable-stage operator on 4 devices, mirroring the "ops: all" case.
python3 tools/run_tests.py \
    --gpus 0,1,2,3 \
    --stages all \
    --dump-output \
    --output logs_results_$(date +%Y%m%d_%H%M)
```

<!--
   To reproduce the `ops` input instead of a full-stage run, pass
   `--ops` with a comma-separated operator list, matching the workflow's
   non-empty `ops` input:
-->
   如果想复现 `ops` 输入指定算子（而非全量阶段）的场景，可以改用
   `--ops` 参数传入逗号分隔的算子列表，对应工作流中非空的 `ops` 输入：

```shell
python3 tools/run_tests.py --gpus 0,1,2,3 --ops "abs,sum,add" --dump-output
```

<!--
4. Generate the same summary artifacts the CI job produces (operator labels,
   CSV, and HTML reports):
-->
4. 生成与 CI 任务相同的汇总产物（算子标签、CSV 以及 HTML 报告）：

```shell
OUTPUT_DIR=logs_results_<date>_<time>
python3 tools/add_labels.py "${OUTPUT_DIR}"
python3 tools/psum_text -o "${OUTPUT_DIR}/result.csv"  "${OUTPUT_DIR}"
python3 tools/psum_html -o "${OUTPUT_DIR}/result.html" "${OUTPUT_DIR}"
```

<!--
5. Inspect `${OUTPUT_DIR}/summary.json` for aggregated pass/fail results, or
   open `${OUTPUT_DIR}/result.html` for the human-readable report. Each
   operator also gets its own `<op_name>/accuracy_result.json` and
   `<op_name>/performance_result.json`, plus stdout/stderr logs when
   `--dump-output` is passed.
-->
5. 查看 `${OUTPUT_DIR}/summary.json` 获取汇总的通过/失败结果，或打开
   `${OUTPUT_DIR}/result.html` 查看可读性更好的报告。每个算子还会生成
   独立的 `<op_name>/accuracy_result.json` 和
   `<op_name>/performance_result.json`，若传入 `--dump-output`，还会生成
   对应的 stdout/stderr 日志。

<!--
To trigger the same run in CI instead of reproducing it locally, dispatch
the workflow manually (**Actions -> ops-test -> Run workflow**) and fill in
the inputs described above: `branch`, `vendors`, `ops`, `upload_log`, and
`send_feishu`.
-->
如果不在本地复现，而是想在 CI 中触发同样的运行，可以手动分发该工作流
（**Actions -> ops-test -> Run workflow**），并填写上文介绍的输入项：
`branch`、`vendors`、`ops`、`upload_log`、`send_feishu`。

<!--
See [Testing Python Operators](../unittest) for the lower-level `pytest`
options that `tools/run_tests.py` drives under the hood.
-->
关于 `tools/run_tests.py` 底层所调用的 `pytest` 选项，请参阅
[测试 Python 算子](unittest.md)。
