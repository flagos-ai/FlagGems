# FlagGems 持久项目记忆

> 最后核验：2026-07-20 UTC；工作区：`/workspace/FlagGems`。本文件是项目内持久上下文，不是事实数据库。每次会话先读本文，再用当前 Git 状态、代码和机器环境核验易变信息。

## 1. 新会话恢复约定

- Codex 在仓库内启动时会读取根目录 `AGENTS.md`；该文件明确要求先读本文件，因此这是跨会话自动恢复上下文的可靠入口。
- `.codex/memory/FlagGems.md` 本身不是 Codex CLI 保证自动扫描的系统文件；它依靠 `AGENTS.md` 入口加载。不要假定 `/root/.codex` 下任意 Markdown 会自动注入上下文。
- 旧资料位于 `/root/.claude/projects/-workspace-FlagGems/memory/`，属于 Claude 的项目记忆，Codex 不会自动读取。旧索引包括 project overview、operator dispatch、pointwise codegen、test infrastructure、XPU/CUDA mapping、failing ops 和 fix workflow。
- 会话开始建议依次检查：本文、`git status --short --branch`、`git log -5 --oneline --decorate`、目标算子代码/测试、相关结果文件。不要盲目执行旧记忆里的 `git pull`，以免覆盖或干扰用户工作。
- 已安装个人技能 `/root/.codex/skills/flaggems-operator-repair/`。涉及算子失败检测、诊断、修复、验证、文档或进度维护时使用 `$flaggems-operator-repair`；技能包含机器/分发参考、测试诊断流程、修复交付规则、历史 backlog、报告模板和只读工具。

## 2. 项目定位与架构

FlagGems 是 FlagOS 体系内用 Triton/FlagTree 实现的跨厂商高性能算子库。它通过 PyTorch `torch.library.Library("aten", "IMPL")` 注册 ATen 实现，使普通 PyTorch API 在 `flag_gems.enable()`、`only_enable()` 或 `use_gems()` 作用域内分发到 FlagGems kernel。

关键目录：

- `src/flag_gems/ops/`：通用算子实现。
- `src/flag_gems/fused/`：融合算子。
- `src/flag_gems/runtime/backend/_<vendor>/`：厂商能力、跳过表、专用实现和调优配置。
- `src/flag_gems/runtime/backend/device_finder.py`：厂商探测。优先环境变量，再看 torch 属性，最后执行设备查询命令。
- `src/flag_gems/runtime/op_registrar.py`：将配置中的实现注册到设备 dispatch key。
- `src/flag_gems/__init__.py`：公共 API 与总注册配置。
- `src/flag_gems/utils/pointwise_dynamic.py`：pointwise 动态代码生成。
- `tests/`：pytest 精度测试，通常每个算子一个文件。
- `benchmark/`：性能测试。
- `ctests/`：可选 C++ extension 测试。
- `conf/operators.yaml`：算子清单；`src/flag_gems/backends.yaml`：后端安装配置。
- `setup.sh`：统一创建 `.venv`、安装厂商依赖和编译器并写入激活环境。

厂商实现优先级可概括为：架构专用实现 > vendor 专用实现 > 通用实现；vendor unused/skip 配置会阻止不支持的实现注册。昆仑芯描述为 `vendor_name=kunlunxin`、`device_name=cuda`、`dispatch_key=CUDA`，因为 xpytorch/torch_plugin 对外暴露 CUDA 兼容接口。

## 3. 当前机器环境（2026-07-20 快照）

- Linux x86_64，工作区 `/workspace/FlagGems`。
- 8 张 Kunlunxin P800 OAM，每张约 96 GiB；`xpu-smi` 可用。
- 当前 shell 的 Python 来自 `/root/miniconda/envs/python310_torch29_cuda`。已观察到 PyTorch `2.9.0+cu129`、定制 Triton `3.0.0`；这些版本属于当前机器快照，需在新会话复核。
- 昆仑芯运行栈通过 `torch_plugin` 初始化 XPU runtime，并借助 `torch_xmlir`/符号重写让 XPU 呈现为 CUDA。标准 PyPI Triton 不可直接替代厂商定制编译器。
- 仓库统一环境应优先使用 `./setup.sh kunlunxin`（会重建 `.venv` 且联网安装，执行前确认是否确有必要），日常测试先 `source .venv/bin/activate`。已有机器环境能工作时不要无故重装。

## 4. 测试流程与原理

### 单算子快速测试

```bash
source .venv/bin/activate  # 若项目 venv 已建立
python -m pytest tests/test_<op>.py -sv
python -m pytest -m "<op_mark>" -sv --ref=cpu
```

测试通常构造设备输入，复制 reference 输入，然后：

1. 在 FlagGems 作用域外用原生 torch 得到 reference；`--ref=cpu` 时 `utils.to_reference()` 把 reference 输入移到 CPU。
2. 在 `with flag_gems.use_gems():` 内调用同一个 torch API，ATen dispatch 命中 FlagGems Triton kernel。
3. 用 `flag_gems.testing`/test utils 的 equal 或 close 断言比较，容差可能按 dtype/vendor 配置。

`tests/conftest.py` 提供：

- `--ref {device,cpu}`：reference 所在设备。CPU reference 有助于绕开同一厂商原生实现的共同错误。
- `--quick`：缩减用例集合。
- `--record {none,log,json}` 与 `--output`：记录结果；终场默认也会写/合并 `accuracy_result.json`。
- `--collect-marks`：导出用例 marker 信息。
- marker 用于按算子选择，如 `pytest -m nextafter`。collection 阶段其他测试模块报错可能让目标 marker 尚未运行，必要时直接指定测试文件。

### 批量/CI 测试

- `tools/run_tests.py --ops <op> --gpus "0,1" --dump-output --output-dir results_new`：按 GPU 并行跑精度与性能任务，收集机器、torch 和结果信息；适合算子分发测试。具体参数用 `--help` 复核。
- `tools/test-op.sh <PR_ID>`：CI 根据 `CHANGED_FILES` 找测试文件；常规设备测试后，对适用文件再跑 `--ref=cpu --quick`；benchmark 文件走 core level。PR 模式 fail-fast，全量模式生成 Markdown 和 coverage。
- `ctests`：CI 设置 `CMAKE_ARGS=-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DCMAKE_BUILD_TYPE=Release` 安装后，进入 wheel build 目录执行 `ctest -V`。
- 修复验证最低要求：目标失败 case、目标测试文件、相关 marker；涉及注册/公共工具时再扩大到邻近算子。硬件异常后先确认进程和设备状态，必要时清理 `~/.triton/cache`、`~/.flaggems`，但缓存删除会影响其他任务，不能默认执行。

常见失败分类：kernel 逻辑/数据竞争、dtype promotion 或精度、越界/对齐、片上 SRAM 超限、厂商编译器/autotune 缺陷、vendor API 缺失、注册或 skip 配置错误。不要用放宽全局容差掩盖确定性逻辑错误；只有明确是设备数值特性且误差有依据时才调整容差。

## 5. 后端与 vendor 分发过程

这里的“vendor 分发”有两层：代码运行时分发与 CI/分支交付。

### 运行时分发

1. `DeviceDetector` 从 `GEMS_VENDOR`、`FLAGGEMS_VENDOR`、`GEMS_BACKEND`、`FLAGGEMS_BACKEND`，torch vendor 属性或 `xpu-smi` 等查询得到 vendor。
2. vendor descriptor 给出 `device_name`、dispatch key、dtype 能力等。昆仑芯使用 CUDA dispatch key。
3. FlagGems 汇总 generic、vendor 和 arch 配置；过滤用户 exclude、vendor unused ops、条件未满足项及 C++ patched ops。
4. `GeneralOpRegistrar` 调用 `lib.impl(op, fn, dispatch_key)`；vendor/arch registrar 覆盖或优先选择专用实现。
5. torch API 进入 ATen dispatcher，落到已注册 kernel；退出 `use_gems()` 后恢复原生行为。

### 环境/CI 后端分发

1. `.github/backends.json` 和 workflow runner label 决定 backend（如 `nvidia-cuda133`、`ascend-cann900`、`mthreads-musa520`；其他 label 可直接作为 backend）。
2. `.github/actions/setup-flaggems/action.yml` 调用 `./setup.sh <backend>`，随后可执行 `tools/gpu_check_<vendor>.sh`。
3. `setup.sh` 从 `src/flag_gems/backends.yaml` 读取 Python、vendor 环境、FlagTree/Triton 包和 CMake backend；使用固定 uv 版本创建 `.venv`，安装 `.[backend]`、选择编译器并安装 `.[test]`。
4. backend-test workflow 把 PR 的 changed files 交给 `tools/test-op.sh`；按评论触发的 `/test op:runner` workflow 则 checkout PR，解析 runner/backend，再用 `tools/run_tests.py` 跑指定算子。
5. weekly workflow 使用 `.github/configs/weekly/*.yml` 的硬件 runner、容器镜像、环境变量和输出目录进行矩阵验证。

### Git 分支交付约定

- `upstream`：`flagos-ai/FlagGems` 官方仓库；`origin`：用户 fork `llaboon/FlagGems`。
- `master` 跟踪主线；算子修复采用独立主题分支，例如 `klx/<op>-fix`。先从明确的目标基线建分支，再做最小修复、测试、提交并 push 到 `origin`，最后向 `upstream/master` 提 PR。
- 不把多个无关算子塞进同一分支；不在未经要求时 commit/push；不直接 push master；同步主线前先检查脏工作区。
- 当前检出分支快照为 `klx/nextafter-fix`，跟踪 `origin/klx/nextafter-fix`；该事实易变，每次必须用 Git 复核。

## 6. 算子修复标准工作流

1. 检查状态：当前分支、未提交改动、目标分支基线、已有失败 JSON/分析文档。保护用户已有改动。
2. 复现：先跑最小 node id 或 marker，保存 dtype、shape、错误类型、最大绝对/相对误差；区分 collection、编译、运行期和断言失败。
3. 对照：阅读测试、generic 实现、昆仑芯 override、注册表、unused/skip 配置和相邻算子。用 CPU reference 或小 PyTorch 脚本确认语义。
4. 定因：检查广播/stride、promotion、整数与复数语义、mask 边界、block size、atomic/data race、autotune config、硬件资源和编译器限制。
5. 修复：优先通用正确修复；仅厂商差异则放在 `_kunlunxin` override。skip 只能用于确认无法在当前 compiler/runtime 实现且有清楚原因的情况，并尽量缩到特定 vendor/dtype/case。
6. 验证：复现 case -> 完整目标文件/marker -> 邻近回归；记录命令和通过/跳过/失败数。不能运行时明确说明硬件或依赖阻塞。
7. 交付：检查 diff 和 Git 状态；按用户要求才 commit/push/PR；更新本文中稳定知识及进度。

## 7. 已知修复历史与待办（旧记忆合并，需复核）

### 2026-07-20 sinc 修复

- 分支：`klx/sinc-fix`，基于当前本地 `master`。
- 根因：通用实现直接计算 `sin(pi*x)/(pi*x)`；昆仑芯上 `x` 接近零时发生严重消减误差，少量元素可偏离 `1` 达 `0.1` 量级，影响 `sinc` 和 `sinc_` 的 fp16/fp32/bf16。
- 修复：`src/flag_gems/ops/sinc.py` 在 `|pi*x| < 1e-2` 使用 fp32 三阶 sinc 泰勒多项式，其他范围保留正弦除法。
- 测试文件未修改；按用户要求，非必要不新增或修改测试函数。近零点曾通过临时诊断脚本验证，但未保留为仓库测试。
- 验证命令：`python -m pytest tests/test_sinc.py -sv --ref=cpu`；最终结果 `36 passed`。

### 2026-07-20 sinc 分支交付追溯更正

- 核验发现：`klx/sinc-fix` 从 `e12a220a` 创建，首次 push 到 `origin/klx/sinc-fix` 时没有任何 sinc 修复 commit；之后该分支与 `upstream/master` 均被 rebase/推送到 `0237c11f0`，因此官方 PR 比较页提示 `there isn't anything to compare` 是正确结果。
- `git reflog` 显示 21:29 执行 `rebase (start): checkout upstream/master` 后直接完成到 `0237c11f0`，因为分支没有可 replay 的独有提交。此前 sinc 修改只存在工作区，未 commit；切换到 `master` 前的 stash `stash@{2026-07-20 21:24:06}` 仅包含 `tests/test_apply_repetition_penalties.py`，不包含 sinc 修改，后续 reset 导致 sinc 工作区改动丢失。
- 当前已恢复 `/workspace/FlagGems/src/flag_gems/ops/sinc.py` 的近零泰勒分段修复；`python -m pytest tests/test_sinc.py -q --ref=cpu` 结果为 `36 passed`。当前修复仍是未提交工作区改动，未 commit/push。
- 正确交付方式：先保护/处理当前无关的 `tests/test_apply_repetition_penalties.py` 修改，在 `klx/sinc-fix` 上只 stage `src/flag_gems/ops/sinc.py`，创建 sinc 专用 commit，再正常执行 `git push -u origin klx/sinc-fix`（远端当前正好位于该 commit 的父提交，可 fast-forward，不需要 force）；然后以 `flagos-ai/FlagGems:master` 为 base、`llaboon/FlagGems:klx/sinc-fix` 为 compare 创建 PR。

### 2026-07-20 acos 修复

- 状态：fixed；分支 `klx/sinc-fix`，未提交/未 push（工作树含用户已有改动）。
- 根因：昆仑芯 P800 `tl_extra_shim.acos` 在合法输入域内存在确定性约 3e-3 误差；原始测试基线为 9 failed / 9 passed（历史 JSON 为 8/18）。
- 修复：`src/flag_gems/runtime/backend/_kunlunxin/ops/acos.py` 在 `abs(x) <= 1` 使用 `atan2(sqrt(1-x*x), x)`，域外保留 intrinsic 以维持 NaN 语义；移除该算子的 vendor 黑名单项。
- 验证：`python -m pytest tests/test_acos.py -q --ref=cpu` 两次均 `18 passed`；无测试文件修改。一次未保留的直接公式尝试因域外 NaN 传播不一致被撤回。

旧的 2026-06-11 精度分析曾跟踪 49 个失败项。已有本地/远端主题分支包括：

- scatter 系列：旧记忆标记已通过 atomic 修复数据竞争；本地 `klx/scatter-fix`。
- conv_transpose1d：`klx/conv-transpose1d-skip` 曾因 XPU Triton 3.0 autotune `ZeroDivisionError` 选择 vendor skip；另有未完成 fix 分支。
- geglu/reglu/swiglu 等：`klx/glu-family-skip`，旧记忆称厂商 TE 签名及编译限制导致测试 skip。
- scaled softmax：存在 `klx/scaled-softmax-fix`。
- nextafter：当前分支 `klx/nextafter-fix`。
- copy：`fix/kunlunxin-copy-complete`。

旧跟踪表仍列有精度类（acos、square、diff、vector_norm、softmax/log_softmax、logaddexp、tril、var 等）、资源/访存类（avg_pool3d、gcd、nonzero、logsumexp、max_pool2d_backward、top_k_per_row_prefill）及厂商能力类（margin_ranking_loss、embedding_backward、reflection_pad1d_backward）待处理。但主线和测试在持续变化，开始任何一项前必须查看当前测试结果和 upstream 代码，不能把旧状态当作现状。更详细旧资料仍可只读参考 `/root/.claude/projects/-workspace-FlagGems/memory/flaggems-failing-ops.md` 与 `/workspace/*_analysis.md`（若存在）。

## 8. 维护规则

- 只记录可跨会话复用的架构、流程、环境特点和已验证进度；临时日志不要堆入本文。
- 更新进度时写日期、分支、commit（若有）、验证命令和结果。
- 版本、当前分支、GPU 占用等易变内容标注为快照，不覆盖实时检查。
- 若本文与代码冲突，以当前代码和实测为准，并及时修订本文。
