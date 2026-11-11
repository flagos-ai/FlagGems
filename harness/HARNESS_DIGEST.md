# FlagGems Kunlunxin(XPU) Harness 修复经验汇总（Digest）

> 生成日期：2026-10-21
> 汇总范围：`../AGENTS.md`（总指挥，固定为 `fg_env/AGENTS.md`；FlagGems 仓库内同名文件是上游文件，勿引用）、`harness/HARNESS_SUMMARY.md`、`harness/meta/TODO.md`、
> `harness/meta/LOG.md`（134 条记录）、`harness/solution/`（56 个算子目录）+ `solution/performance/`（39 份性能归档）、
> `harness/meta/regression.txt`（67 条回归规格）、`harness/meta/blacklist.txt`、
> `harness/KernelOpt_Knowledge/xpu3/{KERNEL_OPT_EXPERIENCE,PROFILING_GUIDE}.md`。
>
> 本文是**索引与提炼**，不是替代品：具体命令、原始数据、逐 shape 结果仍以各源文件为准。
> 建议阅读顺序：**本文（全貌）→ HARNESS_SUMMARY.md（方法论）→ KERNEL_OPT_EXPERIENCE.md（编译陷阱字典）→
> PROFILING_GUIDE.md（运行期归因）→ `solution/<op>/solution.md`（单算子证据）**。

---

## 0. 一页速览

**目标**：让 FlagGems 在 Kunlunxin(XPU) 后端上的算子做到「功能 `--ref cpu` 全绿 + 性能 dtype 等权 Gems Speedup ≥ 0.8x」，
且**只改 backend `_kunlunxin/`、零 fallback、证据可复现**。

**六条最高价值结论**（后文均有展开）：

1. **先查 dispatch/注册，再动内核**。vendor 导出缺口、`import` 期硬绑定、`.out`/in-place 变体漏注册的收益可达 ×14～×1000，
   且通常只需在 `_kunlunxin/ops/__init__.py` 加 import + `__all__` 两行。反例：`iand_scalar` 0.054→19.1x、`ior_scalar` 0.055→1.12x。
2. **`grid = lambda meta: ...` 在 XPU 后端参与编译缓存键 ⇒ 每次调用全量重编译**。指纹是「单次固定数百 ms、与张量大小无关」，
   profiler 看不见（host 侧）。改成常量 tuple 即修，实测 `special_logit 0.12→24.4x`、`hypot_ 0.61→44.3x`。
3. **通用 `pointwise_dynamic` 不带 XPU `CodeGenConfig` = 逐 shape 重编译 + tiny tile 无 unroll**，
   16M 元素普遍慢 40–200x。补 `CodeGenConfig(kunlunAutoGrid=True, prefer_1d_tile, unroll_num=8/16, buffer_size_limit=4096)`
   是 `acosh/cosh_out/sinh/log2/log10/mish/hardswish_` 等一批算子从 <1x 到 1–5x 的共同修法。
4. **本后端「能编译 ≠ 正确」**：bf16 核内 load、masked load/store、`tl.sum`/`tl.cumsum`/2D axis-1 归约、`tl.exp2`、
   atomic_add 都存在**按形状/结构分带的静默误编译**。必须按不变量硬路由（`N%8`、TILE≤64/1024、M/N=1）+ 矩阵外探针验证。
5. **性能失败先分类，别硬堆参数**：launch-bound（launch ~10–18.5µs/次）、离散访存、masked 路径（~2x）、软 extern（×3–40）、
   host-device 同步、多 kernel 串行、结构天花板——六类慢源对应完全不同的修法，profiling/IR 是分诊依据。
6. **有一批算子是结构性天花板，投入即浪费**：`sort_stable`（gather-store 墙）、`linalg_*`（串行消元 vs LAPACK）、
   `softmax_out`/`max`/`min`（vendor 融合核）、`moe_align`/`masked_scatter`（离散 scatter）。已在台账标 BLOCKED。

**当前存量**（对应 §6 台账）：功能+性能达标 **36 行**（部分行含函数式/in-place 多个变体）；功能 PASS、性能 BLOCKED **16 行**
（含 1 行已迁 `[FallBack]`、1 行已被上游重写翻盘，实际 BLOCKED 约 14）；有效提升但 <0.8x **4 行**；
SKIP **2**；FallBack **2**；PENDING **1**。另有 **67 条**可用回归规格。

---

## 1. 环境、卡与运行方式

### 1.1 环境装载

```bash
cd /ssd2/v_liuyu60/fg_env/FlagGems     # 或容器内 /workspace/FlagGems
source harness/env.sh                  # 仅设 conda PATH，不固化卡/缓存变量
```

- conda 环境：容器 `/workspace/envs/py310`，宿主 `/ssd2/v_liuyu60/fg_env/envs/py310`（同一 bind mount）。
- `CUDA_VISIBLE_DEVICES` / `XPU_EVENT_KL3_ENABLE` / `PYTHONWARNINGS` / `FLAGGEMS_CACHE_DIR` **由每次选卡后显式 export**。
- XPU 经 torch_xmlir **伪装成 `cuda:0-7`**；`torch.xpu` 不存在，统一用 `flag_gems.device` 与 `torch.cuda.synchronize()`。
- 知识库要求的补充变量（按需）：`DISABLE_VLLM_HOOK=1`（否则 vllm hook 打断 import）、
  `PYTHONPATH=<repo>/src`（防加载错 worktree 造成假失败）；**禁** `source tools/env.sh kunlunxin`（LD_LIBRARY_PATH 遮蔽符号注入）。

### 1.2 选卡与 NOC（每次跑测试/benchmark/profiling 前）

1. `/probe-xpu-cards`：`xpu-smi -m` 看利用率/显存/进程，只选**即时空闲**的卡（0-7 均可，不复用旧探测）。
2. `/probe-reset-xpu`：`dmesg -T | grep -i noc`，只认**明确指向该卡**的 NOC；其他卡历史记录不阻塞。
3. 确认 NOC 才对该卡 `printf '1' > /proc/kunlun/dev<card_id>/soft_reset`，随后复检；无 NOC 禁止重置。
4. 分配环境：`CUDA_VISIBLE_DEVICES=<card>`、`XPU_EVENT_KL3_ENABLE=1`、`PYTHONWARNINGS="ignore::UserWarning"`、
   **本进程唯一** `FLAGGEMS_CACHE_DIR`。

**设备纪律（血泪）**：
- 本机存在互不知情的第二个会话，同一张卡可能被重复派发；测量前后查 `/proc/<pid>/environ` 的 `CUDA_VISIBLE_DEVICES`，
  并且 **ppid 链拉到底**（只看环境变量值会 100% 误判）。
- `xpu-smi` 的 `0 MiB/0%` **不能证明健康**：故障后卡也回 0；只有 dmesg `KL_XID` 流 + 功能探针有信息量。
- KL3/status 700/721/299 是**传染性**的：一次触发污染设备上下文，后续所有核挂起；先软复位再继续。
- `torch.randn` 在 XPU 上有 ~10% 重复值；唯一性场景改用 `arange`/`randint`。
- `support_fp64=False`：`float64` 设备张量**静默降级 fp32**，任何 `dtype==float64` 断言必失败（用 `utils.fp64_is_supported` 门控）。

### 1.3 标准运行命令

```bash
# 最终验收（功能+性能一次跑完，产物统一落 harness/run_logs/<op>）
python tools/run_tests.py --ops "<op>" --gpus <idle_card> --dump-output --output-dir harness/run_logs/<op>

# 功能：完整文件 + CPU 参考（诊断可先 marker/单节点）
cd tests && pytest --timeout=900 -m "<op>" --ref cpu -svv

# 性能：完整矩阵（诊断可 -k / 单 case）
cd benchmark && pytest -m "<op>" -svv -s

# 回归（仅用户当次明确要求时）
DEVICES=<card> PARALLEL=3 bash harness/run_regression.sh

# 代码风格门禁（批次收尾，全仓）
pre-commit run --show-diff-on-failure --color=always --all-files
```

- `--output` 是 **JSON 文件路径**，不是目录（传目录会在 `pytest_terminal_summary` 阶段 `IsADirectoryError`，把全绿变 exit 1）。
- `tests/` 根目录全量收集会被 vllm 环境文件打断（`test_FLA/`、`test_mhc_ops.py` 收集期 assert/ImportError），
  **一律用显式文件路径 + `-m`**；marker 名与文件名对不上会 0 selected。
- `run_regression.sh` 用 shell 拆分会破坏带引号的多词 `-k`（`test_tril.py` 等 3 条已知），需手工复验，不算算子失败。

### 1.4 验收口径（不可协商）

- **功能**：完整 `tests/test_<op>.py`，`--ref cpu`，**3 次独立进程 + 3 个独立 `FLAGGEMS_CACHE_DIR`**，exit 0、无 failed/error、非全 skip。
- **性能**：完整 benchmark 全 dtype/shape/layout 矩阵；**dtype 等权 Gems Speedup**（先每 dtype 内求均值，再对 dtype 均值算术平均）≥ 0.8x；
  同时逐 case 比较固定 Gems latency，防止 torch 侧抖动造成假提升。
- **有效提升**：相对 baseline **严格 >1.0x**；有效但 <0.8x → 记 `BLOCKED（源码保留，有效提升）`；只有无提升/回退才撤销源码。
- **双向不回退**：功能修复不能破坏 benchmark；性能修复不能破坏功能。
- **收尾 AB 强制**：`git stash` 得 A（改动前）、`pop` 得 B（改动后），同卡同环境、各自独立 cache、同矩阵；
  B 相对 A 不得有任何回退。AB 结束工作区必须停在 B。
- **量化伪影先排除**：JIT 重编译 100–340ms 一次性成本（min-of-runs 可消）、torch 参考侧抖动（只信固定 Gems 延迟）、
  并发卡污染可达 300x；`balanced=nan`/UNSTABLE>20% 判「集无效重测」，**绝不当不达标**。
- ⚠️ `FLAGGEMS_CACHE_DIR` **不是 Triton 缓存隔离键**（全局 `~/.triton/cache`）；fresh 的是「进程 + 结果」。

---

## 2. 红线与门禁

| 门禁 | 规则 | 违反后果 |
|---|---|---|
| 修改范围 | 只改 `src/flag_gems/runtime/backend/_kunlunxin/`；测试侧修复只改 `tests/` | 触碰 `ops/`、`fused/` 一律回退（`div_mode` 首版即因此 `git checkout --` 重做） |
| 禁 fallback | backend 内直接调 `torch.mm/addmm/sigmoid/...` 或 `torch.ops.aten.*` 承担被测计算 = fallback | 不得记 PASS，单列 `[FallBack]` |
| `_copy_from` | **2026-10-16 reviewer 判定：`torch.ops.aten._copy_from` 同样是 fallback**，不再视为原生快路径 | 历史 `slice_backward` 等方案作废 |
| `torch.empty/zeros` | 纯分配、不承担被测语义 → 允许（需逐条说明） | — |
| fallback 静态检查 | 每次 backend 改动后跑 `../AGENTS.md` 的 grep 门禁；`[FAIL]` 须逐条澄清 | 未澄清不得 PASS |
| 黑名单 | `harness/meta/blacklist.txt`：12 个 All-dtype-failed + 8 个 `CUSTOMIZED_UNUSED_OPS` | 不得修改/恢复 dispatch/加回归 |
| 3× fresh-cache | 三次独立进程与缓存目录 | 任一次失败不得更新 PASS |
| AB 测试 | 改动前后受控对比 | 缺 AB 不得判 PASS/IMPROVED |
| pre-commit | 批次全部结束后全仓跑，自动格式化改动只允许落在本批文件 | 未全绿不得宣布批次完成 |
| 记录 | 每算子写 `solution/<op>/solution.md`；同步 `TODO.md` + `LOG.md` | 缺记录视为未完成 |

> 注：`_copy_from` 在**非 backend 的测试/参考桥接**中曾被用作性能手段；红线收紧后，任何方案引用它都必须重新审视。

---

## 3. 定位方法论

### 3.1 先证明 dispatch 真的生效（四件套）

1. `flag_gems.<op>.__module__` / `co_filename` 指向 `_kunlunxin.ops.*`；
2. `use_gems()` 前后 CUDA-dispatch kernel 是否存在（False→True）；
3. 运行期捕获 `GEMS_KUNLUNXIN` 日志或 monkeypatch 命中；
4. 与 CPU 参考 maxdiff（「测试全绿」可能走 native composite 假通过）。

**注册管线三层**（`src/flag_gems/__init__.py` → `SpecOpRegistrar.apply()` → `GeneralOpRegistrar`）：

- `SpecOpRegistrar` 用 `importlib` 加载**顶层** `_kunlunxin.ops`，`inspect.getmembers` 收集函数注入 `flag_gems` 顶层 globals；
  `__all__` 非必需，**import 即收集**。
- 随后构建 `_FULL_CONFIG`（`("op.schema", 函数名)`），求值时引用的已是 vendor globals —— 这是绑定机制。
- `use_gems()` 进入时按 **ATen schema 全名**注册到 CUDA key（本后端 device_name="cuda"），退出注销。
- **模块双元性**：顶层 `_kunlunxin.ops` ≠ `flag_gems.runtime.backend._kunlunxin.ops`（两个模块对象），
  验证用 `__module__` 字符串 / `co_filename` / `__globals__ is ...`，**不要用 `is` 比较对象**。
- **`SpecOpRegistrar` 只替换 `flag_gems` 顶层，不替换 `flag_gems.ops` 子模块属性**；
  `from flag_gems.ops.X import f` 在 import 期就把引用固化成 common 版（`iand_scalar`/`ior_scalar` 的根因）。
- `.out` / in-place / `.Tensor` / `.Scalar` 是**不同 schema**，漏一个变体漏一条；`torch.unique`→`_unique2`，无 `aten.unique`。
- `CUSTOMIZED_UNUSED_OPS` 按**函数名**让通用实现落回原生（当前含 `atan2_out, cumsum, grid_sampler_3d_backward, randperm, topk,
  unique, slice, conv_transpose1d, mkldnn_rnn_layer`）。

### 3.2 慢源三桶分类（先排除测量伪象）

| 桶 | 特征 | 首要证据 | 对应修法 |
|---|---|---|---|
| 0 测量伪象 | dtype×shape 特异、复跑漂移 | 稳定态直测复现 | 重测，不改码 |
| A launch-bound | kernel 很轻、launch 占大头 | profiling 中目标核占比低 / 次数巨大 | 多行 2D-tile、合并 launch、in-place、标量参数、host 校验 |
| B memory-bound | 带宽未饱和 / 离散访存 | IR `stride=-1` / `isDiscrete` | constexpr-N、去 pow2 padding、`NEED_MASK`、block-DMA |
| C compute-bound | 指令多 / 归约结构差 | VFPU/VINT/VSFU 分类、软 extern 调用 | 数学近似、饱和算术替代 select、缩依赖链 |

### 3.3 三种互补手段（缺一不可）

| # | 手段 | 回答什么 | 产物 |
|---|---|---|---|
| ① | 检索 `KERNEL_OPT_EXPERIENCE.md` | 这类慢源**以前见过没有**、已知解法与收益量级 | 经验条目 |
| ② | dump IR（`MLIR_ENABLE_DUMP=1`） | kernel **长什么样**：stride、mask、块大小 | `harness/perf_ir/ir_*.log` |
| ③ | 运行期 profiling（P800 Analyzer） | **时间花在哪个 kernel**、ms/次数/占比 | `harness/perf_ir/prof_*.log` + `kernel_summary_*.md` |

**profiling 30 秒速查**：

```bash
CUDA_VISIBLE_DEVICES=<idle> XPUAPI_DEBUG=0x1 XPURT_DISPATCH_MODE=PROFILING \
XPU_EVENT_KL3_ENABLE=1 PYTHONWARNINGS="ignore::UserWarning" FLAGGEMS_CACHE_DIR=$(mktemp -d) \
python <采集脚本>.py > harness/perf_ir/prof_<op>_pre.log 2>&1

python3 ~/.comate/skills/p800-profiling-analyzer/scripts/get_kernel_time.py \
    harness/perf_ir/prof_<op>_pre.log -xpu3
```

- **stdout/stderr 必须合并到同一文件**；`get_op_info.py`/`split_profiling.py`/`efficiency`/`fc_tune` 对单算子**不适用**（实测 0 records）。
- ③ 必须回答「目标 kernel 是不是耗时大头」：实测 `add` 只占 17.6%，80.8% 在随机初始化 kernel —— 不看就会优化错对象。
- post-profiling 同时看 **ms 与 count**：count 变化能暴露「1 个 kernel 被拆成 3 个」的伪优化。
- IR 信号：`triton_xpu.load/store` 的 `stride`（1=连续块 DMA，-1=离散），`isDiscrete`；
  gm2lm/lm2gm 创建期 `offsetState=-1` 是默认值，不是判据。

---

## 4. 性能经验（按杠杆从高到低）

### 4.1 注册/导出缺口（最高杠杆，零内核改动）

**表现**：`flag_gems.<op>.__module__` 指向 `flag_gems.ops.*`（common），benchmark 大 shape 固定几十 ms。

**根因族**：
1. `_kunlunxin/ops/__init__.py` 未导出该符号/变体；
2. common 模块 import 期硬绑定：`from flag_gems.ops.bitwise_and import bitwise_and_scalar_` 固化 common 版；
3. ATen 别名/变体未单独注册（`less_` vs `lt_`，`.out` vs 函数式）；
4. `.out`/in-place 走 native composite **数值正确但 vendor 是死代码**（只有 dispatch 探针能暴露）。

**修法**：新建 `_kunlunxin/ops/<op>.py` 委托已调优 vendor 内核 + `__init__.py` 两行 import/`__all__`。

**案例**：`iand_scalar` 0.054→**19.1x**（bool 55x）、`ior_scalar` 0.055→**1.12x**、
`hsplit` 0.19→**0.9973x**（改用 view 语义，消灭 `tensor_split` copy kernel）；
经验库另记 divide ×240、less_ ×1350、nan_to_num_ ×602、copy ×1277。

### 4.2 `grid = lambda meta: ...` → 每次调用重编译（1 行修复，×3–×200）

- XPU 后端把 launch grid 烘进 `XPUOptions` 并进入编译缓存键；函数体内每次新建 lambda = 新键 = **全量重编译**。
- **指纹**：latency 固定数百 ms 且与张量大小无关；`kernel_cache`/`_bound_launchers` 无界增长；profiler 看不到（host 侧）。
- **修法**：`grid = (cdiv(n, BLOCK_SIZE),)` 常量 tuple（BLOCK_SIZE 本身已是 host 字面量/局部变量）。
- **案例**：special_logit 0.1198→**24.36**、logit_backward 0.1827→**3.73**、
  hypot 0.5176→**11.37**、hypot_out 2.9351→**12.74**、hypot_ 0.6093→**44.25**。
- 横向扫描口径：`grep -rn "grid = lambda" src/flag_gems/runtime/backend/_kunlunxin/ --include=*.py` ≈ 76 处/51 文件
  —— 这是**待分诊上界**，只有位于会被反复调用的函数体内才病态。
- 最干净对照实验：`harness/solution/hypot/probe_compile.py`，打印 `device_caches[dev][0]` 条目数，
  lambda 写法每调用 +1 且 ~320ms，常量 tuple 恒定、0.13ms。

### 4.3 通用 `pointwise_dynamic` 无 XPU `CodeGenConfig`（一批一元算子）

- **表现**：16M 元素 40–200ms，逐 shape 重编译 + IR 膨胀 + tiny tile 无 unroll。
- **标准配方**（按核体微调）：

```python
config_ = CodeGenConfig(
    512, (65536, 65536, 65536), 32, True,
    prefer_1d_tile=True, buffer_size_limit=4096,
    kunlunAutoGrid=True, unroll_num=8,      # 或 16
    isCloseVectorization=False,             # log 系必须 True（正确性）
)
```

- **案例**：acosh 0.318→**2.056x**、cosh_out 0.263→**2.374x**、sinh 0.262→**1.747x**、
  log2 0.086→**1.07x**、log10 ~0.005→**1.067x**、mish 0.153→**1.235x**、hardswish_ 0.855→**5.435x**。
- **注意**：
  - `isCloseVectorization` **无全局最优**：hardsigmoid_backward True→60x 回归；nan_to_num_ True→1.7x 提升；
    log 系 True 是正确性要求（False 会 bf16 误编）。必须逐核 A/B。
  - 双变体同进程共存需用源码文本区分（`1.0 +` vs `1.0000000000000000 +`），否则 codegen 缓存键冲突。
  - `pointwise_dynamic` 包装层本身有小 shape host 债：**每多一个张量入参 ≈ +7µs**；
    用 `out0=torch.empty_like(A)` 预分配 + `kunlunAutoGrid=True` 可治（rsub_scalar 0.633→0.922 等 7 例）。
  - **`vendor override` ≠ `已调优`**：41 个符号的 vendor 调用点大多不传 `config=`，落平台默认（unroll=0/buffer=0）。

### 4.4 IR 爆炸

- **症状**：`ir-*.log` 数十–上百 MB、数百 module、90 万行；benchmark 卡顿/超时。
- **根因**：`@triton.autotune` 逐 shape 重编译；无界 `heur_block_*`/`next_pow2` 生成巨型 int64 offset tile。
- **修法**：删 autotune → 单 config + 手写按 N/shape 分派；block 加**硬 cap**；大 constexpr tile 加 `buffer_size_limit`。
- **案例**：`dot` 914921 行 IR → 消除；`gather` IR 33MB → ~6 GBPS 稳定（BLOCK_M cap=8 + buffer=2048）。
- **补充**：`tl.static_range` 大 N 全展开 → `Failed to tune buffer size`（16×16=256 即触发），改 `tl.range` 动态循环。

### 4.5 Launch-bound（launch ~10–18.5µs/次，与算法复杂度同级）

- **表现**：大 M/小 N、outer×slice_len 巨大，数十万 tiny program。
- **修法**：
  - 多行 2D-tile：`[TILE_M, N]` 一个 program，`TILE_BUDGET≈8192`，`TILE_M=max(1, BUDGET//N)`；
  - 合并 launch：CUDA Graph（注意 capture 会 eager 执行，replay 前必须重新 stage 输入）、in-place 直写、标量参数免 H2D；
  - 能 host 做的标量/符号校验不要下发设备（sym_constrain_range 0.025→1.012）；
  - `_pick_block` 分档 + 12-CTA 中档窗 + 自适应 `(BLOCK, num_warps)`。
- **案例**：`rms_norm [10000,256]` 8.548→2.437ms；`slice_backward` 22.3→0.648ms；
  `fractional_max_pool2d` 0.069→**4.516x**（每 output 一个 program → flat 1D tile）；
  `index_fill [200,40999,3]` 482→21.3ms；`repeat_interleave` flat grid+串行 inner → 3D grid，0.670→**0.986x**。

### 4.6 访存形态与 mask（memory-bound）

- **离散访存两反模式**：① N 作为运行时参数（`m_off*N+n_off` 无法证 stride-1）；② `TILE_N=next_pow2(N)` padding。
  **修法**：归约/连续维声明 `tl.constexpr`，tile 恰好覆盖 `[0,N)` 不 padding，行掩码保留、列不掩码（rms_norm fp16 ~2x）。
- **恒真 mask 仍走慢路径**：加 `NEED_MASK: tl.constexpr`，可整除走 unmasked（rms_norm bf16 +39% / fp16 +31%）。
- **bf16 先转 fp32 再进核**：`tl.load(bf16).to(f32)` 可能 2x GEP 标度 OOB（93–98% 垃圾/NaN）。
- **数据依赖 gather-store 是硬墙**：~4.3ns/elem + 每 store 一个 mfence（仿射 ~0.5ns/elem）；
  能打包就打包键（u64），否则接受 0.4–0.8x 或算法级重写。

### 4.7 归约/扫描结构（本后端最大正确性风险区）

- `tl.sum`：不带 `buffer_size_limit` 只对 BLOCK ≤8192 完整；带 2048 时 32768 可、16384 误编译。
  **block 卡安全点**：8192（无 buffer）/ 32768（buffer=2048）。
- 2D `axis=1` 归约只在 `N%8==0` 正确；`tl.sum/tl.min(axis=0)` 直接报错/OOR → 只能 `tl.dot` 或每行一程序 1D。
- `tl.cumsum`：必须「沿扫描轴 chunk + BN 宽 carry」，carry 与 cumsum 拆两核；块内 exclusive（`tl.cumsum(s)-s`）。
- 1D 在线核 TILE 只接受 **≤64 或恰 1024**（128/256/512 全编译失败，非单调）。
- **`tl.where(mask, x, 0)` 放进归约** → 整核误编译（50%+ 错）；应保留 `mask + other=0.0`，或先物化再 unmasked 归约。
- 案例：`dot` 大 N 从 50–166% 错→精确；`segment_reduce` 从 `uni_sram` 编译失败→**16.06x**；
  `nll_loss_forward` 两段式融合归约 0.049→**1.355x**；`cumsum_/cumprod_` 大行标量前缀修复。

### 4.8 软 extern / fp64 / 数学近似（最大单项收益 ×3–15）

- 软 libdevice extern（`_ZN3xpu*`、`tl_extra_shim.*`、`tl.math.*`）每 lane 一个真实 call，erfc/atan2f/floor/lgamma 全部灾难。
- **修法**：多项式（deg-4～12 Horner）、硬件 SFU（`xpu.rsqrt`、核心 `tl.exp`）、魔数加减：
  erfc ×12.7、ndtr ×9.8、expit ×15.4、special_log1p ×11.7、asin 0.607→0.936、round 0.222→0.948。
- **fp64 软仿真**：Kunlun3 无 fp64 硬件，`x.to(tl.float64)` 的取模/除法慢 ~30x → 换 `tl_extra_shim.fmod`（fmod 0.0406→0.342x）。
- **代数恒等式**：mish 把 exp+log+tanh 降到 1 exp + 1 除法（0.153→1.235x）；
  softplus 用 `max(x,0)+log1p(exp(-|x|))` 稳定形式（0.555→0.84x）。
- **除法改写**：`tensor / scalar` → `tensor * (1.0/scalar)`，约 1.9x（silu_and_mul_out 0.787→1.256）。

### 4.9 host 同步与多 kernel 串行

- **`torch.sum(0-dim)` / `.item()` 链**每次触发 host-device 同步（~163–230µs），能完全主导小算 latency。
  `nll_loss_forward` 三个 0-dim 算子链 → 两段式融合归约，0.049→**1.355x**（~27.7x）。
- `assert_async` 的 Triton kernel + scratch + `.item()` 简化为直接 `tensor.item()`，0.723→**1.09x**。
- 串行 3-kernel（mean/var/backward）是 `cudnn_batch_norm_backward` 0.17x 的主因；能融合则融合，不能则记天花板。
- `@libentry` 的 per-call Python 派发/keying 在大 shape 可忽略、小 shape 直接进 latency（softplus_backward 去 `@libentry` 0.966→1.26x）。

### 4.10 foreach 系列：meta-pointer 标量墙

- common `_foreach_*` → `foreach_*_list` 共享执行器，把 N 个 `data_ptr` 打包进 meta 张量，kernel 内 `tl.load(meta+t)`
  **运行时指针无法被 SVOpt 向量化**（~45GB/s 标量墙）。
- **修法**：按总流量二分——小流量单次 meta launch（省 host），大流量逐 tensor 复用扁平 copy/math kernel（直接指针 + block-DMA 555–853GB/s）。
- **结果**：foreach_addcmul 0.134→**0.604x**、foreach_neg 0.25→**0.530x** / in-place 0.46→**0.617x**、
  foreach_copy_ 0.315→**0.785x/0.793x**（均未达 0.8x，源码保留）。

### 4.11 结构天花板清单（已证伪，不再投入）

| 算子 | 墙 | 证据 |
|---|---|---|
| `sort_stable` | 数据依赖 gather-store (~4.3ns/elem + mfence) | Triton 基数排序比 vendor 慢 500–750x |
| `linalg_lu` / `linalg_svd` / `ormqr` / `linalg_lu_factor_ex` | O(k) 串行消元 / OSJ 顺序旋转 vs LAPACK BLAS-3 | 0.046–0.27x |
| `max` / `min` | vendor `reduce_1d` 融合核快 2–20x | 0.2552/0.2567x 同源 |
| `moe_align_block_size` / `masked_scatter_backward` | 离散 scatter ~6 GBPS | ~0.09x 或不可测 |
| `softmax_out` | 3-kernel/内存流量结构 | 0.69/0.68x |
| `fmod` | native fmod 融合核 ~2–3x 于逐元素 Triton | 0.342/0.522x |
| `smooth_l1_loss`（历史实现） | fused loss+reduction vs 两段式 | 0918 上游重写后 1.76/1.37x，已翻盘 |
| 2D axis=1 归约 | ~1–8 GB/s | 无法追平 vendor fusion |
| launch floor | 0.08–0.25ms | 小 shape speedup 0.17–0.5 是地板 |
| fp16 vs fp32 | 核内 `.to(f32)` 后 fp32 主导 | gems fp16 结构性低于 fp32 |

---

## 5. 正确性经验

### 5.1 注册 / dispatch（最高频）

- **schema 名错**：`rnn_relu` 必须注册 `rnn_relu.input`（ATen 只有 `.input`/`.data`），裸名无效。
- **`.out` 变体缺失**：函数式已导出但 `.out` 未导出 → 走通用 out 实现全 fail。
  新增 XPU out wrapper（`out0=out` 直写或函数式 + 回写）。
- **`import` 期硬绑定**：common op 内 `from ... import f` 固化 common 版，vendor 覆盖不生效（iand/ior/divide/expand_copy）。
- **测试入口决定 dispatch**：`flag_gems.ops.<op>` 永远走 common；`torch.<op>` + `use_gems()` 才走 backend。
  `div_mode` 的正解是改测试路由 + backend 特化，而不是改 common op（红线）。
- **修前先 `git diff` / `git log -- <file>`**：上游 rebuild/merge 会整文件回退历史修复
  （masked_scatter_backward、linalg_svd、hypot_、log_softmax_out、not_equal、mkldnn 均遇到过）。
  「零改码 PASS」必须先排除「被回退」。

### 5.2 语义与边界

- **空张量/零尺寸**：所有 kernel 入口先 `numel()==0` / shape 含 0 短路（`gather` 曾因 `index.numel()//N` 除零）。
- **bf16/int16 位运算**：有符号 int32 无法表示 `1<<31`（copysign bf16）；int16→f32 向量转换会触发
  `size mismatch when packing elements for LLVM struct`（logical_not_ 编译崩溃）→ 整数走直接比较（`x==0`）。
- **半精度 floor 除法**：CPU reference 每步 round 回 half，fp32 opmath 会 off-by-1 →
  `_half_floor_div` 每步 `.to(x.dtype)`（div_mode 7 failed → clean）。
- **NaN/inf 语义**：`tl.maximum/minimum` lower 成 fmax/fmin（忽略 NaN），`hypot` 手写 overflow-safe 公式在 inf/inf 得 NaN →
  用 `tl.hypot` 或补齐边界；`~finitef` 不可靠，改显式 `isnan|isinf`。
- **`tl.log2` 在 XPU 是自然对数**（`logaddexp2` 隔离证明）→ 必须 `tl.log(x)*1.4426950408889634`（log2 曾 18/18 全挂）。
- **`tl.device_assert` 降级为 no-op** → `assert_async` 对 falsy 不抛异常，需 override 恢复 ATen 语义。
- **tie-break**：`max_dim`/`kthvalue` 类必须构造重复值反例；随机输入全绿是假通过。
- **零矩阵/退化情形**：svd 零矩阵返回 `U=I, S=0, V=I`（torch 约定），属退化构造非计算 fallback。

### 5.3 编译器静默误编译（最危险）

- **bf16 核内 load**：`<16 x bf16>` load + `<32 x bf16>` GEP → 93.5–98% 垃圾。**bf16 先转 fp32 再进核**。
- **masked load/store 一律假设不可靠**：`tl.dot` 核内任何 masked 2D load（含恒真）静默错；
  masked store 越界仍写；`other=运行时标量` 不可靠；2D masked multi-store 误编译。
  规避：`constexpr` 拆无 mask 路径、值掩码 `tl.where` 代替地址掩码、clamp 地址 + 寄存器中和。
- **`tl.exp2` 三相语义**：含 dot/loop 的核 `exp2(v)=min(2^v,1.0)`；微核却是 `e^v`；<-16 输入 LUT 钳位。
  softmax/FA 必须 `m` 取 mask 后行最大、保留 `*_LOG2E`、显式 `tl.where(qk>-40,p,0)`、预填 `-1e30`（不用 -inf）。
- **int64 比较**落 `triton_xpu.cmpf` 致 legalizer 崩溃 → int64 只做 `!=`/位运算。
- **`tl.histogram` 不支持**；`tl_extra_shim.pow` 整数指数链接失败；`tl.zeros((), dtype)` codegen 错。
- **`arange` 尺寸 ≠ 目标矩阵维度** → 向量访存 mis-lower（svd 向量化 V 失败，8 failed，已回退标量循环）。
- **单列 transpose** `[BK,1]→[1,BK]` → `rs_col` len=1 fault（SDNN 719）；改 `tl.trans(acc).sum(-1)`。
- **动态 trip count + warp_specialize** → SDNN 702 死锁；改 `tle.pipe` 显式同步。

### 5.4 设备异常（KL3 / status 700 / 721 / 299）

- 触发族：`%`-wrap 把 2D 折叠成单列（M=1/N=1）、循环内 masked load + i32→i64 宽 cast、
  u16 位载 + bitcast、无 mask 宽 store、strided 2D `copy_` 对 2 字节转置、超长核/死循环。
- 特征：**传染性**，一次触发后同卡后续所有核挂起；恢复靠 `printf '1' > /proc/kunlun/dev<id>/soft_reset`。
- NOC 实测：`unique_consecutive` 大 shape（167M，grid=20480）稳定复现
  `kl3_wait_for_noc_idle() timeout`，根因假设是 `tl.sum` 对含 masked-out 垃圾的整块做无 mask 归约；
  同卡 `scatter_src_/softplus_backward/topk/view_as_complex` 无问题。

### 5.5 测试侧 / 参考侧问题（不是算子 bug，不能记 PASS）

- **测试自身 bug**：`test_fmod_.py` 把 reference 克隆写在循环外，原地改写导致从第 2 轮起基准错误（1 file 修复，算子零改动）。
- **caplog 文案**：上游把 `GEMS <OP>` 改成 `GEMS_KUNLUNXIN <OP>`，测试断言子串失败 → 整片全挂与精度无关。
  修法按 logger 来源：vendor 用 `getLogger(__name__)` 时加行内 shim；硬编码 logger 名时改字符串。
- **fp64 skip guard**：Kunlun3 `support_fp64=False`，测试 `torch.randn(fp64)` 静默降 fp32 撞 dtype 断言 →
  测试开头加 `if dtype==float64: pytest.skip`（special_bessel_j0/y1、ormqr）。
- **native 参考不可用**：`torch.log2/log10/hardswish_/fmod_` 对 bf16 `[NOT IMPLEMENTED]`；
  `cudnn_batch_norm` 前向在 XPU 无 override；`adaptive_max_pool3d(return_indices=True)` 返回垃圾索引；
  `deep_gemm` 依赖导致整个测试恒 skip。→ 记 NON_BUG/SKIP 或修 benchmark 参考桥接，**不得伪造成 PASS**。
- **测试侧路由**：`div_mode` 通过改 `tests/test_div.py` 的 kunlunxin 分支走 `torch.div + use_gems()` 命中 backend。

### 5.6 FP64 / fp16 / bf16 平台属性

- fp64 设备张量静默降级 fp32（任何 fp64 dtype 断言必败）。
- fp16 累加必须在核内 `.to(fp32)` 归约后再 cast，否则大 N 精度爆。
- bf16 的「税」在算术 op lowering（`tl.minimum`/`tl.where`），不在搬运；`tl.where` 对 bf16/f32 可慢 2.7–5x，
  「dtype 门控 body 改写」不是安全杠杆。

---

## 6. 算子修复台账

> 指标均为 **dtype 等权 Gems Speedup**（除特别说明）。同一算子多轮复验时取最新值并注明。

### 6.1 功能 + 性能 PASS（达标 ≥0.8x）

| 算子 | 指标 | 关键根因 / 修法 |
|---|---|---|
| iand_scalar | 0.054→**19.1x** | common import 期硬绑定；委托 vendor `bitwise_and_scalar_`（2 文件、零内核改动） |
| ior_scalar | 0.055→**1.12x** | 同上，委托 `bitwise_or_scalar_`（含 int32 字打包） |
| hsplit | 0.19→**0.9973x** | 通用实现走 `tensor_split` 拷贝；改 Python slice 返回零拷贝 view |
| repeat_interleave | 0.670→**0.976/0.986x** | flat grid + 串行 inner → 3D grid `(D,outer,chunks)`，block_i≤32768 |
| softplus | 0.555→**0.84x** | `@libentry`+恒真 mask → pointwise_dynamic + CodeGenConfig + threshold 门控稳定恒等式 |
| softplus_backward | 0.966→**1.26x** | 去 `@libentry` + `NEED_MASK` + `num_warps` 分层 |
| weight_norm_interface | 1.031→**1.404x** | multirow `NEED_MASK` + 统一 tile 分派（TILE_M*N≤8192） |
| smooth_l1_loss / backward | **1.759x / 1.370x**（0918） | 上游重写为融合两段式归约；零改码复验 |
| log_softmax_backward_data | 达标 | 补 `.out` 导出；K==1 连续直写；去恒真 mask + multirow |
| nll_loss_forward | 0.049→**1.355x**（0918 0.902x） | 三个 0-dim 同步算子链 → 两段式融合归约 + BLOCK_N 128→1024 |
| nll_loss_nd_backward | **~1.89x** | 0918 零改码；nll_loss2d vendor 可微 override |
| nll_loss2d_backward | **2.947x** | 0918 零改码 |
| segment_reduce | 0.662→**16.06x**（0918 7.49x） | 标量累加循环 → 均匀 segment 两条向量化快路径；uniformity 按张量身份缓存 |
| segment_reduce_backward | **10.14x** | 0918 零改码（历史均匀 segment 三快路径保留） |
| segment_reduce_out | **5.76x** | 委托函数式 + `out.copy_()` 回写 |
| special_gammaincc | **~7.61x** | 修 `lgamma` stub 链接失败；caplog 文案单行修复 |
| assert_async | 0.723→**1.09x** | `tl.device_assert` no-op → override；性能轮改直接 `tensor.item()` |
| adaptive_max_pool3d_backward | **3.92x** | 去 atomic scatter（无原子双路径）+ 补 forward override |
| hypot / hypot_ | **~11x** | `tl.hypot` 语义修复 + `grid=lambda` 改常量 tuple |
| heaviside_ | **~30.9x** | 零改码（dispatch 与通用实现已达标） |
| cumsum_ | 0.0975→**~1.12x**（0916/0918） | 行扫描 carry 修复 + int64 累加语义；上游 chunk 扫描 |
| acosh | 0.318→**2.056x** | 通用 pointwise_dynamic → XPU CodeGenConfig |
| cosh_out | 0.263→**2.374x**（0918 1.746x） | 同上（unroll 16 / isCloseVectorization=True） |
| hardswish_ | 0.855→**5.435x** | 手写 kernel + 恒真 mask → pointwise_dynamic + CodeGenConfig + fp32 累加 |
| log10_ / log10_out | ~0.005→**1.067x / 1.06x** | 同上（log 系 `isCloseVectorization=False`） |
| log2 | 0.086→**1.07x** | 两层：`tl.log2` base-e 语义修复 + CodeGenConfig |
| fractional_max_pool2d | 0.069→**4.516x**（0918 3.30x） | 每 output 一 program → flat 1D tile；修 sample dtype 语义 |
| logical_not_ | **~1.24x** | int16 bitcast 编译崩溃 → 编译期 dtype 分支 |
| cumprod_ | **17.81x**（0918 18.74x） | 修 in-place 回写/`.contiguous()` 整数崩溃 + 大行前缀 mis-lower |
| mish / mish_ | 0.153→**1.235x** | 代数恒等式降 3 超越函数到 1 exp+1 除法 + CodeGenConfig |
| sinh / sinh_ | 0.262→**1.747x** | 通用 pointwise_dynamic → CodeGenConfig（镜像 cosh） |
| safe_softmax | **1.81x** | 零改码（multirow 2D 快路径 + per-row fallback 已在源） |
| special_bessel_j0 | **25.5x** | 上游有理逼近 override；仅补测试 fp64 skip guard |
| special_bessel_y1 | **~20.3→30x** | 新建有理逼近 override（`Y1(±0)=-inf` 语义）+ 测试 skip guard |
| div_mode / div_mode_ | **1.79x** | fp16/bf16 floor 除法每步 round 回 half；backend 特化 + 测试路由 |
| le / less_equal | benchmark 4 passed | flagtree `cmpf→ui32` 约束：`max_tile_size` 1024→512（待 CI 确认） |

### 6.2 功能 PASS、性能 BLOCKED（源码保留）

| 算子 | 指标 | 根因 / 已做改动 |
|---|---|---|
| linalg_lu | ~0.072x | 通用分块 kernel lowering 失败 → 四类小 kernel 串行消元；结构地板 |
| linalg_lu_factor_ex | ~0.06x | **0918 上游改成 native fallback，红线违规 → 见 [FallBack]** |
| ormqr | 0.080→**0.267x** | RIGHT-only + transpose 恒等式 + reduce-OUTSIDE；BLAS-1 vs BLAS-3 地板 |
| linalg_svd | 0.033→**0.046x** | round-robin 列对并行 + 收紧 sweep；OSJ 顺序 launch 地板 |
| cudnn_batch_norm_backward | 0.125–0.169x | unmasked 向量 load 破坏 per-channel 标量 load → 目标化 mask；3-kernel 串行地板 |
| masked_scatter_backward | ~0.088x | 二次离散 store 被丢 → host `torch.zeros` 填零、kernel 只留一次 store |
| softmax_out / backward_out | 0.6905x / 0.6830x | 补 `.out` 导出 + 宽行 `softmax_kernel_wide` + N==1 短路 |
| index_fill | 0.949→**0.986x** | 小 inner position-blocked kernel（482→21.3ms，~22.6x），总速仍 <1.0 |
| moe_align_block_size_triton | 不可测 | 两项 benchmark 参考均不可用；离散 scatter 结构天花板 |
| max / max_dim | **0.2567x** | packed-key 单遍有误编译/陈旧索引 → 重写为 min.py 三遍式镜像；vendor 融合核 2–20x |
| sort_stable | ~0.52x（bf16 提升 2x） | radix 8→4 pass；gather-store 地板 |
| thnn_fused_lstm_cell_backward_impl | 0.782→**0.793x** | `BATCH constexpr` + static_range；`sum(dim=0)` 归约地板 |
| fmod_scalar / fmod_tensor | 0.058→**0.522x** / 0.041→**0.342x** | fp64 软仿真 → `tl_extra_shim.fmod`；native 融合核更快 |
| smooth_l1_loss（历史） | 0.95x / 0.38x | 0914 两段式；0918 上游重写后已翻为 PASS |
| norm | 未严格 >baseline | 修 full-reduce OOB + fp16 mid 溢出；perf gate 未过 |
| svd | ~0.02x | 功能 44/44：通用 static_range Jacobi compile-explode → runtime-loop OSJ + kernel 内累积 V（病态矩阵正交性）+ fp16/bf16 走 f32；性能 batch→grid + sweeps 32→12，OSJ 顺序旋转地板 |

### 6.3 有效提升但未达 0.8x（≥0.5x，「可提炼成功经验」）

| 算子 | 指标 | 要点 |
|---|---|---|
| nonzero_numpy | 0.4485→**0.6019x** | power-of-2 shift/mask dense kernel + float 计数改整数比较（~2x） |
| foreach_addcmul_scalar_list | 0.134→**0.604x** | 按总流量分派：小流量单 meta launch / 大流量逐 tensor flat kernel |
| foreach_neg / foreach_neg_ | 0.25→**0.530x** / 0.46→**0.617x** | 同上；meta-pointer 标量墙 |
| foreach_copy_ | 0.315→**0.785/0.793x** | 4MiB 分界；大流量复用 `_copy_flat_kernel`（555–853GB/s） |

### 6.4 SKIP / NON_BUG

| 算子 | 分类 | 原因 |
|---|---|---|
| special_shifted_chebyshev_polynomial_w | SKIP | 官方 tests/benchmark `skipif(vendor not in ("nvidia","thead"))` 全 skip；独立验证 21.06x |
| get_paged_mqa_logits_metadata | SKIP | 依赖 `deep_gemm` CUDA 扩展，XPU 恒 4 skipped |
| logical_not（函数式） | PENDING | 0.862x <1.0x，memory-bound 天花板；`logical_not_` 已 1.24x |

### 6.5 [FallBack]（不得记 PASS）

| 算子 | 事实 | 处置 |
|---|---|---|
| linalg_lu_factor_ex | 上游 `d312aa024` 把 O(k) Triton 实现改成 `redispatch_boxed` 到 vendor LAPACK `getrf`；性能 ~1.015x 是伪达标 | 记 FallBack；原 Triton 实现按 ~0.06x BLOCKED 跟踪 |
| mkldnn_rnn_layer | override 内直接 `torch.addmm/mm/sigmoid/tanh` + autograd 重算前向 | `git revert --no-commit 184d22eab`，重回 `CUSTOMIZED_UNUSED_OPS` |

---

## 7. DEAD END（别再试）

**编译器/语义**
- `tl.where` 放进归约（where-in-reduce）→ 整个归约误编译；
- persistent `while` grid-stride 累加 → `TritonXPUMask`/`uni_sram`；
- 大 `tl.static_range` 全展开（Jacobi/pooling）→ `Failed to tune buffer size`；
- `tl.sum(axis=0)` / `tl.trans` 轴-0 归约；
- 2D `axis=1` 归约且 `N%8!=0`；2D `tl.min(axis=1)`；
- `tl.dot` 核内任何 masked 2D load（含恒真 mask）；
- bf16 核内 `tl.load(...).to(f32)`、u16 位载 + bitcast；
- `tl.cumsum` 独立扫描（非 chunk+carry）；
- `tl.histogram`、`tl_extra_shim.pow` 整数指数、int64 `<`/`>=`；
- 1D 在线核 TILE 128/256/512；
- `arange` 尺寸 ≠ 目标矩阵维度的向量访存；
- 运行时标量 `other=`、循环内标量边界 `tl.load`。

**性能**
- `@triton.autotune` 多配置（XPU 逐 shape 重编译）；
- 无界 `heur_block_*` / `next_pow2` padding；
- `grid = lambda`（每次调用重编译）；
- 改全局 `config_` 迁就某一族（copy-family 曾让 rank-4 跨步 15.4s→>900s）；
- `tl.where` / 饱和替代方案不区分 dtype 一刀切（bf16/f32 可慢 2.7–5x）；
- 只优化占比很小的子 kernel（lstm cell 仅占 torch 总时间 ~2%，-0.7% 回退）；
- SFU 配方 `exp2(e*log2(x))` 对负底数 + 整数指数给 NaN（pow 功能 16/36 失败，已回退）。

**流程**
- 在 common `ops/` 里改以绕过注册（红线，已被回退）；
- 用 `_copy_from` 作为 backend 快路径（红线收紧后属 fallback）；
- 只跑一次 benchmark 就宣布达标（JIT 重编译/参考抖动/并发污染）；
- 信任单轮 `balanced`（nan/UNSTABLE 时改对抗下界或重测）。

---

## 8. 命令速查

```bash
# 环境
cd /ssd2/v_liuyu60/fg_env/FlagGems && source harness/env.sh
xpu-smi -m
dmesg -T | grep -i noc

# 单算子功能（诊断）
cd tests && CUDA_VISIBLE_DEVICES=<c> XPU_EVENT_KL3_ENABLE=1 PYTHONWARNINGS="ignore::UserWarning" \
  FLAGGEMS_CACHE_DIR=$(mktemp -d) pytest --timeout=900 -m "<op>" --ref cpu -q

# 单算子性能
cd benchmark && CUDA_VISIBLE_DEVICES=<c> ... pytest -m "<op>" -s

# 统一验收（功能+性能）
python tools/run_tests.py --ops "<op>" --gpus <c> --dump-output --output-dir harness/run_logs/<op>

# IR
MLIR_ENABLE_DUMP=1 pytest tests/test_<op>.py -m "<op>" --ref cpu -svv > harness/perf_ir/ir_test_<op>.log 2>&1

# profiling
CUDA_VISIBLE_DEVICES=<c> XPUAPI_DEBUG=0x1 XPURT_DISPATCH_MODE=PROFILING XPU_EVENT_KL3_ENABLE=1 \
  FLAGGEMS_CACHE_DIR=$(mktemp -d) python <script>.py > harness/perf_ir/prof_<op>_pre.log 2>&1
python3 ~/.comate/skills/p800-profiling-analyzer/scripts/get_kernel_time.py harness/perf_ir/prof_<op>_pre.log -xpu3

# 回归 / 风格
DEVICES=<c> PARALLEL=3 bash harness/run_regression.sh
pre-commit run --show-diff-on-failure --color=always --all-files
```

---

## 9. 遗留项与下一步

1. **FallBack 清理**：`linalg_lu_factor_ex`（上游 native redispatch）与 `mkldnn_rnn_layer`（已 revert）需决定是否恢复真实
   Triton 实现；在恢复前不得记 PASS、不得加入 regression（`linalg_lu_factor_ex` 当前仍在 regression.txt，需复核）。
2. **`grid = lambda` 横向清扫**：76 处/51 文件为待分诊上界，优先 `flash_api.py`、`weightnorm.py`、`addmm.py`、
   `instance_norm.py`、`nllloss.py`、`attention.py`；用 `probe_compile.py` 同款探针量化。
3. **flagtree CI 差异**：`le/less_equal` 的 `max_tile_size` 修复依赖「同族 512 全绿 + sizePerCore=[1024] 强相关 +
   triton `ad274881` cmpf 约束修复」三重证据，本机无 `/opt/flagtree` 无法端到端复现，待 CI 复跑确认。
4. **NOC 根因**：`unique_consecutive` large path 的无 mask `tl.sum` 假设待验证并修复；该卡级故障会污染后续所有任务。
5. **未达 0.8x 但有效**：`nonzero_numpy` 0.60、`foreach_addcmul` 0.60、`foreach_neg` 0.53/0.62、
   `foreach_copy_` 0.79、`index_fill` 0.99 —— 源码已保留，可继续针对剩余 dtype/路径单变量优化。
6. **文档一致性**：`TODO.md` 中存在历史重复/滞后条目（如 `cumsum_`、`smooth_l1_loss` 的旧 BLOCKED 结论），
   以 `LOG.md` 最新日期条目 + 性能归档 §0 为准。
7. **profiling 手册仅本地**：`PROFILING_GUIDE.md` 未推送到知识库远程，换机 fresh clone 时按 `../AGENTS.md` 向用户索取，
   **不得凭记忆编造流程**。

---

## 10. 源文档索引

| 文档 | 内容 | 何时读 |
|---|---|---|
| [`../AGENTS.md`](../AGENTS.md) | 最高约束：红线、Bootstrap、验收口径、多 Agent 分工（固定为 `fg_env/AGENTS.md`） | 开工前必读 |
| `HARNESS_SUMMARY.md` | 修复经验总纲：慢源分类、IR 信号、门禁链 | 定位/改码前 |
| `harness/meta/TODO.md` | 算子状态看板（PASS/BLOCKED/SKIP/FallBack） | 接任务时 |
| `harness/meta/LOG.md` | 134 条按日期叙事记录，含每轮命令/结论/踩坑 | 查历史决策 |
| `harness/meta/regression.txt` | 67 条可执行回归规格 | 收尾/回归 |
| `harness/meta/blacklist.txt` | 禁改算子 + `CUSTOMIZED_UNUSED_OPS` | 开工前 |
| `harness/solution/<op>/solution.md` | 单算子：基线、根因、改动、3×验证、限制 | 复用/复核 |
| `harness/solution/performance/<op>.md` | 单算子：pre/post profiling、逐 dtype、AB | 性能验收 |
| `harness/KernelOpt_Knowledge/xpu3/KERNEL_OPT_EXPERIENCE.md` | 编译器陷阱 43 条 + 性能技术栈 + 十五大黄金律 | 改码前字典 |
| `harness/KernelOpt_Knowledge/xpu3/PROFILING_GUIDE.md` | P800 profiling 采集/分析/坑 | 性能任务强制 |
| `harness/skills/{probe-xpu-cards,probe-reset-xpu,summarize-gems-speedup}` | 选卡、NOC 复位、dtype 等权汇总 | 每次运行/汇总 |
| `harness/perf_ir/` | IR dump、profiling 原始日志、微基准脚本 | 复核证据 |
