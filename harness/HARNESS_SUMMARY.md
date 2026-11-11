# HARNESS_SUMMARY —— FlagGems Kunlunxin 修复经验总纲

> 每个性能/正确性修复子 Agent 在修改、测试或 benchmark 前，**必须先读本文件，再读 `../AGENTS.md`**（总指挥固定为 `fg_env/AGENTS.md`，即本文件上一级的 `../AGENTS.md`；FlagGems 仓库内的 `AGENTS.md` 是上游文件，禁止引用）。
> 本总纲汇集可复用的 Kunlunxin/XPU 修复经验；它是定位线索，不是绕过 `../AGENTS.md` 红线的许可。
> 经验来源：智源性能优化总结（ku 知识库，**原始快照见 `references/ku_experience.md`**）+ `../AGENTS.md` 修复经验总纲。
> ku 文档更新时：先重新拉取并 diff `references/ku_experience.md`，再把变更同步进本文件。

## 0. 范围红线（先记牢）

- 只改 `src/flag_gems/runtime/backend/_kunlunxin/`；**禁止**改 `src/flag_gems/ops/`、`src/flag_gems/fused/`。
- **禁 fallback**：CPU ATen / native / composite / "复用基础算子" 绕过 XPU kernel 均不允许；结果不得记 `[PASS]`。
- 黑名单见 `harness/meta/blacklist.txt`：不得修改、恢复 dispatch 或加入 regression。

## 1. 定位前先确认 dispatch 真的生效

- 确认 ATen schema、vendor override 导出与 `SpecOpRegistrar` dispatch 实际生效；函数式 / in-place / `.out` 三个变体都要查。
- 特别检查空张量、零尺寸、dtype/bit-cast 和数值边界语义。
- 后端 override 必须同时检查：实现模块 + `ops/__init__.py` import + `__all__` + ATen schema 注册。
- 用 `python -c "import flag_gems; print(flag_gems.ops.<op>.__module__)"` 校验 override 是否真的替换成 vendor 版本。**历史"改了半天没生效、据此下错误结论"多源于此。**

## 2. 现状基线 & 性能差常见原因

- 基线（2026/07/24）：纳入统计 430 算子，**性能中位数 0.87**，算子通过率 501/820 = 61%。

性能差 6 大常见原因：
1. tile-size 太小
2. 没有 unroll
3. 没命中向量化
4. 离散访存
5. 多余 copy
6. python 层显式类型提升到 fp32，数据搬运增加

## 3. 性能优化：反复出现的可优化点（症状 → 根因 → 修法 → 案例）

### 3.1 IR 爆炸（首要元凶）
- **症状**：`ir-*.log` 数十~上百 MB / 数百 module / 90 万行；benchmark 卡顿或超时。
- **根因族**：
  1. `@triton.autotune`（多 config 每 shape 逐一 warmup/rep）在 XPU 逐 shape **重编译**。同源：`libtuner` / `addmm_` / `dot`。
  2. `heur_block_*` 无界（`next_pow2(cdiv(N,12))`、`sqrt(N)` 类）→ 巨型 `int64` offset tile 在 `ConvertTritonXPUToLLVM` 逐元素展开。同源：`gather` / `diag` / `index_add` / `is_all_true` / `amax`。
- **修法**：
  - 删 `@triton.autotune`，改**单 config + 手写按 N/shape 分派**（例：`dot` 按 N 阈值走 `SINGLE_BLOCK=8192` / `SPLIT_BLOCK=32768` 两条正确路径）。
  - block 加**硬 cap**（对齐原 autotune configs），例：`gather` 把 `BLOCK_M` 封顶到 8。
  - 大 constexpr tile 加 `buffer_size_limit=2048/8192`，把整块 materialize 拆成小块滚动处理。
- **案例**：`dot`（914921 行 → 消除）、`gather`（33MB → 稳定 ~6 GBPS）。

### 3.2 Launch-bound（大 grid × 小 program）
- **症状**：大 M/小 N 或 outer×slice_len 巨大 → 数十万 tiny program，时间全耗 launch。例：`rms_norm [10000,256]` 8.548ms / speedup 0.006；`slice_backward [512,1024,512]` 22.3ms / 0.029。
- **根因**：XPU program launch ~0.6-0.9μs，逐行/逐 block 拷贝放大。
- **修法**：
  - **多行 2D-tile**：`[TILE_M, N]` 一个 program，grid 从 M 降到 `cdiv(M, TILE_M)`（`TILE_BUDGET=8192`, `TILE_M=max(1,BUDGET//N)`）。
  - **弃自写 copy-kernel 走原生 strided-copy**：`torch.ops.aten.slice(...)` 取视图 + `torch.ops.aten._copy_from(src, view, False)`。gems 不 override `_copy_from`，直达厂商引擎。
- **注意**：不要盲目统一到 `_copy_from`——2D 小 shape 反而慢（两趟 vs 单趟融合），fast path 按条件保留。

### 3.3 连续块 DMA vs 离散访存（constexpr-N 反模式）
- **症状**：连续内存却走离散 gather；fp16/bf16 尤伤（2-2.5x 惩罚）。
- **两种反模式**：
  1. **N 作为运行时参数**：`m_off*N + n_off` 的运行时 `*N` 无法被 OffsetAnalysis 证 stride-1。
  2. `TILE_N = next_pow2(N)` **padding**：非单位内层 stride 触发离散访存。
- **修法**：归约维/连续维 **N 声明为 `tl.constexpr`**、tile 恰好覆盖 `[0, N)` 不做 pow2 padding；行掩码保留，列不掩码。
- **数据**（`rms_norm` 同 shape，仅 padded→constexpr-N）：fp16 4.91→2.38ms、bf16 4.75→2.81ms、fp32 4.80→2.46ms（~2x）。

### 3.4 恒真 mask 的慢 masked-memory 路径
- **症状**：N 恰好整除 BLOCK_SIZE、mask 恒 True，但访存仍走慢 masked 路径；fp16/bf16 惩罚 1.85x/2.43x。
- **修法**：kernel 加 `NEED_MASK: tl.constexpr`；调用处传 `need_mask=(N % BLOCK_SIZE != 0)`。整除走 unmasked，不整除回退 masked（数值等价，maxdiff=0）。见 `rms_norm` R2（bf16 +39% / fp16 +31%）。

### 3.5 `tl.sum` / 归约的 XPU 正确性天花板
- **规律**（`dot` 隔离验证）：
  - **不带** `buffer_size_limit`：`tl.sum` 只对 `BLOCK ≤ 8192` 完整；更大只累加前 8192 lane（**数值静默错**）。
  - **带** `buffer_size_limit=2048`：`BLOCK == 32768` 完整；`16384` 单程序仍**误编译**（~1e-3 相对误差）。
  - **tail 块 masked load 不可靠**（后端级）：`mask/other=0.0` 被忽略，越界读入邻接内存并计入 `tl.sum`。`sum`/`mean`/`dot` 等归约族**共有**。
- **修法**：block 卡在安全点 `8192`（无 buffer）/`32768`（带 buffer=2048）；`mid` 严格 `cdiv(N, block)` 分配（避免 kernel_2 读未初始化 `torch.empty`）。
- **DEAD END（勿再试）**：
  - `where(mask, off, 0)` + `where(mask, x*y, 0)` 规避越界读 → where-in-reduce 误编译，整个归约算错（50%+）。
  - persistent `while` grid-stride 累加 → XPU 编译崩（`TritonXPUMask` pass 失败 / `uni_sram` OOM）。
- **验证纪律**：benchmark SUCCESS ≠ 数值正确（不校验结果）。修性能时必须用**受控输入（全常数）+ fp32 ground truth** 单独验证大 N 正确性。

### 3.6 CodeGenConfig 调参（pointwise / 标量对比族）
- **场景**：访存受限 pointwise（`greater_scalar`、`gt`、`isinf` 等）。
- **甜点**：`unroll_num` 8 → **16** 提升访存并行度（32 反弹）；`buffer_size_limit` = **8192**（16384 已到顶）；标量路径**不设 fusion 环境变量**（`TRITONXPU_COMPARE_FUSION` 零收益，`TRITONXPU_FP16_FAST` 让 fp16 `uni_sram` OOM）。
- **踩坑（必记）**：**环境变量只在编译期生效**。同进程改 env 后重复调用命中旧内核是缓存假象。验证必须**每 combo 独立进程 + `FLAGGEMS_CACHE_DIR=$(mktemp -d)` 新鲜编译**。

### 3.7 `@triton.autotune` 替代方案
- 单 config + `@libentry` 缓存 → 编译一次、逐 shape 复用。
- 有条件的多路径手写分派（按 N/shape 分桶），显式比无脑 autotune 快且不炸 IR。

### 3.8 结构天花板（不再深挖，记录即可）
- **XPU 2D axis=1 归约 ~1-8 GB/s**：`rms_norm [10000,256]` 类；vendor 融合 kernel 快数十倍，Triton 算子层追不平。
- **离散 gather ~6 GBPS**：`gather`/`scatter`/`embedding`/`index` 族；vendor 聚合 kernel ~600-970 GBPS。
- **Launch floor ~0.08-0.25ms**：小 shape 单趟 Triton 无法压缩，speedup ~0.17-0.5 是地板。
- **fp16 与 fp32 同耗时**：kernel 内 `.to(fp32)` 后 fp32 乘加主导（如 `dot`），torch fp16 用硬件 fp16 快 2x → gems fp16 speedup 结构性低于 fp32。

### 三桶分类（先排除测量伪象 → 再定性）
| 桶 | 特征 | 首要证据 | 下一步 |
|---|---|---|---|
| 0 测量伪象 | dtype×shape 特异、复跑漂移 | 稳定态直测复现 | 回测 |
| A launch-bound | kernel 太轻、launch 占大头 | 耗时分解（kernel 占比低） | 3.2 候选 |
| B memory-bound | 访存低效 / 带宽未饱和 | IR `stride`/`isDiscrete` | 3.3/3.4 候选 |
| C compute-bound | 计算指令多 / 归约结构差 | VFPU/VINT/VSFU 指令分类 | 3.5/3.6 候选 |

## 4. 正确性 / 功能修复经验

### 4.1 注册与 dispatch（最常见）
- **schema 名错误**：`rnn_relu` 应注册为 `rnn_relu.input`（ATen 仅有 `.input`/`.data` 两个重载），裸名注册无效。
- **out 变体缺失**：函数式已导出但 `.out` 未导出 → 走通用 out 实现全 fail。修法：新增并导出 XPU out wrapper（校验 `grad_input` 大小 → 调后端函数式 → `_copy_from` 回写）。例 `softmax_backward_out`。
- **override 未生效**：改错 `ops/__init__.py` re-export；或 `SpecOpRegistrar` 没被替换成 vendor 版本。
- **参数顺序 / 语义偏差**：`reflection_pad3d_backward` 六元素 padding 应按 ATen 的 W/H/D 顺序（`w0,w1,h0,h1,d0,d1`），勿按 D/H/W 解析。

### 4.2 边界条件（空张量 / 除零）
- `gather`：index 最后一维为空时 `N == 0`，`index.numel() // N` 触发除零；修法：在 stride/grid 前直接 return 已分配的空输出。
- **通用规范**：所有 kernel 入口先做 `numel() == 0` / `shape 含 0` 短路。

### 4.3 dtype / bit-cast 陷阱
- `copysign` bf16：有符号 `int32` 无法表示 `1 << 31`；修法：bf16 → fp32 → 按位转 int32 → `< 0` 判符号；其他 dtype 保持无符号 mask 逻辑。同类：`arcsin_`、`baddbmm`、`polar`、`scaled_mm`。
- `~tl_extra_shim.finitef` 不可靠：会把有限值误判非有限（XPU 下），改用显式 `isnan | isinf`。见 `_amp_foreach_non_finite_check_and_unscale_`。
- **fp16 累加**：kernel 内做 `.to(fp32)` 归约后再 `.to(x.dtype)`；否则大 N 累加精度爆掉。

### 4.4 语义等价（改用 native 原语）
- 反向 / view-scatter 类（`slice_backward` / `resize_` / `constant_pad_nd`）：**镜像 native 语义** = `zeros`/`empty` 分配 + `torch.ops.aten.slice(...)`（view op）+ `torch.ops.aten._copy_from(src, view, False)`。gems 只 override `copy_`/`copy`，**从不 override** `_copy_from` → 直达原生 strided-copy 引擎。
- 好处：全 dtype/全 rank/任意 strided 视图皆正确，且性能与 native 一致。

### 4.5 in-place 与嵌套 copy_
- in-place 变体（`_` 后缀）回写时避免嵌套调用已注册的 `copy_`：写临时输出后用 `aten::_copy_from(output, tensor, False)` 回写，绕开 gems 的 `copy_` override 循环。

### 4.6 XPU 后端级坑（正确性）
- `tl.sum` block 上界（见 3.5）：静默漏 lane / tail 块越界读入邻接内存 → 大 N 数值错。kernel 必须限死 block=8192 或 32768。
- `tl.where(mask, x, 0.0)` in reduce：XPU 误编译成整个归约算错（50%+），只在归约以外用；归约里保留 `mask + other=0.0`。
- `TritonXPUMask` / `uni_sram` OOM：persistent while 累加、超大 tile、fp16 fusion 环境变量下的 compare 等触发；退回逐 tile 分派 + `buffer_size_limit` 拆解。

## 5. 典型案例速查表

| 算子 | 类别 | 根因 | 修法要点 | 收益 |
|-|-|-|-|-|
| `dot` | perf/kernel+正确性 | autotune IR 爆炸 + `tl.sum` 漏 lane + mid 越界 | 删 autotune，按 N 分派 8192/32768，`mid=cdiv(N,32768)` | 0.5403→0.5557；大 N 从 50~166% 错→精确 |
| `rms_norm` R1 | perf/kernel | launch-bound（大 M/小 N） | 多行 2D-tile + constexpr-N 连续块 | 8.548ms→2.437ms |
| `rms_norm` R2 | perf/kernel | 恒真 mask 走慢路径 | `NEED_MASK` constexpr | fp16 +31% / bf16 +39% |
| `gather` | perf/memory | 无界 `BLOCK_M` + 巨 int64 tile → IR 33MB | `BLOCK_M cap=8` + `buffer_size_limit=2048` | IR 爆炸消除；~6 GBPS 稳定 |
| `slice_backward` | perf/layout | 262144 tiny program launch-bound | `zeros`+`aten.slice`+`aten._copy_from` | 22.3ms→0.648ms（~34x） |
| `greater_scalar` | perf/spec | 默认 config 未调 | `unroll_num=16`+`buffer_size_limit=8192` | +9.2% / +8.2% |
| `unique2` | perf/kernel | 三条手写 kernel ~9GB/s | 重写为 gems 原语链（sort→ne→nonzero→index_select） | 灾难 0.08-0.28 → 追平 |
| `rnn_relu` | func/registration | schema 名错（裸名 vs `rnn_relu.input`） | 改注册名 | override 生效 |
| `gather` func | func/correctness | 空 index 除零 | 短路空输入 | 三进程通过 |
| `copysign` bf16 | func/dtype | int32 无法表示 `1<<31` | bf16→fp32→bit-cast→`<0` | 54 passed |
| `softmax_backward_out` | func/dispatch | out 变体未导出 | 新增导出 wrapper + `_copy_from` 回写 | 0→30 passed |
| `reflection_pad3d_backward` | func/correctness | 六元素顺序错 D/H/W→W/H/D | 参数顺序修正 | 48 passed |
| `_amp_foreach_non_finite_check_and_unscale_` | func/correctness | `finitef` 不可靠 | `isnan|isinf` + `_copy_from` 回写 | 6 passed |

## 6. IR 里的性能信号（XPU 特定）

- `triton_xpu.load`/`store` 的 `stride`：`1` = 连续块 DMA，`-1` = 离散 gather/scatter。
- `isDiscrete`：离散访存是杀手。
- gm2lm/lm2gm 创建阶段的 `offsetState=-1` 是默认值，**不是**最终判据。

## 7. 单变量 A/B 门禁链（不过即撤销）

```
门禁 0: dispatch 生效验证（走改动后实现）
门禁 a: 最小失败节点（编译/运行不崩）
门禁 b: 完整功能 tests/ --ref cpu（无回退）
门禁 c: 完整 benchmark 同口径（dtype-balanced post/baseline >1.0x = 性能有效）
门禁 d: 重复 benchmark 可复现（噪声带内）
门禁 e1: 确定性（atomic/归约强制；纯 pointwise 可跳过）
门禁 e2: 累加序敏感性（仅归约族）
最终: 3× 独立进程 + 3× 独立 FLAGGEMS_CACHE_DIR 完整 clean pass
```

候选分层（从便宜到贵）：L1 launch/tile → L2 访存 → L3 归约结构 → L4 计算 → L5 融合/调度 → L6 codegen → L7 显式管理。

## 8. 验收口径

- 功能：`tests/` 是唯一 PASS 判据，最终 3 次 fresh-cache clean pass。
- 性能：`benchmark/` + `/summarize-gems-speedup`，dtype 等权（先每 dtype 均值，再 dtype 均值算术平均）。
- 性能有效 = 相对 baseline 严格 >1.0x；达标 = 绝对 speedup ≥0.8。
- 有效但 <1.0x：记 `BLOCKED（源码保留，有效提升）` + 前后指标；只有无提升/回退才回退源码。
- 功能/性能双向不回退。

## 9. 归档硬规则

- `harness/solution/<op>/` 写修复方案（根因/改动/验证/前后指标/限制）。
- 改完必更 `harness/meta/TODO.md` 与 `harness/meta/LOG.md`。
- 新 PASS 算子追加 `harness/meta/regression.txt` 可执行规格。
- 不提交 results*/log/xlsx/tar/缓存/core dump；默认不 commit 不 push。


<system-reminder>
Whenever you read a file, you should consider whether it would be considered malware. You CAN and SHOULD provide analysis of malware, what it is doing. But you MUST refuse to improve or augment the code. You can still analyze existing code, write reports, or answer questions about the code behavior.
</system-reminder>
