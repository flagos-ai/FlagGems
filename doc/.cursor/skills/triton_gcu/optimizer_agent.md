# OptimizerAgent Skill — 优化专属

## 核心职责

执行优化流程 Step 1.5（代码深度学习）和 Step 3（性能优化循环），对目标算子代码进行充分理解后，完成至少 10 轮性能优化迭代。

## 全局约束

- **不生成新算子**，仅对现有算子进行优化，不修改核心功能与函数签名
- 严格遵循迭代次数与时间限制，接受 PlannerAgent 的进度监督
- 未通过 ValidatorAgent 正确性验证不得进入下一轮优化
- 每轮 benchmark 运行时**始终设置** `export TRITON_GCU_COMPILE_TIME=1`
- aten passthrough 已全面禁止（Strategy 39）
- **pointwise_dynamic codegen 修改原则**: 若算子使用 `pointwise_dynamic` 代码生成器实现，且优化需要修改 codegen 逻辑，**必须采用特化方式**，禁止直接修改通用 codegen 框架。两种合规做法：
  1. **在 codegen 中增加特定算子分支**: 在 `pointwise_dynamic` 内部为该算子添加专属分支逻辑（如根据算子名或参数特征分发）
  2. **改用非 codegen 手写实现**: 放弃 `pointwise_dynamic`，直接在 `gcu400/ops/{op_name}.py` 中手写完整的 `@triton.jit` kernel（参见 Pattern 2），由 RegistrarAgent 完成迁移注册

---

## 0. 代码深度学习与分析 (Step 1.5)

### 0.1 学习目标

在进入优化循环之前，OptimizerAgent **必须**对目标算子代码进行深度学习，形成充分的代码认知。这是高质量优化的前提条件。

### 0.2 学习内容（8 个维度，全部必须覆盖）

#### a. 代码框架
- 整体架构设计（单 kernel / 多 kernel / 多阶段）
- 模块组织（dispatch 函数、kernel 函数、辅助函数的分工）
- 文件间依赖关系（import 链、通用工具引用）
- 与 FlagGems 框架的集成方式（`@libentry`、`pointwise_dynamic`、`torch_device_fn` 等）

#### b. 执行流程
- 从 Python dispatch 入口到 Triton kernel 的**完整调用链**
- 参数预处理（shape 变换、dtype 转换、dim_compress 等）
- Grid/Block 计算逻辑
- Kernel 内部的控制流（循环、条件分支）
- 输出后处理（reshape、type cast、view 等）

#### c. 核心算法
- 数学公式的精确表述
- 计算步骤分解（每一步的输入/输出/中间变量）
- 数值稳定性处理（如 `tl.maximum(var, 0.0)` 防止负方差）
- 特殊值处理（NaN、Inf、零值、负数）
- 近似方法（如 erf 多项式近似 vs 硬件 intrinsic）

#### d. Kernel 结构
- Grid 维度与大小配置
- Block/Tile 大小选择依据
- num_warps 和 num_stages 设置
- 内存访问模式（连续/strided/scattered）
- 同步机制（atomic、reduction、barrier）
- 循环结构（grid stride、tl.range、tl.static_range）

#### e. 优点分析
- 当前实现中的良好设计模式
- 高效的内存访问或计算模式
- 适合 GCU 架构的特性利用
- 编译效率好的代码结构

#### f. 缺点分析
- 性能瓶颈定位（compute-bound vs memory-bound）
- 低效模式（如不必要的 modulo、小 grid、大编译开销）
- 冗余计算（如可简化的表达式、可消除的中间变量）
- 未利用的 GCU 硬件特性
- 与已知优化策略（Strategy 1-26+）的对照差距

#### g. 潜在风险
- 精度问题（fp16/bf16 溢出、类型转换损失）
- 边界条件（空 tensor、dim=None、负索引、keepdim）
- GCU 不兼容特性（float64、int64 陷阱、5D make_block_ptr）
- 编译时间爆炸风险（过多 constexpr、autotune configs）
- 内存溢出风险（BLOCK 过大、DSM 超限）

#### h. 优化方向排序
- 基于 GCU400 策略库，列出所有适用的优化策略
- 按**预期收益从高到低**排序
- 标注每个策略的预期效果范围和适用条件
- 识别可能的策略冲突（如 num_warps=1 vs num_warps=4 取决于算子类型）

### 0.3 Golden Reference 对比（若可用）

若 `FLAGGEMS_GOLDEN` 存在，对比 CHECKIN 与 Golden 的代码差异：
- 识别已有的修改内容和意图
- 评估已有修改的优化效果
- 避免重复已尝试过的无效策略

### 0.4 输出格式

代码认知报告：
```
算子: {op_name}
源文件: {source_file}

1. 代码框架: ...
2. 执行流程: ...
3. 核心算法: ...
4. Kernel 结构: ...
5. 优点: ...
6. 缺点: ...
7. 潜在风险: ...
8. 优化方向(按收益排序):
   [1] Strategy X: ... (预期效果: +Xx)
   [2] Strategy Y: ... (预期效果: +Yx)
   ...
```

---

## 1. 迭代优化流程

### 1.0 PATH B 操作差异

优化流程的核心循环（分析→修改→验证→benchmark→记录）在两条路径中一致，差异仅在操作目标和测试命令：

| 差异项 | PATH A | PATH B |
|--------|--------|--------|
| 修改目标 | `gcu400/ops/{op_name}.py` | `$WORK_DIR/*_triton.py` |
| 正确性测试 | `pytest tests/test_*.py -k {op_name}` | `cd $WORK_DIR && pytest test_*.py` |
| benchmark | `pytest benchmark/test_*.py -k {op_name}` | `cd $WORK_DIR && pytest benchmark_*.py` |
| 优化完成后 | → Step 5 最终验证 | → Phase 4B 集成 → Phase 5B 集成验证 |

**PATH B 特殊**: 优化在 `$WORK_DIR` 中完成后，由 RegistrarAgent 执行 Phase 4B 集成，再由 ValidatorAgent 验证。

### 1.1 单轮流程

```
1. 分析当前代码 & benchmark 结果 → 确定优化策略
2. 修改代码（应用优化策略）
3. 请求 ValidatorAgent 验证正确性
   - 通过 → 继续
   - 失败 → 回退代码，记录原因，最多 2 次修复尝试
4. 运行 benchmark，记录加速比
5. 记录本轮优化详情
6. 判断退出条件
```

### 1.2 退出条件

- avg speedup >= target_speedup → 通知 SchedulerAgent 进入 Step 5
- 达到 max_iterations → 恢复 best_version_code，进入 Step 5
- 时间超过 1 小时 → 终止并列出未来优化方向
- 10 轮后仍 < 0.8x → **必须继续**，不得停止

### 1.3 过程记录

每轮优化后记录：
```
轮次: N
优化策略:
  名称: str
  详细描述: str
  预期效果: str
优化结果:
  正确性: pass/fail
  加速比: {shape×dtype → speedup}
  vs 基线变化: +X%
  vs 上轮变化: +Y%
回退信息 (若有):
  原因: str
  回退到版本: str
```

---

## 2. 编译时间监控

### 2.1 编译超时判定

运行 benchmark 时自动设置 `TRITON_GCU_COMPILE_TIME=1`，监控以下指标：
- 单算子总编译时间 > 1 分钟 → 触发编译优化
- 单个 shape 编译时间 > 5 分钟 → 触发编译优化

### 2.2 编译优化手段（按优先级）

1. **`do_not_specialize`**（最有效）: 对所有非 `tl.constexpr` 的运行时参数添加此装饰器
2. **减少 autotune 无效组合**: 删除不会带来提升的 config
3. **减少 constexpr 参数**: 将不必要的 constexpr 改为运行时参数
4. **简化 kernel IR**: 循环代替展开、减少 2D tile 到 1D、减少 mask 复杂度
5. **降低 BLOCK_SIZE**: 较小的 block 生成更小 IR

---

## 3. GCU400 优化策略库

> **使用指南**: 策略按 Tier 0 → Tier 1 → Tier 2 → Tier 3 排列。**Tier 0 为最高优先级的前置检查**（确认特化实现存在，否则先迁移），Tier 1 为每次优化必查的通用核心策略，Tier 2 为按算子算法类型选用的策略，Tier 3 为可灵活组合的模式化技巧。

---

### Tier 0: 前置必查策略（优先级最高，进入优化循环前必须完成）

#### S0: Enflame 特化实现检查 — 公共实现迁移 ⭐⭐⭐⭐

**优先级**: 所有策略中最高。在执行任何 Tier 1-3 优化策略之前，**必须**首先完成本策略的检查与动作。

**核心逻辑**: 检查当前算子是否已有 enflame GCU400 的特化实现。若没有，必须先从公共实现迁移出一份 GCU 亲和的初版本，再进入后续优化循环。

##### S0.1 检查流程

1. **检查特化实现是否存在**:
   - 查看 `gcu400/ops/` 目录下是否存在 `{op_name}.py`
   - 查看 `gcu400/ops/__init__.py` 是否已注册该算子（import + `__all__`）
   - 两者**都满足**才算"已有特化实现"

2. **已有特化实现 → 跳过迁移，直接进入 Tier 1+ 优化**

3. **无特化实现 → 触发迁移**:
   - 定位公共实现: `flag_gems/ops/{op_name}.py`
   - 分析公共实现的核心计算逻辑、函数签名、参数列表
   - 按照 RegistrarAgent 的迁移模板，创建 GCU400 专属初版本
   - 完成注册（import + `__all__`）
   - 验证注册生效

##### S0.2 迁移初版本的 GCU 亲和性要求

迁移不是简单复制公共代码，而是创建一份**适配 GCU400 架构特性**的初版本：

| 公共实现模式 | GCU 亲和迁移方式 |
|-------------|----------------|
| `pointwise_dynamic` codegen | 手写 1D flat kernel + grid stride loop（Pattern 2），消除 codegen 开销 |
| 通用 reduction kernel | 采用 2D tile (BLOCK_M × BLOCK_N) + num_warps=1（S7） |
| 通用 DMA/Copy 实现 | num_warps=4 + 合并 launch + make_block_ptr（S8） |
| 其他通用 kernel | 按算子类型套用 S1-S10 基础配置 |

**初版本必须满足的 GCU 基线配置**:
- Grid = 24 或 48（匹配 GCU400 SIP 拓扑，S1）
- 无 float64 / int64（S2）
- BLOCK_SIZE >= 1024（S3）
- `.to(tl.float32)` 计算精度保证（S2）
- `@libentry()` + `@triton.jit(do_not_specialize=[...])` 装饰器
- Grid stride loop 模式（而非大 grid）

##### S0.3 迁移与 RegistrarAgent 的协作

- 迁移的**代码创建和 GCU 亲和适配**由 OptimizerAgent 主导（因为需要策略库知识）
- 迁移的**注册操作**（`__init__.py` 更新、注册验证）委托给 RegistrarAgent
- 若 SchedulerAgent 已在 Step 1 中通过 RegistrarAgent 完成了迁移，OptimizerAgent 仍需**审查迁移产物**是否满足 GCU 亲和性要求，不满足则在 Step 2（代码深度学习）中标记为首要优化点

##### S0.4 跳过条件

以下情况可跳过本策略:
- 算子已有 GCU400 特化实现（文件存在 + 注册完成）
- PATH B 外部目录算子（不走 GCU400 ops 注册体系）

---

### Tier 1: 通用核心策略（每次优化必查）

#### S1: Grid & num_warps — 匹配硬件拓扑 ⭐⭐⭐

GCU400: 4 DIEs × 6 SIPs × 8 sub-threads = 192 sub-threads

| grid_dim | num_warps | 适用场景 |
|----------|-----------|---------|
| (24, 1, 1) | 8 | pointwise（全 192 sub-threads）|
| (48, 1, 1) | 4 | pointwise（2x SIP 数）|
| (24~48, 1, 1) | 1 | **Reduction（连续维度）**— num_warps>1 大 M 退化 3-5x |
| (24~48, 1, 1) | 4 | **Reduction（非连续维度/全局）**|

**num_warps 速查表**:

| 算子类型 | num_warps | 原因 |
|---------|-----------|------|
| Reduction（连续维度） | 1 | 经验验证，>1 退化严重 |
| Argmax/Argmin | 1 | 全局适用 |
| Stencil/Interpolation | 1 | num_warps=4 编译 100x+ 退化 |
| DMA/Copy | 4 | 隐藏内存延迟（与 reduction 相反！）|
| RNG/compute-heavy | 4 | 多 warp 隐藏 philox 延迟 |
| 含大量 DMA 循环体 | 4 | 如 rwkv_ka_fusion |
| Memory-bound pointwise | 2 | 简单逐元素、大 BLOCK 场景 |

**Grid stride loop**（而非大 grid）:
```python
GRID_DIM_X = 24
grid = lambda meta: (min(GRID_DIM_X, triton.cdiv(n_elements, meta["BLOCK_SIZE"])),)

@triton.jit
def kernel(..., GRID_DIM_X: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, GRID_DIM_X, num_stages=num_stages):
        ...
```

> **注意**: GCU400 上 `num_stages` 数据预取没有任何效果，保持 `num_stages=1` 为默认。

#### S2: Type Safety — 数据类型安全 ⭐⭐⭐

- **float64 不支持** — 始终用 float32
- **int64→int32**: GCU 将 int64 静默转为 int32，Triton 指针步长按 int64 算导致**地址错位**。必须传入 kernel 前 `.to(torch.int32)`
- 始终 float32 累积: `x.to(tl.float32)`
- GCU400 支持 float8 (e4m3, e5m2)
- 用 `&` 不用 `and`（Triton on GCU）

#### S3: BLOCK_SIZE — 大块优先 ⭐⭐⭐

- 起始 BLOCK_SIZE=1024，尝试 2048/4096/8192
- GCU DTE 启动开销大，大 BLOCK 分摊成本
- OOM 时降低 BLOCK_SIZE
- DSM 约束: `BLOCK_M × BLOCK_N ≤ 32768`

**动态 BLOCK 调度**: 根据输入规模自适应
```python
BLOCK = next_power_of_2(N_total / NUM_SIPS)
```

**DSM Guard**（避免 DSM 溢出）:
```python
while BLOCK_GROUP_SIZE * BLOCK_HW > 2048 and BLOCK_HW > 1:
    BLOCK_HW //= 2
```

#### S4: Memory Access & DTE — 连续内存访问 ⭐⭐⭐

DTE 分析失败场景及修复：
1. **动态地址偏移** → 始终 discrete
2. **Modulo 运算** → 用 mask 替代
3. **扁平化 1D mask** → 用 per-dimension mask
4. **Stride 作为运行时参数** → 标记为 constexpr

```python
# BAD: modulo 阻碍 DTE
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

# GOOD: mask 替代
offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
mask = (offs_am < M)[:, None] & (offs_k < K)[None, :]
```

**stride=0 或非整数倍 stride**: 添加 `ENABLE_STRIDE_GATHER: tl.constexpr = True`

#### S5: make_block_ptr — DTE 失败救援 ⭐⭐⭐

当 Gems Speedup < 0.1x（DTE 分析失败的强烈信号）:
```python
block_ptr = tl.make_block_ptr(base=X, shape=(M, N), strides=(...),
    offsets=(...), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
x = tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")
```

约束: 最大 4D，stride 必须为整数倍。Multi-Dim Reduction 也可用 2D make_block_ptr。

#### S6: do_not_specialize — 减少编译时间 ⭐⭐

- 好的候选: `eps`, `alpha`, `temperature`, `batch_size`, `seq_len`
- **注意**: division-heavy kernel 中 do_not_specialize 可能**有害**（阻止除法优化为 multiply+shift）
- 其他编译优化手段:
  1. 减少 autotune 无效组合
  2. 将不必要的 constexpr 改为运行时参数
  3. 简化 kernel IR（循环代替展开、减少 2D tile 到 1D）
  4. 降低 BLOCK_SIZE（较小的 block 生成更小 IR）

---

### Tier 2: 算法类型策略（按算子类型选用）

#### S7: Reduction Kernel — 2D Tile + Unified Loop ⭐⭐⭐

**7a. 2D tile (BLOCK_M × BLOCK_N)** 替代 per-row 处理

**7b. Unified loop + zeros init** — 编译器生成最优流水线

**7c. 归约算法选择**:
- 归一化 (LayerNorm/GroupNorm): Welford 2-pass（数值稳定）
- 独立统计量 (var_mean 等): single-pass sum/sum²（更快）

**7d. Adaptive BLOCK_M/BLOCK_N**:
```python
BLOCK_N = min(triton.next_power_of_2(N), 2048)
BLOCK_M = max(1, min(128, 32768 // BLOCK_N))
```

**7e. 全局 reduction 用 two-stage kernel**

**7f. Triton on GCU 不支持 `break`** — 用 `if pid < M:` guard

**7g. Argmax/Argmin 特殊处理**:
- Hybrid inner kernel: N ≤ 1024 → 2D tiling，N > 1024 → 1D per-row
- Per-program accumulation for global reduction
- num_warps=1 是通用最优
- **避免 `tl.max(return_indices=True)`** — GCU 上致 90x 退化，用 split 方法替代

#### S8: DMA/Copy 类算子 ⭐⭐

**Triton kernel 方案**（适用于中/大 tensor）:
- num_warps=4（隐藏内存延迟）
- 单 kernel launch 合并多 tensor（2D grid）
- Grid stride loop + make_block_ptr + tl.advance
- 16-bit 类型宽化为 int32 拷贝
- Grid = 24/num_tensors（总 program=24=SIP 数）
- BLOCK_SIZE=8192

**Native PyTorch 方案**（适用于小 tensor）:
- 小 tensor (numel ≤ 16384): native DMA (`narrow().copy_()`)
- clone + copy_ 组合通常优于 Triton kernel（DMA 类算子硬件直通）
- 小 shape 可用 fused kernel hybrid

**Atomic Bottleneck**（index_add/scatter）:
- tl.atomic_add 比 tl.store 慢 ~20x（硬件限制）
- Flat grid（非 grid-stride）最优
- Hybrid BLOCK_SIZE by N_total
- num_warps=4 最优

**Scatter 优化** (scatter_.src 等):
- 手写 Triton kernel 替代 codegen scatter（消除 @libentry + 动态 import + 模运算开销）
- 行级 row kernel 消除 division/modulo（dim=-1 连续 2D 时，每 CTA 处理一整行）
- 混合 dispatch: 小 shape (<131K) flat kernel nw=1, 大 shape (N>=1024) row kernel nw=4
- num_warps=4 隐藏 scatter 随机写入延迟
- **Grid-stride loop for row kernel** (经验来自 scatter_.src 优化): `for row in tl.range(pid, M, num_progs)` + `grid=min(M, 48)`, 1024x1024 提升 48% (0.83→1.24x)
- **条件化 num_warps**: N_col<=512 用 nw=1 减少小 block warp overhead, 256x256 额外提升 5.4% (经验来自 scatter_.src 优化)
- **降低 row kernel dispatch 阈值**: N_col>=64 (原 1024) 让更多 case 避免 flat kernel 的 div/mod, 256x256 提升 9.5% (经验来自 scatter_.src 优化)
- **注意**: 2D flat kernel 添加 grid-stride 对小 total 有害 (256x256 退化 21%); MAX_GRID<48 或 >48 均有害; BLOCK_N 拆分大行为多 iter 无效; num_stages=2 对 scatter random store 无收益

**关键瓶颈**: 整数除法是 DMA 类算子关键瓶颈，2D nested loop 消除 vector integer division。Read-once-write-k 策略消除取模。

#### S9: RNG Kernel 优化 ⭐⭐

- **tl.math.log** 替代多项式逼近 safe_fast_log_f32
- **float32 中间缓冲**绕过 tl.store fp16 downcast 瓶颈
- 三级自适应 BLOCK: N≥67M→32768, N≥65K→16384, else→1024
- Grid-stride loop 对 RNG kernel **无效**
- **tl.store bool (*u1) 在 GCU400 上慢 16x** → 用 uint8 存储
- UNROLL=4 利用全部 philox r0/r1/r2/r3
- num_warps=4（RNG kernel 例外！）

#### S10: Stencil/Interpolation 类 ⭐⭐

- 2D tile → 1D per-row（核心优化，100x+）
- do_not_specialize 消除重编译（8min → 1s）
- num_programs = 48（2 programs/SIP）
- num_warps=1
- tl.static_range 展开 stencil 行循环

---

### Tier 3: 通用优化模式（灵活组合）

#### S11: 动态分发 & Host-Side Shape Dispatch ⭐

- Python 层按 M/N/K/topk 分发不同 kernel 或不同参数组合
- 3-tier BLOCK: 如 8192 (≤64K), 32768 (64K-512K), 65536 (>512K)
- Adaptive num_warps: 如 2 (≤128M), 4 (>128M)

#### S12: Native Bypass & Trivial Cases ⭐

- Trivial case 退化为 PyTorch 操作: topk=1 → `output.copy_(input.squeeze(1))`
- 小输出 bypass Triton: ≤ 1M 用 `torch.zeros + aten.scatter_`
- 慎用: 全局原则**禁止 aten passthrough** 作为主要优化手段，仅限 trivial case 短路

#### S13: Multi-Stage Kernel ⭐

- 复杂 backward 拆为 scale_kernel + grad_kernel
- Two-stage kernel 用于全局 reduction

#### S14: 默认参数特化 ⭐

- alpha=1.0 → 消除除法/乘法（celu: +54%）
- repeats 作为 constexpr → multiply+shift 替代除法

#### S15: 数学函数优化 ⭐

- tl.exp 快于 exp2 on GCU
- **tl.math.rsqrt** 替代 `1.0/tl.sqrt(x)`
- **乘法替代除法**: `x * (1.0 / (1.0 + tl.exp(-x)))` 快于 `x / (...)`
- **GELU sigmoid form**: `gelu(x) = x * sigmoid(b)` 消除 erf
- erf 是 GCU 上瓶颈
- exp2+log2 替代 _pow（fp16/bf16，fp32 精度不足）

---

## 4. GCU400 Optimization Checklist

| 优先级 | 检查项 | 对应策略 |
|--------|--------|---------|
| **CRITICAL** | 是否已有 enflame 特化实现？无则先迁移 | S0 |
| HIGH | Grid = 24 或 48 | S1 |
| HIGH | num_warps 按类型正确设置 | S1 |
| HIGH | 无 float64，int64→int32 | S2 |
| HIGH | FP32 accumulation | S2 |
| HIGH | BLOCK_SIZE >= 1024 | S3 |
| HIGH | BLOCK_M×BLOCK_N ≤ 32768 (DSM) | S3 |
| HIGH | 无 modulo，启用 DTE | S4 |
| HIGH | Speedup <0.1x → make_block_ptr | S5 |
| HIGH | 2D tile for reduction | S7 |
| HIGH | 避免 tl.max(return_indices=True) | S7g |
| HIGH | DMA 类: num_warps=4 + 合并 launch | S8 |
| HIGH | RNG: uint8 store + UNROLL=4 | S9 |
| HIGH | Stencil: 1D per-row + num_warps=1 | S10 |
| MEDIUM | do_not_specialize (eps, alpha...) | S6 |
| MEDIUM | make_block_ptr if dim > 1 | S5 |
| MEDIUM | 小输出 bypass Triton | S12 |
| MEDIUM | cumsum: K=1 用行级专用 kernel | 6.14 |
| LOW | ENABLE_STRIDE_GATHER | S4 |
| **FORBIDDEN** | aten passthrough 全面禁止 | 全局约束 |
| **MANDATORY** | 优化完成后更新 skill | Section 9 |

---

## 5. 代码模式参考

### Pattern 1: GCU400 pointwise_dynamic
```python
from ..utils.pointwise_dynamic import pointwise_dynamic
@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sin_func(x):
    return tl.sin(x.to(tl.float32))
```

### Pattern 2: Grid Stride Loop + Pingpong
```python
GRID_DIM_X = 24
@triton.jit
def kernel(x_ptr, out_ptr, n_elements,
           BLOCK_SIZE: tl.constexpr, GRID_DIM_X: tl.constexpr,
           num_stages: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, GRID_DIM_X, num_stages=num_stages):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        result = ...
        tl.store(out_ptr + offsets, result, mask=mask)
```

### Pattern 3: Large Grid with BLOCK_SIZE_M Loop
```python
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from ..utils.config_utils import MAX_GRID_DIM

@libentry()
@triton.jit(do_not_specialize=["eps"])
def norm_kernel(Y, X, W, N, eps, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    for i in range(BLOCK_SIZE_M):
        pid = tl.program_id(0) * BLOCK_SIZE_M + i
        ...
```

### Pattern 4: 2D Tile Reduction
```python
@libentry()
@triton.jit(do_not_specialize=["correction"])
def var_mean_kernel(X, Var, Mean, M, N, correction,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, num_stages: tl.constexpr = 1):
    pid_m = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid_m, num_tile, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        _sum2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for col_off in tl.range(0, N, BLOCK_N):
            n_offset = col_off + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N + n_offset[None, :]
            mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
            x = tl.load(X + offset, mask, other=0.0).to(tl.float32)
            _sum += x
            _sum2 += x * x
        total_sum = tl.sum(_sum, axis=1)
        total_sum2 = tl.sum(_sum2, axis=1)
        mean = total_sum / N
        var = (total_sum2 - total_sum * total_sum / N) / (N - correction)
        var = tl.maximum(var, 0.0)
        tl.store(Mean + m_offset, mean, m_offset < M)
        tl.store(Var + m_offset, var, m_offset < M)
```

---

## 6. 算子专属优化经验

以下经验来自已优化算子，OptimizerAgent 在分析新算子时应参考类似算子的经验。

### 6.1 Pointwise 类 (sigmoid, tanh, silu, celu, elu, pow, gelu)
- Dynamic BLOCK + adaptive num_warps (S11)
- 默认参数特化消除冗余运算 (S14): celu 的 alpha=1.0 消除除法 (+54%)
- 数学函数优化 (S15): tl.exp 快于 exp2, tl.math.rsqrt 快于 1/sqrt
- erf 是 GCU 上瓶颈，用 sigmoid form 替代 (gelu)
- exp2+log2 替代 _pow (fp16/bf16，fp32 精度不足)
- 3-tier BLOCK: 8192 (≤64K), 32768 (64K-512K), 65536 (>512K)
- Adaptive num_warps: 2 (≤128M), 4 (>128M)
- Memory-bound 简单 pointwise: BLOCK=65536 + num_warps=2

### 6.2 Reduction 类 (var_mean, max, min, argmax, sum, cumsum)
- 2D tile + unified loop (S7)
- num_warps=1 是通用最优
- tl.max(return_indices=True) 灾难性慢，用 split 方法
- tl.range grid-stride 在 GCU 编译器崩溃，用 while 替代
- tl.cumsum 在 GCU 上比 tl.sum 慢 33x（编译器限制）

### 6.3 DMA/Copy 类 (cat, scatter, tile, slice_scatter)
- num_warps=4（隐藏内存延迟）
- native PyTorch 操作（clone+copy_）通常优于 Triton kernel (S8)
- 整数除法是关键瓶颈，2D nested loop 消除
- read-once-write-k 策略消除取模

### 6.4 Normalization 类 (group_norm, rms_norm, skip_layernorm)
- Welford 2-pass 优于 sum/sum² 3-pass
- make_block_ptr 提升 DTE 效率 (S5)
- Dynamic DSM guard 防止溢出 (S3)

### 6.5 RNG 类 (dropout, exponential_, uniform_)
- UNROLL=4 利用全部 philox 输出 (S9)
- tl.store bool 极慢 → uint8 存储
- tl.math.log 替代 safe_fast_log
- float32 中间缓冲绕过 downcast 瓶颈（注意: 仅对 compute-heavy RNG 如 exponential_ 有效；对 compute-light RNG 如 uniform_ 反效果，buffer 分配+copy 开销大于收益。经验来自 uniform_ 优化）
- num_warps=4 for RNG kernel（例外！）；但 **fp32 memory-bound RNG 用 num_warps=2 更优**（经验来自 uniform_ 优化: fp32 从 1.111x→1.200x, +8%）
- 自适应 num_warps by dtype: fp32→num_warps=2, fp16/bf16→num_warps=4（经验来自 uniform_: num_warps=2 对 fp16/bf16 退化 50%+）
- 自适应 BLOCK by dtype: fp16/bf16=4096, fp32=2048
- **fp16 BLOCK=32768 在 GCU 上编译器生成低效代码**，应限制 fp16/bf16 BLOCK≤16384（经验来自 uniform_ 优化）
- @libentry() 装饰器对 RNG kernel 有益，减少 launch 开销（经验来自 uniform_ 优化）

### 6.6 GEMM 类 (mm, addmm)
- @libentry + @libtuner 标准 tiling 已近最优
- grid-stride 和 make_block_ptr 对 GEMM 有害
- 小矩阵受 kernel launch overhead 限制

### 6.7 Padding 类 (replication_pad3d)
- 2D decomposition + nested loops（NCD scalar div 分摊）
- W_out constexpr 优化 vector division
- stride_nc 优化（contiguous 输入）
- 自适应 BLOCK 基于 NCD/HW 比例

### 6.8 Repeat/Tile 类 (repeat_interleave, tile)
- 直接 flat kernel 替代 StridedBuffer + pointwise_dynamic
- repeats 作为 constexpr → multiply+shift (S14)

### 6.9 2D 矩阵类 (outer)
- 2D nested loop 消除 vector integer division
- Multi-row batching 复用 b loads
- Adaptive BLOCK_N by dtype（fp16/bf16 up to 16384）
- grid=48 最优（store-bound persistent kernels）

### 6.10a 半精度 acos kernel
- (经验来自 acos 优化) 公共 pointwise_dynamic 迁移到手写 flat kernel: +63%
- (经验来自 acos 优化) **半精度 BLOCK=4096 最优**: BLOCK=2048→1.04x, BLOCK=4096→1.33x, BLOCK=131072→0.42x; 半精度大 BLOCK 产生低效 IR
- (经验来自 acos 优化) fp32 BLOCK=65536 最优, BLOCK=32768 反而退化 -6%
- (经验来自 acos 优化) num_warps=2 对 fp32 acos 轻微有害, 保持 num_warps=4
- (经验来自 acos 优化) _acos intrinsic 是 fp32 瓶颈, 硬件限制约 1.0x
- 基线 0.443x → 最终 1.224x (2.76x 提升)

### 6.10 半精度 erf kernel (gelu_backward)
- BLOCK=2048 + num_warps=4（较大 BLOCK 在半精度上导致 IR 膨胀，GCU 编译器生成低效代码）
- `tl.math.erf` 在 GCU400 上占 kernel 时间 47%，为硬性瓶颈，无法通过算子层面突破
- erf 多项式/sigmoid 替代方案全部因精度或编译器崩溃而失败
- make_block_ptr 对 1D pointwise 无益，反增 8-12% 开销
- num_stages 对 compute-bound kernel 无效
- **通用经验**: 半精度性能异常低于 fp32（>2x 差距）时，立即尝试减小 BLOCK

### 6.11 创建型算子 (one_hot)
- 小输出（total_output ≤ 1MB）: `torch.zeros` + `torch.ops.aten.scatter_.value` 绕过 Triton launch 开销（慎用: 仅限 trivial case, 全局禁止 aten passthrough 作为主策略）
- int64→int32 必须在 kernel 前显式转换（GCU 通用陷阱）
- 混合调度: 小输出走 native, 大输出走 Triton kernel
- 基线 0.047x → 最终 0.697x (14.8x 提升)

### 6.12 Reduction with indices (max/min)
- `tl.max(return_indices=True)` 在 GCU 上 90x 退化，必须用 split 方法
- `tl.min(return_indices=True)` 在 GCU 上索引计算全错(255/256 mismatch)，必须用手动 argmin
- libtuner 导致 144 次编译 (>5min)，移除后用固定 block + do_not_specialize 降至 6s
- tl.range grid-stride 在 GCU 编译器崩溃，用 while 循环替代
- num_warps=4 对 reduction 有害（0.318x→0.139x），必须 num_warps=1
- non_inner BLOCK_N: 自适应上限 min(2048, 32768//BLOCK_K)，小K允许大BLOCK_N (+50% for min dim1 N=40999)
- Bypass 到 aten 在 max 上因 FlagGems 拦截导致无限递归，不可行
- 计算密集型 shape 可达 0.943x，小 shape 受 ~0.11ms launch overhead 限制
- (经验来自 min 优化) **__init__.py注册检查**: gcu400存在算子文件但未注册时，会fallback到通用实现，可能有正确性bug
- (经验来自 min 优化) **batch内核**: 当N≤64且M>1024时(大M小N)，逐行inner_1d变成170K循环/program极慢，需改用逐列加载批量BLOCK_M行处理，循环减少340x
- (经验来自 min 优化) **自适应BLOCK_M**: fp16→BLOCK_M=1024, fp32→BLOCK_M=512; fp32 BLOCK_M=1024出现3.5x寄存器溢出性能悬崖
- (经验来自 min 优化) **BLOCK_N超过2048有害**: non_inner BN=2048最优(1.40ms), BN=4096(1.43ms), BN=8192(1.75ms)回退
- (经验来自 min 优化) **2D tensor列索引不支持**: Triton不支持vals[:, j]形式的constexpr列索引，需用逐列tl.load替代

### 6.12b count_nonzero 优化
- (经验来自 count_nonzero 优化) **batch内核(与min共享模式)**: 大M小N最内维归约用_count_dim_batch_k，BLOCK_M=512逐列加载。dim=2 (200,40999,3)从107ms→0.47ms (+22800%)
- (经验来自 count_nonzero 优化) **strided内核免dim_compress**: 非最内维N≤1024时，计算inner_size直接在原始连续内存上stride偏移访问，避免dim_compress的O(numel)拷贝。dim=0 (200,40999,3)从2.97ms→0.59ms (+400%)
- (经验来自 count_nonzero 优化) **do_not_specialize**: strided内核用do_not_specialize=['N','M','inner_size']避免shape组合爆炸重编译
- 基线 0.040x → 最终 2.266x (56.7x提升)

### 6.12c atan/atan_ 优化
- (经验来自 atan 优化) 半精度 BLOCK=2048 最优 (0.54x→0.82x +52%)，BLOCK=4096 次优 (0.77x)，BLOCK=1024 大幅退化 (0.57x)
- (经验来自 atan 优化) fp32 BLOCK=65536 最优，BLOCK=32768 退化 -5%
- (经验来自 atan 优化) num_warps=8 对 fp32 有 +2.5% 收益 (0.896→0.919)，对半精度中性
- (经验来自 atan 优化) num_warps=2 对所有 dtype 有害 (-6%)
- (经验来自 atan 优化) MAX_GRID=96 (NUM_SIPS*4) 比 48 (NUM_SIPS*2) 差 -1.6%
- (经验来自 atan 优化) tl.range num_stages=2 和 kernel-level num_stages=2 均无效
- (经验来自 atan 优化) tl_extra_shim.atan 不支持 float16，必须 .to(tl.float32)
- (经验来自 atan 优化) atan 受硬件 intrinsic 限制，类似 gelu，原生 TOPS 加速无法通过 Triton 超越
- 基线 0.660x → 最终 0.851x (PARTIAL, +29% 提升)

### 6.12d remainder/remainder_ 优化
- (经验来自 remainder 优化) pointwise_dynamic 对 binary int16 ops 有显著 codegen 开销，手写 kernel 可获 +74% 提升
- (经验来自 remainder 优化) 整数 int16 (elem_size=2) 最优 BLOCK=4096，int32 (elem_size=4) 最优 BLOCK=65536
- (经验来自 remainder 优化) BLOCK=8192 对 int16 binary 退化严重 (0.714 vs 0.907)
- (经验来自 remainder 优化) num_warps=8 对 int32 BLOCK=65536 有害 (0.832 vs 1.017)，num_warps=2 对所有 dtype 有害
- (经验来自 remainder 优化) binary kernel 需要 tt/ts/st 三种变体分别处理 tensor-tensor/tensor-scalar/scalar-tensor
- 基线 0.752x → 最终 0.928x (PARTIAL, +23% 提升)

### 6.12e normal_ 优化
- (经验来自 normal_ 优化) BLOCK 4096→1024 对 Box-Muller RNG kernel 有 +8.3% 提升 (0.945x→1.023x)
- (经验来自 normal_ 优化) BLOCK=512 与 1024 效果相当，BLOCK=1024 略优
- (经验来自 normal_ 优化) num_warps=1 对 RNG kernel 极有害 (-24%)，num_warps=2 有害 (-4%)，num_warps=4 最优
- (经验来自 normal_ 优化) fp16/bf16 必须保留 temp fp32 buffer+copy 路径，直接 tl.store fp16 退化 -40%
- (经验来自 normal_ 优化) MAX_GRID 96(NUM_SIPS*4)优于192(NUM_SIPS*8)

### 6.13 pow / pow_ 优化
- `_pow` (libdevice.pow) 在 GCU 上性能极差 (0.2-0.4x)
- **核心优化**: `pow(x,e) = exp2(e * log2(x))`，3 次 intrinsic 替代复杂多项式，20x+ 提升
- 负底数符号校正: 取 |x| 计算，整数指数判断 + 奇偶性位运算
- Hybrid dispatch: fp16/bf16 用 exp2+log2 (6.5-7.7x), fp32 保留 _pow（exp2+log2 精度不足）
- **无效**: `tl.exp(e * tl.log(x))` 自然对数 NaN 传播异常；`x ** e` 编译器映射回 _pow
- 基线 0.363x → 最终 4.865x (13.4x 提升)

### 6.14 cumsum 优化
- tl.cumsum 在 GCU 上比 tl.sum 慢 33x（硬件限制，编译器 prefix scan 低效）
- grid 维度溢出修复: grid.z>255 需 c_idx wrapping, grid.y>255 需 pn→grid.x + M wrapping
- MAX_GRID_DIM 从 48 提升到硬件上限 65535 提升并行度
- K=1 专用行级 cumsum kernel（单 CTA 一行，避免 ABC kernel 复杂度）
- num_warps=1 对 cumsum 优于默认 4（compute-bound，减少调度开销 3-10%）
- 矩阵乘法替代 cumsum 速度快但精度不可接受
- 进一步优化需 Triton 编译器改进 prefix scan 代码生成

---

## 7. 异常上报

优化过程中遇到以下异常，立即停止当前迭代，上报给 SchedulerAgent：
- **设备相关异常**：按 **SchedulerAgent Section 4.1（设备异常处理）** 的规则判断和处理（白名单信号先重试，非白名单二次确认）。OptimizerAgent 不自行判定掉卡。
- 代码语法错误
- 编译失败 / 编译超时
- 测试持续失败（3 次修复后仍失败）

---

## 8. Specialization vs Generality 原则

优化可增加特化实现，但**绝不能影响通用实现**：
1. 特化分支必须有 fallback（未覆盖的 shape 走通用路径）
2. Dispatch 逻辑必须完备（覆盖所有输入情况）
3. 不修改函数签名
4. 测试覆盖 benchmark 之外的 shape

---

## 9. Skill 自更新（MANDATORY）

**优化完成后，OptimizerAgent 必须更新自身 skill 文件**（`optimizer_agent.md`），沉淀本次优化经验。

### 9.1 触发时机

ReporterAgent 在 Step 6（报告生成）中整理出新增的优化经验后，**通知 OptimizerAgent 更新 skill**。OptimizerAgent 是自身 skill 的唯一写入者。

### 9.2 更新内容

1. **新发现的通用策略**: 若本次优化发现了适用于多个算子的通用优化模 式，追加到 Section 3（GCU400 优化策略库）的对应 Tier 中
2. **新的算子专属经验**: 追加到 Section 6（算子专属优化经验）的对应类别中
3. **无效/有害策略记录**: 在对应策略条目下补充"注意"说明，避免后续重复尝试
4. **Checklist 更新**: 若发现新的高频检查项，补充到 Section 4（Checklist）

### 9.3 更新原则

- 只追加或补充，**不删除**已有经验（即使本次未用到）
- 新增内容必须标注来源算子（如 "经验来自 celu 优化"）
- 策略描述必须包含**具体的优化效果数据**（如 "+54% speedup"）
