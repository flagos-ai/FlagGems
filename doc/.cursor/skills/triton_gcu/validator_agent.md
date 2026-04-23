# ValidatorAgent Skill — 验证专属

## 核心职责

执行优化流程 Step 2（初始验证与基线）、迭代中正确性验证、Step 5（最终验证），负责正确性验证与性能基准测试，确保优化过程中代码正确性。

## 全局约束

- **必须记录初始基线数据**，确保优化效果可量化对比
- 验证严格遵循测试规则，不遗漏关键测试 case
- **数据精度 > 性能** — 无论性能提升多大，精度不符必须回退
- **优先使用工程自带 pytest**，不得直接编写独立脚本替代

---

## 1. 初始验证 (Step 2)

### 1.1 运行正确性测试

#### PATH A (内置算子)
```bash
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest tests/ {TEST_CMD_FLAG} {op_name} -v 2>&1
```

#### PATH B (外部目录)
```bash
cd $WORK_DIR && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest test_*.py -v 2>&1
```

- 通过 → `safe_version_code` = 当前代码
- 失败 → Error: `ORIGINAL_FAILED`，上报 SchedulerAgent 终止

### 1.2 记录初始 Benchmark 基线 ⭐

**重要**: 优化前必须运行 benchmark 记录初始 speedup 基线。需要在**两个仓库**上分别获取数据。

#### 1.2.1 Golden Reference 基线（不变量）

首先在 `FLAGGEMS_GOLDEN` 上运行 benchmark，获取**未修改代码的原始性能**作为对照基准：

```bash
# 在 golden reference 上获取原始基线（PATH A 示例）
cd ${FLAGGEMS_GOLDEN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest -s -v benchmark/ -k {op_name} --level core --record log 2>&1
```

记录为 `golden_baseline_speedup`，此数据在整个优化过程中**不会改变**，用于衡量最终优化效果。

**单仓库模式说明**: 若 `FLAGGEMS_GOLDEN == FLAGGEMS_CHECKIN`（同一套代码），则跳过此步骤。在 Step 1.2.2 中获取的 CHECKIN 初始数据即为 golden 基线（`golden_baseline_speedup = initial_speedup`）。此数据必须在任何代码修改之前记录。

#### 1.2.2 CHECKIN 工作仓库基线

然后在 `FLAGGEMS_CHECKIN`（当前工作代码）上运行 benchmark：

```bash
# 在工作仓库上获取当前基线（PATH A 示例）
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest -s -v benchmark/ -k {op_name} --level core --record log 2>&1
```

记录为 `initial_speedup`，后续优化效果均与此值对比。

#### 1.2.3 FlagGems Benchmark 测试命令 (PATH A)

根据算子类型选择正确的测试文件：

| 算子类别 | Benchmark 文件 | 测试命令示例 |
|---------|---------------|-------------|
| Unary pointwise (sin, cos, relu...) | `benchmark/test_unary_pointwise_perf.py` | `pytest -s -v benchmark/test_unary_pointwise_perf.py -k {op_name} --level core --record log` |
| Binary pointwise (add, mul...) | `benchmark/test_binary_pointwise_perf.py` | 同上模式 |
| Reduction (var_mean, sum, max...) | `benchmark/test_reduction_perf.py` | `pytest -s -v benchmark/test_reduction_perf.py::test_general_reduction_perf[{op_name}] --level core --record log` |
| Norm (rms_norm, group_norm...) | `benchmark/test_norm_perf.py` | 同上模式 |
| BLAS (mm, bmm...) | `benchmark/test_blas_perf.py` | 同上模式 |
| Distribution (normal_, multinomial...) | `benchmark/test_distribution_perf.py` | 同上模式 |
| Fused ops (silu_and_mul...) | `benchmark/test_fused_perf.py` | 同上模式 |
| Select & Slice (index_select...) | `benchmark/test_select_and_slice_perf.py` | 同上模式 |
| Tensor constructor (zeros, ones...) | `benchmark/test_tensor_constructor_perf.py` | 同上模式 |
| Tensor concat (cat...) | `benchmark/test_tensor_concat_perf.py` | 同上模式 |

**关键参数**:
- `--level core` — 核心级 benchmark（推荐快速评估）
- `--level comprehensive` — 全量 benchmark（最终验证用）
- `--record log` — 记录结果
- `-s` — 显示 stdout（必需，用于看 benchmark 输出）
- 关注 **Gems Speedup** 列: >1.0 表示快于 native aten

**通用回退命令**（可在两个仓库上分别执行）:
```bash
# CHECKIN 工作仓库
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest -s -v benchmark/ -k {op_name} --level core --record log 2>&1

# Golden Reference（对照基准，可选）
cd ${FLAGGEMS_GOLDEN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest -s -v benchmark/ -k {op_name} --level core --record log 2>&1
```

**或用 pytest mark**:
```bash
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest benchmark/ {BENCH_CMD_FLAG} {op_name} -v -s 2>&1
```

#### PATH B Benchmark
```bash
cd $WORK_DIR && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest benchmark_*.py -v -s 2>&1
```

### 1.3 基线数据记录

解析 benchmark 输出，记录：
- `golden_baseline_speedup` = golden reference 上的平均 speedup（若可用）
- `initial_speedup` = CHECKIN 工作仓库上所有 case 的平均 speedup
- `initial_benchmark_details` = 每个 case 的详细数据（input size, torch time, triton time, speedup）
- 每种 dtype 的平均 speedup

输出初始基线摘要：
```
┌──────────────────────────────────────────────┐
│ Initial Benchmark Baseline (Pre-opt)          │
│ Operator: {op_name}                           │
│ Platform: Enflame GCU{arch_version}           │
│                                               │
│ Golden Ref Speedup: {golden_baseline}x        │  ← 不变量
│ CHECKIN Speedup:    {initial_speedup}x        │  ← 优化起点
│ Min Speedup:        {min_speedup}x            │
│ Max Speedup:        {max_speedup}x            │
│ Target Speedup:     {target_speedup}x         │
└──────────────────────────────────────────────┘
```

### 1.4 早期退出判断

若 `initial_speedup >= 1.2x`，通知 SchedulerAgent 可跳过后续优化。

---

## 2. 迭代验证 (Step 3 中每轮)

### 2.1 正确性验证要求

每轮优化后执行正确性验证：

1. **测试 case 数量**不低于 benchmark case 数量
2. 可适当减少但需包含典型 micro accuracy test
3. **【强制】优先使用工程自带 pytest**

```bash
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest tests/ {TEST_CMD_FLAG} {op_name} -v 2>&1
```

### 2.2 精度验证原则（不可违反）

1. **数据精度 > 性能**: 出现 mismatch 必须回退，不可接受
2. **每轮必验**: 避免在错误基础上继续优化
3. **最终轮次必须完整验证**: 运行完整 correctness test suite
4. **特化路径也要验证**: 每个分支都必须经过验证
5. **回归测试**: 优化某 shape 后验证其他 shape 未被破坏

### 2.3 工程自带 pytest 优先规则

**【强制】**:
- 必须优先使用工程自带 pytest（如 `tests/test_unary_pointwise_ops.py`）
- 遇到 collection error 时先尝试绕过：
  - `-p no:repeat` 禁用冲突插件
  - `--ignore` 跳过无关文件
- 确实无法修复才允许编写最小化独立测试脚本
- 独立脚本的 test case 和 tolerance 必须与工程 pytest 一致

### 2.4 验证结果反馈

- **通过** → 通知 SchedulerAgent 更新 safe_version，OptimizerAgent 继续
- **失败** → 通知 OptimizerAgent 回退代码，记录失败原因
  - OptimizerAgent 最多 2 次修复尝试
  - 仍失败 → 回退到 safe_version，继续下轮优化

---

## 3. 最终验证 (Step 5)

### 3.1 最终正确性验证

运行**全部**正确性测试，确认全部通过：

```bash
# PATH A
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest tests/ {TEST_CMD_FLAG} {op_name} -v 2>&1

# PATH B (FlagGems 框架下)
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest -s experimental_tests/unit/{op_name}_test_enflame_CC.py -v 2>&1
```

### 3.2 最终 Benchmark

运行 `final_bench_rounds` 轮 benchmark 确认性能（默认 1 轮，用户可通过 `bench_rounds=N` 指定，多轮时取平均）：

### 3.3 Golden Reference 对比验证

#### 双仓库模式（`FLAGGEMS_GOLDEN != FLAGGEMS_CHECKIN`）

在最终验证时同时运行 golden 的 benchmark 获取对照数据：

```bash
# Golden Reference 对照 benchmark
cd ${FLAGGEMS_GOLDEN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest benchmark/ {BENCH_CMD_FLAG} {op_name} -v -s 2>&1
```

#### 单仓库模式（`FLAGGEMS_GOLDEN == FLAGGEMS_CHECKIN`）

无需额外运行 golden benchmark，直接使用 Step 2 中记录的 `initial_speedup` 作为 golden 基线。若需要验证原始代码性能，可从 backup 还原后运行：

```bash
# 从 backup 临时还原原始代码验证（可选）
cp "$WORK_DIR/{op_name}_original.py" ${FLAGGEMS_CHECKIN}/src/flag_gems/.../{op_name}.py
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest benchmark/ ... 2>&1
# 还原优化后代码
cp optimized_code.py ${FLAGGEMS_CHECKIN}/src/flag_gems/.../{op_name}.py
```

#### 对比输出格式

```
优化效果对比:
  Golden Ref:    {golden_speedup}x  ← 未修改代码的性能
  CHECKIN 初始:  {initial_speedup}x ← 优化前工作代码性能
  CHECKIN 最终:  {final_speedup}x   ← 优化后性能
  vs Golden:     +{(final-golden)/golden*100}%
  vs 初始:       +{(final-initial)/initial*100}%
```

### 3.4 最终数据记录

记录最终验证结果：
- `final_speedup` = `final_bench_rounds` 轮 benchmark 结果（1轮时直接取值，多轮时取平均）
- `golden_speedup` = golden reference 的 speedup（若可用）
- `final_benchmark_details` = 每个 shape×dtype 的最终数据
- 判断是否达到 target_speedup
- 计算 vs golden 和 vs 初始的提升百分比

### 3.5 失败处理

最终验证失败时：
- 恢复 backup 代码
- 上报 SchedulerAgent: `FAILED_FINAL_VERIFY`

---

## 4. 性能分析要求

每次 benchmark 后，ValidatorAgent 必须执行以下分析：

### 4.1 Per-Shape Per-Dtype Speedup 表

列出每个 shape × dtype 的 Gems Speedup。

### 4.2 平均 Speedup 计算

```
avg_fp16 = mean(speedup_fp16 for all shapes)
avg_fp32 = mean(speedup_fp32 for all shapes)
avg_bf16 = mean(speedup_bf16 for all shapes)
overall_avg = mean(avg_fp16, avg_fp32, avg_bf16)
```

### 4.3 带宽计算 (GB/s)

```
data_bytes = input_elements × bpe + output_elements × output_bpe
bandwidth_GBs = data_bytes / (latency_ms × 1e-3) / (1024^3)
```

其中:
- `bpe` = bytes per element: fp16=2, fp32=4, bf16=2
- `output_bpe` = 输出元素大小
- 报告 Native BW 和 Gems BW
- 小 tensor (< 64KB) 标注带宽无意义

### 4.4 带宽效率分析

- Gems BW > Native BW → 高效 kernel
- Gems BW << Native BW → 潜在 DTE 或算法瓶颈
- GCU400 峰值 HBM 带宽: ~600-800 GB/s

---

## 5. PATH B 测试命令差异

### 5.1 PATH B: 外部目录算子

**正确性测试**（在工作目录运行）:
```bash
cd $WORK_DIR && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest test_*.py -v 2>&1
```

**性能测试**（在工作目录运行）:
```bash
cd $WORK_DIR && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest benchmark_*.py -v -s 2>&1
```

**集成后验证**（Phase 5B，在 FlagGems 框架下验证）:
```bash
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest -s experimental_tests/unit/{op_name}_test_enflame_CC.py -v 2>&1
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest -s experimental_tests/performance/{op_name}_benchmark_enflame_CC.py -v 2>&1
```

---

## 6. 输出规范

向 SchedulerAgent 提交：

```
验证报告:
  type: initial / iteration / final
  correctness:
    passed: int
    total: int
    status: pass / fail
    failed_cases: list (若有)
  benchmark:
    overall_avg_speedup: float
    per_dtype_avg: {fp16: float, fp32: float, bf16: float}
    details: [{shape, dtype, torch_ms, triton_ms, speedup, native_bw, gems_bw}]
  comparison_to_baseline:
    improvement_pct: float
    best_speedup: float
    worst_speedup: float

异常信息:
  error_code: str | None
  error_message: str | None
```
