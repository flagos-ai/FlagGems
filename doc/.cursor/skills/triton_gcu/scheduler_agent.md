# SchedulerAgent Skill — 核心管控专属

## 核心职责

统筹全局优化流程，协调所有 Agent 协作，处理任务路由、异常分流、进度管控，确保流程严格遵循优化顺序与约束规则，是多 Agent 协作的核心枢纽。

## 全局约束

- 无 MCP 依赖，所有操作仅通过 Agent 直接执行
- 严格遵循"不生成新算子、必记录基线、按流程执行"的核心原则
- aten passthrough 已全面禁止（Strategy 39），不得调用 `torch.ops.aten.{op}.out` 绕过 FlagGems

---

## 1. 任务接收与解析

### 1.1 参数解析

从用户输入中解析以下参数：
- **first_arg** (必需): 算子名 / 外部目录路径 / 实验算子文件名
- **target_speedup** (可选, 默认 1.2): 目标加速比
- **max_iterations** (可选, 默认 10, **最少 10**): 最大迭代次数
- **final_bench_rounds** (可选, 默认 1): 最终验证 benchmark 轮数，用户可通过 `bench_rounds=N` 指定（N≥1），多轮时取平均值

### 1.2 参数合法性校验

1. 算子名不为空
2. `max_iterations >= 10`（若用户指定 < 10，强制设为 10）
3. `target_speedup > 0`
4. `final_bench_rounds >= 1`（若用户指定 < 1，强制设为 1）

---

## 2. 路径路由（Routing Rules）

根据输入类型，严格按以下规则确定优化路径：

```
IF first_arg 是一个存在的目录路径:
    → PATH B (外部目录算子优化与集成)

ELSE:
    op_name = first_arg
    IF 文件 ${FLAGGEMS_CHECKIN}/src/flag_gems/ops/{op_name}.py 存在
       OR GCU400 专属实现 gcu400/ops/{op_name}.py 存在:
        → PATH A (内置算子原地优化)
    ELSE:
        → Error: NOT_FOUND
```

**路径说明**:
- **PATH A**: 内置算子原地优化 — 优化 `gcu400/ops/` 下的 enflame 后端特化代码（若无则先由 RegistrarAgent 迁移创建）
- **PATH B**: 外部目录算子优化 — 优化外部算子并集成到 FlagGems `experimental_ops/`

> **核心修改原则**: 尽量只修改 enflame 后端的特化算子代码，公共代码（`src/flag_gems/ops/`）尽量不原地修改，差异化算子新增可以。

---

## 3. Agent 调度流程

### 3.1 完整流程编排

按以下严格顺序调度各 Agent，每个步骤完成后方可推进下一步：

```
Step 1: 环境检测 & 代码定位
  ├─ 调度 EnvDetectAgent
  │   输入: 算子名, 路径类型(A/B/C)
  │   输出: 环境报告, 代码路径, 逻辑解析报告
  │   完成条件: 环境可用, 代码路径确认, 测试文件存在
  │
  ├─ (若算子无 GCU400 专属实现 && PATH A)
  │   调度 RegistrarAgent
  │   输入: 通用代码路径, 算子名
  │   输出: GCU400 专属实现代码, 注册验证结果
  │   完成条件: 注册生效
  │
  └─ 调度 PlannerAgent: 记录环境信息

Step 2: 代码深度学习与分析 ⭐
  ├─ 调度 OptimizerAgent (代码学习模式)
  │   输入: 算子源码路径, EnvDetectAgent 的逻辑解析报告, Golden Reference 对应代码
  │   要求: OptimizerAgent 必须对算子代码进行深度学习，形成完整的代码认知报告
  │   必须覆盖:
  │     a. 代码框架: 整体架构、模块组织、文件间依赖关系
  │     b. 执行流程: 从 Python dispatch 入口到 Triton kernel 的完整调用链
  │     c. 核心算法: 数学公式、计算步骤、数值稳定性处理
  │     d. Kernel 结构: grid/block 配置、内存访问模式、同步机制
  │     e. 优点分析: 当前实现中的良好设计与高效模式
  │     f. 缺点分析: 性能瓶颈、低效模式、冗余计算
  │     g. 潜在风险: 精度问题、边界条件、GCU 不兼容特性
  │     h. 优化点: 基于 GCU400 策略库的优化方向排序（按预期收益从高到低）
  │   输出: 代码认知报告
  │   完成条件: 报告涵盖以上所有 8 个方面
  │
  ├─ (若 FLAGGEMS_GOLDEN 可用)
  │   对比 CHECKIN 与 Golden 的代码差异，理解已有修改的意图
  │
  └─ 调度 PlannerAgent: 记录代码分析结果

Step 3: 正确性验证 & 初始基线
  ├─ 调度 ValidatorAgent (初始验证)
  │   输入: 代码路径, 测试文件路径
  │   输出: 正确性结果, 初始 speedup 基线
  │   完成条件: 正确性通过, 基线数据记录
  │
  ├─ 判断: 若 initial_speedup >= 1.2x → 跳过 Step 3, 直接 Step 5
  │
  └─ 调度 PlannerAgent: 记录基线数据

Step 4: 性能优化循环 (≥10 轮)
  ├─ 调度 OptimizerAgent
  │   输入: 优化指令, 代码路径, 基线数据, Step 1.5 的代码认知报告
  │   每轮流程:
  │     a. OptimizerAgent 修改代码
  │     b. 调度 ValidatorAgent (迭代验证)
  │        - 正确性通过 → 记录结果, 继续
  │        - 正确性失败 → 通知 OptimizerAgent 回退
  │     c. OptimizerAgent 运行 benchmark, 记录加速比
  │   退出条件:
  │     - avg speedup >= target_speedup → 进入 Step 5
  │     - 达到 max_iterations 上限 → 恢复 best_version, 进入 Step 5
  │     - 时间超过 1 小时 → 终止并记录未来方向
  │
  └─ 调度 PlannerAgent: 每轮记录优化数据

Step 5: 最终验证
  ├─ 调度 ValidatorAgent (最终验证)
  │   输入: 最终代码, 测试文件
  │   输出: 最终正确性, `final_bench_rounds` 轮 benchmark 结果（默认 1 轮，多轮取平均）
  │   失败处理: 恢复 backup, 报告 FAILED_FINAL_VERIFY
  │
  └─ 调度 PlannerAgent: 记录最终结果

Step 6: 报告生成 & Skill 更新 【🚨最高优先级 — 绝对不可跳过🚨】
  ├─ 🚨 HARD STOP: 未完成 Step 6 等同于整个优化任务未完成
  │   即使已在对话中展示性能数据，仍必须执行全部子步骤
  │   历史教训: abs/abs_ 优化时跳过 Step 6 导致无文档交付
  │
  ├─ 调度 ReporterAgent
  │   输入: 全部优化记录, 最终验证报告, 优化后代码
  │   输出: Word 报告, Wiki 页面, Skill 更新
  │   完成条件: Word已保存 + Wiki URL已获取 + 算子履历已更新 + 最终总结含Wiki URL
  │
  └─ 调度 PlannerAgent: 记录报告生成结果, 更新算子优化履历, 输出完整日志

Step 7: 关键产出物格式检查 【Step 6 完成后必须执行】
  ├─ SchedulerAgent 自行检查以下项目:
  │   1. Word 文件名是否符合 {算子名}_GCU400_优化报告_{YYYYMMDD}.docx 规范？
  │   2. Wiki 页面标题是否符合 {算子名}_GCU400_优化报告_{YYYYMMDD} 规范？
  │   3. 报告是否包含全部 6 个必备章节（概述/性能对比/带宽分析/编译时间/每轮详情/关键发现/代码摘要）？
  │   4. 性能对比表是否使用 "基线→最终" 格式且含均值行？
  │   5. 带宽分析表是否覆盖每种 dtype？
  │   6. 每轮优化详情是否包含所有轮次？
  │   7. Word 和 Wiki 章节结构和数据是否一致？
  |   8. optimizer_agent.md 是否更新算子优化经验？
  |   9. planner_agent.md 是否更新算子优化结果？
  |   10. 最终对话总结中是否列出所有被优化算子的 Wiki page URL？
  │
  ├─ 任何检查项不通过 → 退回 Step 6 让 ReporterAgent 重新生成
  │
  └─ 全部通过 → 算子优化任务正式完成
```

### 3.2 版本管理

SchedulerAgent 维护全局版本状态：

```
safe_version_code = None         # 最近通过正确性验证的代码
safe_version_speedup = None
best_version_code = None         # 历史最优性能的代码
best_version_speedup = 0
initial_speedup = None           # 优化前基线
optimization_log = []            # 每轮优化记录
```

---

## 4. 异常处理

### 4.1 设备异常处理（唯一权威定义，其他 Agent 必须引用此章节）

> **本章节是设备异常判断与处理的唯一权威说明。** 所有其他 Agent（EnvDetectAgent、OptimizerAgent、ValidatorAgent 等）在遇到设备相关异常时，**必须按本章节的规则判断和处理**，不得自行定义判断标准或重复描述掉卡处理流程。

#### 4.1.1 白名单信号（非掉卡，无需触发掉卡流程）

以下信号**不代表设备掉卡**，出现后应**先重试操作**或**缩小测试范围**：

| 信号 | 含义 | 处理方式 |
|------|------|---------|
| `Aborted (core dumped)` / `exit code 134` | 程序逻辑错误、内存越界、断言失败 | 等待 10s 后重试，或缩小范围 |
| `SIP exception` | Kernel 执行错误，设备本身仍可用 | 重试或跳过该 case |
| `Segmentation fault` / `exit code 139` | 程序段错误，非设备问题 | 重试或跳过 |
| `kernel module not installed` / `ufiOpenKFD() call failed` | 前一次崩溃后驱动临时异常，通常等待后可自动恢复 | 等待 10s 后重试 |
| `topsErrorInvalidDevice` / `Error 101` | 前一次 GCU 操作异常导致的级联错误 | 等待 10s 后重试 |
| `out of resource: grid.x` / `Hardware limit` | Grid 配置超出硬件限制 | 修改 kernel 代码（降低 BLOCK_SIZE 或 grid 维度）|

**白名单信号处理流程**：
1. 记录异常信号和上下文
2. 等待 10 秒
3. 重新执行失败的操作（最多重试 2 次）
4. 仍失败 → 跳过当前 case/算子，继续下一个；**不中断用户**

#### 4.1.2 掉卡信号（需二次确认）

以下信号**可能**表示设备真正掉卡：
- `gcu device not available`
- `torch.gcu.device_count() == 0`（之前检测到设备但现在消失）
- `RuntimeError: No GCU device` 或类似设备丢失信息
- 连续 3 次以上白名单重试全部失败，且二次确认设备不可用

#### 4.1.3 掉卡处理流程（两步确认）

1. **一次检测到疑似掉卡信号** → **等待 10 秒**
2. **二次确认设备可用性**:
   ```python
   python -c "import torch; print('has_gcu:', hasattr(torch, 'gcu')); print('count:', torch.gcu.device_count()); x = torch.zeros(1, device='gcu'); print('OK')"
   ```
3. **二次确认设备仍不可用** → 通过 AskQuestion 提示用户在 Docker 外执行 hot reset：
   ```
   ./TopsPlatform*_deb_amd64.run --driver -y
   ```
4. 等待用户确认完成后，恢复优化流程
5. **二次确认设备可用** → 不是掉卡，继续正常流程（可能是 kernel 错误、内存溢出等，按对应异常处理）
6. **不得在未经二次确认的情况下判定掉卡并中断流程**

### 4.2 编译超时

当 OptimizerAgent 上报编译时间异常：
- 单算子总编译时间 > 1 分钟
- 单个 shape 编译 > 5 分钟

→ 通知 OptimizerAgent **优先执行编译时间优化策略**（do_not_specialize、减少 autotune、降低 BLOCK_SIZE 等），再进行性能优化

### 4.3 正确性失败

ValidatorAgent 上报正确性失败时：
1. 通知 OptimizerAgent 回退到 `safe_version_code`
2. 记录失败原因到 `optimization_log`
3. 若连续 3 次正确性失败，暂停优化，上报 FAILED_CORRECTNESS

### 4.4 其他异常

| 场景 | 处理 |
|------|------|
| Python 环境缺失依赖 | Error: `ERROR: Python environment missing required dependencies` |
| 算子未找到 | Error: `NOT_FOUND` |
| 原始代码正确性失败 | Error: `ORIGINAL_FAILED` |
| 无测试用例 | Error: `NO_TEST` |
| PATH B 源目录缺少文件 | Error + 列出缺失文件 |
| FP64/INT64 检测到 | Warning: `UNSUPPORTED_DTYPE` |

---

## 5. 退出码

| 退出码 | 含义 |
|--------|------|
| SUCCESS | 达到目标加速比且验证通过 |
| PARTIAL | 加速比有提升但未达标，验证通过 |
| FAILED_CORRECTNESS | 优化过程中正确性无法通过 |
| FAILED_INTEGRATION | PATH B: 优化成功但集成验证失败 |
| FAILED_FINAL_VERIFY | PATH A: 最终验证失败，已恢复原始代码 |
| NOT_FOUND | 算子未找到 |
| ORIGINAL_FAILED | 原始代码正确性测试失败 |
| NO_TEST | 无对应测试用例 |

---

## 6. 进度同步协议

### 6.1 与 PlannerAgent 同步

每完成一个关键节点，向 PlannerAgent 发送进度更新：
- 格式: `{step, agent, status, timestamp, data}`
- PlannerAgent 返回: 合规/违规状态

### 6.2 流程约束检查点

SchedulerAgent 在以下节点自检：
1. ✅ Step 2 完成前，基线数据是否已记录？
2. ✅ Step 3 开始前，初始正确性是否通过？
3. ✅ Step 3 每轮优化后，是否包含正确性验证？
4. ✅ Step 5 开始前，是否恢复了 best_version_code？
5. ✅ Step 6 开始前，最终验证是否通过？

### 6.3 PATH B 差异化流程

上述 Section 3.1 的 Step 1-6 主要描述 PATH A（内置算子）。PATH B 的关键差异如下：

#### PATH B: 外部目录算子优化 & 集成

**Phase 0B: 初始化**
```
1. 解析 $ARGUMENTS → operator_path, target_speedup, max_iterations
2. 提取 op_name, gpu_name, func_name（从目录/文件名）
3. 创建工作副本:
   WORK_DIR="<operator_path>_flaggems_optimize_$(date +%Y%m%d_%H%M%S)"
   mkdir -p "$WORK_DIR/iterations"
   cp <operator_path>/*.py "$WORK_DIR/"
4. 初始化版本管理（同 PATH A）
```

**Phase 1B: 预检查**
```
验证外部目录中必需文件:
- *_triton.py（Triton 算子实现）
- *_torch.py（PyTorch 参考实现）
- test_*.py（正确性测试）
- benchmark_*.py（性能测试）
缺少任何文件 → Error + 列出缺失文件
```

**Step 1-3 差异**: 同 PATH A，但操作目标为 `$WORK_DIR` 下的 `*_triton.py` 文件

**Phase 4B: FlagGems 集成（PATH B 独有）⭐**
```
Step 4B.1: 放置 Triton 算子
  → ${FLAGGEMS_CHECKIN}/src/flag_gems/experimental_ops/{op_name}_triton_enflame_CC.py

Step 4B.2: 注册到 experimental_ops/__init__.py（检查避免重复）

Step 4B.3: 转换并放置单元测试
  → ${FLAGGEMS_CHECKIN}/experimental_tests/unit/{op_name}_test_enflame_CC.py

Step 4B.4: 转换并放置性能测试
  → ${FLAGGEMS_CHECKIN}/experimental_tests/performance/{op_name}_benchmark_enflame_CC.py
```

**Phase 5B: 在 FlagGems 框架下验证**
```bash
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest -s experimental_tests/unit/{op_name}_test_enflame_CC.py -v
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest -s experimental_tests/performance/{op_name}_benchmark_enflame_CC.py -v
```
- 通过 → Step 6
- 失败 → Error `FAILED_INTEGRATION`

**Phase 6B 报告**: 报告须包含 FlagGems 集成后的文件路径:
```
FlagGems Files:
  Op:        src/flag_gems/experimental_ops/{op_name}_triton_enflame_CC.py
  Test:      experimental_tests/unit/{op_name}_test_enflame_CC.py
  Benchmark: experimental_tests/performance/{op_name}_benchmark_enflame_CC.py
```

### 6.4 多算子批量执行

当输入多个算子时，SchedulerAgent **串行**执行：
```
for op in [算子A, 算子B, 算子C]:
    execute_full_pipeline(op, target_speedup, max_iterations)
    # 环境检测结果可缓存复用
    # ⚠️ 必须完成 Step 1-6 全部步骤后才可进入下一个算子

    # 默认行为：完成后 AskQuestion 询问用户下一步
    if not silent_mode:
        AskQuestion("算子 {op} 优化已完成 (Step 1-6)，请选择下一步操作",
                     options=["继续优化下一个算子", "调整参数后继续", "终止优化"])
    # 静默模式：自动继续下一个算子
```

**关键约束**:
1. 每个算子**必须完成 Step 1→1.5→2→3→5→6 全部步骤**后才可推进下一个算子
2. 不得在优化循环结束后跳过 Step 5（最终验证）和 Step 6（报告生成+Skill 更新）
3. **默认行为（非静默模式）**: 完成一个算子的全部 Step 1-6 后，**必须通过 AskQuestion 询问用户下一步操作**。选项包括：继续优化下一个算子、调整参数、终止。用户确认后才可继续。**这是默认行为。**
4. **静默模式**: 仅当用户**明确指定**"静默模式"或表示不需要中间确认时才启用。完成一个算子全部流程后自动进入下一个算子，仅异常时中断询问

---

## 7. Optimization Defaults（调度参数）

1. **最少迭代次数: 10 轮**。10 轮后若 Gems Speedup < 0.8x，必须继续
2. **优化时间上限: 1 小时**；迭代次数上限: 20 次
3. **优化目标**: Gems Speedup ≥ 1.2x（至少逼近 1.0x）
4. **性能阈值自动流程**: 当 Gems Speedup ≥ 1.2x 或达到上限，触发 Step 5+6
5. **编译时间监控**: benchmark 运行时始终设置 `export TRITON_GCU_COMPILE_TIME=1`
