# FlagGems GCU Operator Optimization — Multi-Agent Orchestration

批量优化 FlagGems GCU 算子的多 Agent 协同入口。支持同时优化多个算子 `{算子A, 算子B, 算子C}`，为每个算子分别调度完整的优化流水线。

## 使用方式

```
/kernel-opt-gcu {算子A} {算子B} {算子C} [target_speedup] [max_iterations] [bench_rounds=N]
```

**示例**:
- `/kernel-opt-gcu relu sigmoid tanh` — 批量优化 3 个内置算子（PATH A），默认目标 1.2x / 10 轮 / 最终验证 1 轮 benchmark
- `/kernel-opt-gcu softmax 1.5 15` — 单算子优化（PATH A），目标 1.5x，最多 15 轮
- `/kernel-opt-gcu softmax 1.5 15 bench_rounds=3` — 同上，但最终验证跑 3 轮 benchmark 取平均
- `/kernel-opt-gcu /path/to/index_put_cc_enflame` — 外部目录算子（PATH B），优化后集成到 FlagGems

## 两条路径说明

| 路径 | 触发条件 | 操作目标 | 集成步骤 |
|------|---------|---------|---------|
| **PATH A** | 输入为已知内置算子名（如 `relu`） | `gcu400/ops/{op_name}.py`（enflame 后端特化） | 无（原地优化） |
| **PATH B** | 输入为已存在的目录路径 | `$WORK_DIR/*_triton.py`（工作副本） | Phase 4B: 集成到 `experimental_ops/` |

> **核心修改原则**: 尽量只修改 enflame 后端的特化算子代码（`gcu400/ops/`），对于公共代码尽量不能原地修改（差异化算子新增可以，但原地修改不可以）。

## 核心原则（全局约束，所有 Agent 必须遵守）

1. **不生成新算子** — 仅优化现有算子，不修改核心功能与函数签名
2. **必记录基线** — 优化前必须运行 benchmark 记录初始 speedup 基线数据
3. **按流程执行** — 严格遵循 Step 1→2→3→4→5→6→7 的优化顺序，任何步骤不可跳过
4. **无 MCP 依赖** — 所有操作由 Agent 直接执行
5. **数据精度 > 性能** — 任何导致正确性失败的优化必须回退
6. **aten passthrough 已禁止** — 不得调用 `torch.ops.aten.{op}.out` 绕过 FlagGems（Strategy 39）
7. **双仓库原则** — 代码修改仅在 `FLAGGEMS_CHECKIN`（工作仓库）中进行，`FLAGGEMS_GOLDEN`（golden reference）作为不可变基准对照。若只有一套 FlagGems 代码，则同一仓库既当 golden 又做优化改写，初始 baseline 和 backup 文件充当 golden 角色
8. **禁止以对话上下文长度为由打断优化** — 不得以"当前对话上下文较长"、"上下文接近限制"、"建议新开对话继续"等理由中断、暂停或终止正在进行的优化流程。静默模式下更**严格禁止**通过 AskQuestion 询问用户是否新开对话或以任何形式建议用户新开对话。优化必须在当前对话中持续执行直到所有算子完成或遇到真正的技术异常（掉卡、正确性失败等）

## 多 Agent 架构

| Agent | Sub Skill 文件 | 职责 |
|-------|---------------|------|
| **SchedulerAgent** | `scheduler_agent.md` | 核心管控：任务解析、路径路由、Agent 调度、异常处理 |
| **EnvDetectAgent** | `env_detect_agent.md` | 环境检测：GCU 设备、Python/Triton 环境、代码定位与逻辑解析 |
| **OptimizerAgent** | `optimizer_agent.md` | 性能优化：至少 10 轮迭代优化循环、GCU 优化策略库 |
| **ValidatorAgent** | `validator_agent.md` | 验证：初始基线记录、迭代正确性验证、最终验证 |
| **ReporterAgent** | `reporter_agent.md` | 报告：Word/Wiki 报告生成、自检清单、Skill 更新 |
| **RegistrarAgent** | `registrar_agent.md` | 注册：公共代码迁移到 GCU400 专属实现、算子注册 |
| **PlannerAgent** | `planner_agent.md` | 监督：进度监控、迭代管控、流程合规检查、日志记录 |

## 协同流程

```
用户输入: {算子A, 算子B, 算子C}
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 根 Skill: 解析算子列表，为每个算子创建独立优化任务    │
└─────────────────────────────────────────────────────┘
    │
    ▼ (对每个算子，启动以下流水线)
┌─────────────────────────────────────────────────────┐
│ SchedulerAgent: 接收任务，解析路径(A/B/C)，协调全流程 │
│   ├─ Step 1 → EnvDetectAgent: 环境检测 & 代码定位     │
│   │            ├─ (若无 GCU400 专属实现)               │
│   │            └─ → RegistrarAgent: 迁移 & 注册        │
│   ├─ Step 2 → OptimizerAgent: 代码深度学习与分析 ⭐    │
│   │            (框架/流程/算法/优缺点/风险/优化点)      │
│   ├─ Step 3 → ValidatorAgent: 初始验证 & 基线记录     │
│   ├─ Step 4 → OptimizerAgent: 性能优化循环(≥10轮)     │
│   │            └─ 每轮 → ValidatorAgent: 迭代验证      │
│   │            (PATH B: 优化后 → RegistrarAgent 集成   │
│   │             → ValidatorAgent 集成验证)              │
│   ├─ Step 5 → ValidatorAgent: 最终验证                │
│   ├─ Step 6 → ReporterAgent: 报告生成 & Skill 更新    │
│   │            🚨 HARD STOP: 未完成=任务未完成 🚨       │
│   │            必须生成Word+Wiki+更新算子履历+列出URL    │
│   ├─ Step 7 → SchedulerAgent: 关键产出物格式检查      │
│   │            10项检查(标题/章节/数据/格式合规性)       │
│   │            不通过→退回 Step 6 重新生成               │
│   └─ 全程   → PlannerAgent: 进度监督 & 日志记录       │
└─────────────────────────────────────────────────────┘
```

## 执行指南

### 1. 参数解析

从用户输入中解析：
- **算子列表**: 所有非数字参数，每个作为独立的算子名或路径
- **target_speedup**: 第一个浮点数参数（默认 1.2）
- **max_iterations**: 第一个整数参数（默认 10，最少 10）
- **final_bench_rounds**: 最终验证 benchmark 轮数（默认 1，用户可通过 `bench_rounds=N` 指定，N≥1）。多轮时取平均值作为最终 speedup

### 2. 并行 vs 串行

- **单算子**: 直接调度 SchedulerAgent 执行完整流水线
- **多算子**: 逐个串行执行（共享同一 GCU 设备，避免资源冲突）；每个算子的 EnvDetectAgent 结果可缓存复用
- **串行执行要求**: 每个算子**必须完成完整的 Step 1→2→3→4→5→6→7 全部步骤**（包括最终验证、报告生成和产出物格式检查）后，才可进入下一个算子的优化。不得在任何中间步骤跳过或暂停转向下一个算子
- **默认行为（非静默模式）**: 完成一个算子的全部 Step 1-7 后，**必须通过 AskQuestion 询问用户下一步操作**（如：继续优化下一个算子、调整参数、或终止）。用户确认后才可进入下一个算子的优化。**这是默认行为，不需要用户明确请求。**
- **静默模式**: 仅当用户**明确指定**"静默模式"、"不需要每个算子间的确认"、或类似表述时才启用。在静默模式下，Agent 在完成一个算子的全部 Step 1-7 后，**自动进入下一个算子**的优化，不通过 AskQuestion 询问用户。仅在遇到异常（掉卡、正确性失败等）时才中断询问用户
- **🚨 禁止以上下文为由中断优化**: Agent **绝对不得**以"前对话上下文较长"、"建议新开对话继续"、"上下文即将到达限制"等任何理由中断或暂停优化流程。静默模式下更**绝对不得**通过 AskQuestion 建议用户新开对话。Agent 必须在当前对话中持续执行直到所有算子优化完成或遇到真正的技术异常（掉卡等）。上下文管理是 Agent 自身的责任，不应转嫁给用户

#### 2.1 多算子 Done List（必须使用）

当优化任务包含多个算子时，**必须**在解析完算子列表后立即生成 `N/N Done List`，并在每个算子完成 Step 7 后更新进度。

**Done List 格式**:
```
========== 优化进度: 2/10 Done ==========
[✅] 1/10 重新读取skill, 再优化 pad (Step 1-7)
[✅] 2/10 重新读取skill, 再优化 replication_pad3d (Step 1-7)
[⏳] 3/10 重新读取skill, 再优化 arange (Step 1-7)          ← 当前
[ ] 4/10 重新读取skill, 再优化 linspace (Step 1-7)
[ ] 5/10 重新读取skill, 再优化 one_hot (Step 1-7)
[ ] 6/10 重新读取skill, 再优化 elu_backward (Step 1-7)
[ ] 7/10 重新读取skill, 再优化 gelu_backward (Step 1-7)
[ ] 8/10 重新读取skill, 再优化 apply_rotary_pos_emb (Step 1-7)
[ ] 9/10 重新读取skill, 再优化 rwkv_ka_fusion (Step 1-7)
[ ] 10/10 重新读取skill, 再优化 moe_sum (Step 1-7)
==========================================
```

**Done List 规则**:
1. **生成时机**: 参数解析完成后立即生成，作为全局进度看板
2. **更新时机**: 每个算子完成 Step 7（或异常中止）后更新状态标记
3. **状态标记**: `[✅]` 已完成 / `[⏳]` 进行中 / `[ ]` 待处理 / `[❌]` 异常中止
4. **展示时机**: 每个算子开始前和完成后各打印一次 Done List

#### 2.2 重新读取 Skill（强制要求）

**关键原则**: 每个算子优化前**必须重新读取所有相关 sub skill 文件**，不得依赖上一个算子优化时的缓存理解。

**原因**: Skill 文件会在优化过程中被更新（OptimizerAgent 的 Section 9 自更新机制、其他设备的优化经验沉淀等），前一个算子优化结束时的 skill 内容可能已与当前不同。

**执行方式**:
```
每个算子的 Step 1 开始前，必须重新执行:
1. 读取 SKILL.md（根 skill）→ 确认全局约束和协同流程无变化
2. 读取 env_detect_agent.md → 获取最新环境检测规则
3. 读取 optimizer_agent.md → 获取最新优化策略库（可能新增了前一个算子沉淀的经验）
4. 读取 validator_agent.md → 获取最新验证规则
5. 其他 sub skill 按需读取
```

> **禁止**: 以"上个算子已读取过 skill"为由跳过重新读取。即使相邻两个算子间隔仅几秒，也必须重新读取。

### 3. 子 Skill 调用方式

每个 Agent 在执行其职责时，**必须先读取对应的 sub skill 文件**获取详细规则和操作步骤。根 skill 仅提供协调框架，具体执行规则在各 sub skill 中定义。

```
Agent 执行流程 (对应 7 Step):
Step 1. SchedulerAgent 调度 EnvDetectAgent → 读取 env_detect_agent.md → 执行环境检测
        (若需迁移) 调度 RegistrarAgent → 读取 registrar_agent.md → 执行迁移注册
Step 2. SchedulerAgent 调度 OptimizerAgent → 读取 optimizer_agent.md → 代码深度学习与分析
Step 3. SchedulerAgent 调度 ValidatorAgent → 读取 validator_agent.md → 执行初始验证 & 基线记录
Step 4. SchedulerAgent 调度 OptimizerAgent → 读取 optimizer_agent.md → 执行优化循环(≥10轮)
        └─ 每轮调度 ValidatorAgent → 迭代验证
Step 5. SchedulerAgent 调度 ValidatorAgent → 最终验证
Step 6. SchedulerAgent 调度 ReporterAgent → 读取 reporter_agent.md → 生成报告 & Skill 更新
Step 7. SchedulerAgent 自行执行 → 关键产出物格式检查(10项)，不通过退回 Step 6
全程:   PlannerAgent 监督 → 读取 planner_agent.md → 进度监控 & 日志记录
```

### 4. 数据传递

Agent 间通过以下结构化数据传递信息：

```
EnvDetectAgent → SchedulerAgent:
  - PYTHON_CMD, FLAGGEMS_CHECKIN, FLAGGEMS_GOLDEN, GCU_ARCH_VERSION
  - 算子源代码路径, 测试文件路径, benchmark 文件路径
  - 核心逻辑解析报告

ValidatorAgent → SchedulerAgent:
  - 正确性验证结果 (pass/fail, 通过数/总数)
  - benchmark 数据 (所有 shape × dtype 的 speedup)
  - initial_speedup 基线数据

OptimizerAgent → SchedulerAgent:
  - 每轮优化记录 (策略名/描述/效果)
  - 当前加速比, 优化后的代码
  - 异常信息

ReporterAgent → SchedulerAgent:
  - Word 报告路径, Wiki 页面 URL
  - Skill 更新内容
  - 自检清单结果

PlannerAgent → SchedulerAgent:
  - 进度报告, 流程违规提示
  - 完整优化日志
```
