# ReporterAgent Skill — 报告专属

## 核心职责

执行优化流程 Step 6，生成优化报告（Word + Wiki）、执行自检清单、更新 skill 文件与 To Be Optimized Ops 列表。

> **🚨🚨🚨 HARD STOP — Step 6 是每个算子优化的最终交付物 🚨🚨🚨**
>
> **未完成 Step 6 等同于整个优化任务未完成。即使已经在对话中展示了性能数据和总结，仍然必须执行 Step 6 的全部子步骤。**
>
> **历史教训：abs/abs_ 优化时，Agent 在展示性能汇总后就认为任务结束，完全跳过了 Step 6（Word 报告、Wiki 页面、skill 更新），导致无文档交付。此错误绝不可再犯。**
>
> **强制检查 — 在结束每个算子的优化对话前，必须自问：**
> 1. Word 报告 (.docx) 是否已保存到工程根目录？
> 2. Wiki 页面是否已创建并获得 URL？
> 3. 已优化算子列表是否已更新（含 Wiki URL）？
> 4. 最终对话总结中是否列出了 Wiki page URL？
>
> **以上 4 项全部为"是"才可结束。任何一项为"否"则必须继续执行。**

## 全局约束

- 报告与 Wiki 页面内容必须**真实、完整**，严格匹配优化过程与验证结果
- 不篡改数据，不遗漏轮次
- Skill 更新需精准对应各 Agent 的专属 skill

---

## 1. 报告生成前准备

### 1.1 必读内容

生成报告前，**必须先重新阅读**：
1. 本 skill 的 "Optimization Report Template" 章节
2. "报告生成后自检清单"（10 项）

确认理解所有必填章节和格式要求后再开始。

---

## 1.5 报告标题规范

### 【强制执行 - 所有报告必须遵循统一标题格式】

| 产出物 | 标题格式 | 示例 |
|--------|---------|------|
| Word 文件名 | `{算子名}_GCU400_优化报告_{YYYYMMDD}.docx` | `all_GCU400_优化报告_20260422.docx` |
| Wiki 页面标题 | `{算子名}_GCU400_优化报告_{YYYYMMDD}` | `all_GCU400_优化报告_20260422` |
| Word 文档内标题 | `{算子名} 算子 GCU400 优化报告` | `all 算子 GCU400 优化报告` |

- `{算子名}` 使用原始算子名（如 `all`、`any`、`uniform_`），不加引号
- `{YYYYMMDD}` 使用报告生成当天的日期
- Word 保存路径: 工程根目录（workspace root）

---

## 2. Optimization Report Template

### 【强制执行 - 报告生成规则】

1. 严格按以下模板的 **6 个章节逐一生成**
2. **Word 报告和 Wiki 页面必须包含完全相同的章节结构和内容**
3. 生成后必须执行自检清单（Section 4）
4. 违反以上任何一条等同于任务未完成
5. **格式不合规的报告等同于未生成** — 必须严格遵循模板章节和表格格式

### 报告模板

```
# {op_name} 算子 GCU400 优化报告

## 1. 概述
【6 个必填项 - 缺一不可】:
- 算子名称（全称）
- 源文件路径（完整相对路径）
- 优化轮数（总轮数 + 保留/回退轮次明细）
- 优化目标（明确数值目标，如 Gems Speedup ≥ 1.2x）
- 最终结论（含最终 speedup 数值范围、关键发现、受限因素）
- 测试指令（benchmark 和 correctness 两条完整命令，可直接复制运行）

**【强制检查】**：生成概述后，逐项核对 6 个必填项是否齐全。

## 2. 基线 vs 最终性能对比
使用 "Golden Ref / CHECKIN基线 → 最终" 列格式：
| Shape | fp16 Golden→基线→最终 | fp32 Golden→基线→最终 | bf16 Golden→基线→最终 |
| 均值  | avg_golden→avg_base→avg_final | ... | ... |

**说明**：
- **Golden Ref**: `FLAGGEMS_GOLDEN` 不变量仓库的性能数据（若可用）
- **CHECKIN 基线**: `FLAGGEMS_CHECKIN` 优化前的性能数据
- **最终**: `FLAGGEMS_CHECKIN` 优化后的性能数据

**重要**：表格之外增加一行，所有类型均值加在一起再求平均 speedup（基线 vs. 最终），
并计算百分比变化。格式："总体均值：0.175x → 2.635x（15.0x 提升）"
若有 Golden 数据，额外显示 vs Golden 的变化。

## 3. 带宽分析
| Shape | dtype | 数据量 | Native延迟 | Gems延迟 | Native带宽 | Gems带宽 |

## 3.5 编译时间分析
| 阶段 | 首次编译 | 全量编译 | 说明 |
说明编译时间优化手段及效果。

## 4. 每轮优化详情
| 轮次 | 策略 | 关键变更 | 代表性指标变化 | 结论 |

**注意**：须对关键性能突破点进行详细说明，给出性能提升百分比。

## 5. 关键发现与经验
- 有效策略总结
- 无效/有害策略总结
- 瓶颈分析
- 编译时间优化总结

## 6. 代码最终版本摘要
```

---

## 3. 报告保存

### 3.1 Word 报告

- 保存路径: **工程根目录**（workspace root）
- 文件名: `{算子名}_optimization_report.docx`

### 3.2 Wiki 页面

- 使用 wiki_helper SKILL 在用户 wiki 主页下创建新 page
- **章节和内容必须与 Word 完全一致**
- Wiki 无法访问时，执行自动登录

### 3.3 Wiki 自动登录

当 Wiki session 过期（403/404 authorized:false）时：

```bash
python /root/.cursor/skills/wiki_helper/wiki_helper_cli.py login
```

凭证已存储在 credentials 文件中。登录成功后 session 保存在 `/root/.cursor/wiki_sessions.json`，有效期 24 小时。

获取 JSESSIONID:
```bash
python3 -c "import json; d=json.load(open('/root/.cursor/wiki_sessions.json')); print(d['wiki.enflame.cn']['cookies']['JSESSIONID'])"
```

---

## 4. 报告自检清单

### 【强制逐项核对，不得跳过】

生成报告后，**必须在对话中显式列出以下 10 项的通过/未通过状态**。未执行自检清单等同于报告未生成。

1. [ ] 概述包含全部 6 个必填项（算子名称、源文件路径、优化轮数、优化目标、最终结论、测试指令）？ **关键1**最终结论跟planner_agent.md中的6.3 记录格式一致，即{算子名} ({状态}: {轮次}轮; avg {基线}x→{最终}x ({提升}x提升); {正确性}; {关键技术}; {未达标原因}) **关键2**测试指令必须要包含benchmark和correctness两条完整命令，可直接复制运行
2. [ ] 基线 vs 最终性能表格使用 "基线→最终" 格式，且包含均值行？
3. [ ] 总体均值变化百分比已计算并以加粗文字单独显示？
4. [ ] 带宽分析表格至少覆盖每种 dtype 的代表性 shape，且附带分析说明？
5. [ ] 编译时间分析包含 "基线" 和 "最终" 两行，且附带优化手段说明？
6. [ ] 每轮优化详情包含**所有**轮次（包括回退的轮次），每轮独立一行？
7. [ ] 关键性能突破点标注了具体百分比提升？
8. [ ] 关键发现与经验包含 4 个子节（有效策略/无效策略/瓶颈分析/编译时间）？
9. [ ] 代码最终版本摘要包含内核结构、参数配置、关键逻辑？
10. [ ] **Word 和 Wiki 的章节结构和内容完全一致**？
11. [ ] **Wiki 页面已成功创建，URL 已获取并记录**？（未创建 Wiki 页面等同于 Step 6 未完成）
12. [ ] **已优化算子列表（planner_agent.md）已更新，包含 Wiki URL**？
13. [ ] **最终对话总结中已列出所有被优化算子的 Wiki page URL**？格式如下：
    ```
    ## 被优化算子 Wiki 文档
    - 算子名: Wiki URL
    - 算子名: Wiki URL
    ```
    **没有列出 Wiki URL 的最终总结视为不完整，必须补全。**

任何一项未通过，必须立即修正后重新提交。

### 4.2 格式合规性自查流程

报告生成后，Agent **必须按以下流程逐步验证格式合规性**，不得跳过：

```
格式自查流程:
1. 标题检查:
   - Word 文件名是否遵循 {算子名}_GCU400_优化报告_{YYYYMMDD}.docx？
   - Wiki 页面标题是否遵循 {算子名}_GCU400_优化报告_{YYYYMMDD}？

2. 章节完整性检查:
   - 是否包含全部 6 个必备章节（概述/性能对比/带宽分析/编译时间/每轮详情/关键发现/代码摘要）？
   - 每个章节是否有实质内容（不能为空或仅一句话敷衍）？

3. 表格格式检查:
   - 性能对比表是否使用 "基线→最终" 格式？
   - 是否包含均值行和总体均值百分比变化？
   - 带宽分析表是否覆盖每种 dtype 的代表性 shape？

4. 数据一致性检查:
   - 报告中的数据是否与实际 benchmark 输出一致？
   - Word 和 Wiki 的数据是否完全一致？

5. 最终确认:
   - 逐项勾选 Section 4.1 的自检清单（13 项）
   - 任何一项未通过则修正后重新自查
```

**历史教训（按严重程度排序）**：
1. 🚨 **abs/abs_ 优化时完全跳过 Step 6**：Agent 在展示性能汇总后认为任务完成，未生成 Word 报告、未创建 Wiki 页面、未更新 skill 文件。这是最严重的遗漏 — 优化工作没有任何持久化文档记录。**此错误绝不可再犯。**
2. remainder 算子首次生成的报告因未执行自检清单，缺少带宽分析、编译时间分析、每轮优化详情等关键章节，导致报告被退回重新生成。
3. **all/any 首次生成的报告格式严重不合规**：缺少带宽分析、编译时间分析、每轮优化详情、代码最终版本摘要等关键章节，Wiki 页面内容与 Word 不一致。**生成报告后必须按照自检清单逐项检查和格式自查流程逐步验证，不能偷懒。**

---

## 5. PATH B 报告差异

PATH B 报告**必须额外包含** FlagGems 集成信息：

```
============================================================
FLAGGEMS GCU OPTIMIZATION & INTEGRATION COMPLETE
============================================================
Operator: {op_name} | Platform: Enflame GCU{arch_version}
Status: {SUCCESS | PARTIAL | FAILED}
Initial Speedup (Baseline): {initial_speedup}x  ← pre-optimization
Final Speedup:              {final_speedup}x     ← post-optimization
Improvement:                +{improvement}%

FlagGems Files:
  Op:        src/flag_gems/experimental_ops/{op_name}_triton_enflame_CC.py
  Test:      experimental_tests/unit/{op_name}_test_enflame_CC.py
  Benchmark: experimental_tests/performance/{op_name}_benchmark_enflame_CC.py
============================================================
```

---

## 6. Skill 文件更新

### 5.1 更新内容

ReporterAgent 整理出优化过程中发现的新增经验后，**通知对应 Agent 自行更新各自的 skill 文件**（各 Agent 是自身 skill 的唯一写入者）：

1. **通用优化模式 & 算子专属经验 & 无效/有害策略** → **通知 OptimizerAgent** 更新 `optimizer_agent.md`（详见其 Section 9: Skill 自更新）
2. **新发现的验证规则** → **通知 ValidatorAgent** 更新 `validator_agent.md`
3. **新发现的环境/路径规则** → **通知 EnvDetectAgent** 更新 `env_detect_agent.md`

ReporterAgent 需向对应 Agent 提供明确的更新内容摘要（策略名称、效果数据、来源算子），由目标 Agent 完成实际写入。

### 5.2 更新算子优化履历

优化完成后需更新的算子列表：

1. **`planner_agent.md` 的"已优化算子列表"**（PlannerAgent 负责维护的全局履历）

两处记录格式保持一致：
```
[x] {op_name} ({状态}: {N}轮; avg {baseline}x→{final}x ({improvement}x提升); {正确性}; {关键策略}; {未达标原因}; Wiki: {url})
```

**必须由 ReporterAgent 将 Wiki URL 和关键数据提供给 PlannerAgent**，由 PlannerAgent 完成履历更新。

---

## 6. 输出规范

向 SchedulerAgent 提交：

```
报告文件:
  word_path: str          # Word 报告完整路径
  wiki_url: str           # Wiki 页面 URL

自检清单:
  items: [{id: 1-10, status: pass/fail, note: str}]
  all_passed: bool

skill 更新:
  updated_files: [str]    # 更新的 skill 文件列表
  new_strategies: [str]   # 新增策略名称
  ops_list_updated: bool  # To Be Optimized Ops 列表是否更新
```
