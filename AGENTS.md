# AGENTS.md —— Kunlunxin 测试/性能 Harness（总指挥）

> 本文件是这套 harness 的最高约束。任何 agent 开工前先读此文件，红线优先于一切。

## 这是什么
FlagGems 在 **kunlunxin(XPU)** 后端上的算子测试 / 基准 / 报表 harness。
用于：批量跑 pytest 算子用例、按结果分桶（timeout/error/failed）、跑基准对比 Gems Speedup、
把 `results/summary.json` 汇总成 Excel。

按 harness-over-harness 规范收拢：装备(scripts) + 交接记忆(meta) + 入口约束(本文件) 全在 `harness/` 下，repo 根保持干净。

## 元规范：本 harness 的生成方式（harness-over-harness）

本 harness 是 **meta-harness 生成器**（`harness/harness-over-harness/`，源自 https://github.com/pprp/harness-over-harness，未被 git 跟踪、随仓库交付）在 FlagGems/Kunlunxin 上的**产物实例**：该生成器承载「如何为任意目标项目搭建 harness system」的方法论、SOP 与模板；本文件（总指挥）+ `harness/meta/`（交接记忆）+ `harness/` 下装备（scripts/commands/skills）即其输出物。理解本 harness 必须了解其生成方式：

- **核心公式**：`Agent = Model + Harness`。模型不可改，Harness 是可控、可复利的部分；本 harness 的全部价值在于把 Kunlunxin 算子测试/修复/性能闭环做到可复用、可演进。
- **结构约定**：根目录只放需被 agent 工具自动发现的入口文件（`AGENTS.md`、`CODEBUDDY.md`）与人类入口 `README.md`；记忆（`harness/meta/`：`TODO.md`、`LOG.md` 等）、流程（`meta/GUIDE/`）、模板（`meta/templates/`）、调研（`meta/references/`）统一收在 `harness/` 之下，避免散落根目录干扰视线。
- **设计哲学**：
  - **文档即免疫系统**：每犯一次错，就往本文件加一条规则让它永不再犯，文档随修复越来越聪明（本文件即此类规则的累积）；
  - **渐进式披露**：本文件是 ~百行级别的路由表，细节下沉到 `HARNESS_SUMMARY.md`、`meta/GUIDE/`、`meta/references/`，一次只加载当前步骤必需的文档；
  - **双轨记忆**：`TODO.md`（结构化、给机器、防误改）+ `LOG.md`（叙事、给人类）解决换班失忆；
  - **闭环自验证**：交付前对照原始需求与本文红线 checklist 自检，不以"看起来对"停手；
  - **熵治理**：主动对抗膨胀，定期精简删除腐烂规则，harness 应趋向简单而非膨胀；
  - **机械化约束 > 口头约束**：能用 linter/CI/脚本强制的红线，不靠文档自觉。
- **标准工作流**（进入本 harness 后固定序列）：⓪ 先执行下文「使用前初始化：Bootstrap」——拉取 `https://github.com/pprp/harness-over-harness` 并按元生成器方式初始化本 harness → ① 通读本文件确认红线与原则 → ② 读 `harness/meta/TODO.md` + `harness/meta/LOG.md`（上一班同事的交接记录，不读就开干 = 失忆症）→ ③ 按任务类型打开 `harness/harness-over-harness/meta/GUIDE/` 对应 SOP → ④ 从 `meta/templates/` 取模板裁剪（不从零写）→ ⑤ 按上文「上层分工」模块化拆分并派发子 Agent（主 Agent 只调度汇总，不亲自写实现）→ ⑥ 闭环自检（对照原始需求 + 本文件红线与验收节）→ ⑦ 记账：同时更新 `TODO.md`（结构化）与 `LOG.md`（叙事）→ ⑧ 沉淀：发现可复用流程则落为 command/skill。
- **Definition of Done**：把本 harness 交给一个全新、对 FlagGems/Kunlunxin 一无所知的 agent，它应能仅凭本文件与 `harness/` 下指令、无需人类逐步指导地完成算子测试/修复/性能闭环；产出的记录自带 `TODO.md` / `LOG.md` 使下一位 agent 无缝接班；全程不触碰本文件红线；可复用流程正确沉淀为 command 或 skill。

## 使用前初始化：Bootstrap（必做，先于一切）

> **硬规则（用户 2026-09-09 明确，先于本文件其他一切条款）**：
> * **必须先拉取 harness 仓库（`harness-over-harness`）与 Kunlunxin 算子优化知识库（`KernelOpt_Knowledge`）并完成本小节初始化，之后才能开始任何算子任务**；未初始化或初始化失败时禁止开工，不得以"跳过初始化"冒充完成；拉取受阻时按第 2 步向用户求助（索要权限，或请用户人工拉取），不得静默降级为用旧副本开工。
> * **profiling 属于前置工作**：性能任务的 profiling 步骤**只从知识库获取**（`harness/KernelOpt_Knowledge/xpu3/PROFILING_GUIDE.md`，第 2 步校验其在位）。手册缺失 = 初始化未完成 = 禁止开工；不得凭记忆或通用 CUDA/Triton 经验自行编造流程。
> * 任务全程必须维护 `harness/meta/LOG.md`（叙事）与 `harness/meta/TODO.md`（结构化），并在 `harness/solution/<op>/` 下写方案 Markdown（复现与基线、根因、改动、正确性验证、性能前后结果、遗留项）。
> * **无 subagent 可用时，允许主 Agent 直接完成子 Agent 的职责**：自行实现、验证、记录；"每个算子派发独立子 Agent"是可选优化，不是开工前提（此豁免记录于 `harness/meta/LOG.md`）。

> 本 harness 是 `harness-over-harness`（https://github.com/pprp/harness-over-harness）的产物实例。**任何 agent 在本仓库开始正式任务前，必须先完成本小节初始化；未完成或初始化失败时禁止开工**（用户明确豁免并记录在 `harness/meta/LOG.md` 的除外）。初始化 = **先拉取元生成器与 Kunlunxin 算子优化知识库，再按元生成器的方式初始化本 AGENT 的 harness 体系**：

1. **拉取 / 更新元生成器**到 `harness/harness-over-harness/`（与仓库交付物一致，该目录未被 git 跟踪、不提交）：

   ```bash
   # 首次拉取
   git clone --depth 1 https://github.com/pprp/harness-over-harness harness/harness-over-harness
   # 已存在则更新到最新
   git -C harness/harness-over-harness pull --ff-only
   ```

   - 拉取失败先重试；仍失败则检查网络/代理并停下向人类报告，不得以旧副本或"跳过"冒充初始化完成。
   - 记录当前版本：`git -C harness/harness-over-harness rev-parse --short HEAD`，写入 `harness/meta/LOG.md`。

2. **拉取 / 更新 Kunlunxin 算子优化知识库**到 `harness/KernelOpt_Knowledge/`（该目录未被 git 跟踪、不提交）：

   ```bash
   # 首次拉取（SSH，端口 30004）
   git clone --depth 1 ssh://git@dev.kunlunxin.com:30004/klx/luoboyuan/KernelOpt_Knowledge.git harness/KernelOpt_Knowledge
   # 已存在则更新到最新
   git -C harness/KernelOpt_Knowledge pull --ff-only
   # 已存在的克隆若 origin 仍指向旧的 HTTPS 地址（会 404），先改正 remote 再 pull
   git -C harness/KernelOpt_Knowledge remote set-url origin \
     ssh://git@dev.kunlunxin.com:30004/klx/luoboyuan/KernelOpt_Knowledge.git
   ```

   - **该知识库只走 SSH，端口固定 30004**（用户 2026-09-18 确认）：`dev.kunlunxin.com:22`、`:2222` 在本集群均不可达，省略端口或写错端口会以 `Connection timed out` 挂住；旧 HTTPS 地址 `…/klx/_code/…` 已废弃（返回 `404 not found!`），不得再用。
   - SSH 身份由环境预先配置（`~/.ssh/id_rsa` + 平台已登记的公钥）。agent **不得自行生成或索取凭据、不得提交私钥、不得把 access token 写入仓库文件或命令行**。
   - **拉取失败的标准处置（先重试一次，仍失败则按错误定位并停下向用户求助）**，不得以旧副本、跳转网页、"跳过"或伪造初始化结果冒充完成：
     - `Permission denied (publickey)` / 认证失败 → 无仓库权限或公钥未登记：**向用户索要权限**，或请用户确认 `~/.ssh/id_rsa.pub` 已加到该仓库。
     - `Connection timed out` / `Connection refused` → 网络或端口问题：先确认用的是 `:30004`；仍不通则请用户开通访问路径。
     - `Host key verification failed` → 请用户侧确认主机指纹，agent 不得用 `StrictHostKeyChecking=no` 自行绕过。
   - **用户授权后可改为人工拉取**：请用户自行 `git clone` 或拷贝只读副本到 `harness/KernelOpt_Knowledge/`；agent 只做存在性与版本校验（`git -C harness/KernelOpt_Knowledge rev-parse --short HEAD`），全程不接触凭据；并在 `harness/meta/LOG.md` 记为"经用户授权的人工拉取"，不得冒充为 agent 克隆成功。
   - 记录当前版本：`git -C harness/KernelOpt_Knowledge rev-parse --short HEAD`，写入 `harness/meta/LOG.md`。
   - 该知识库汇集 Kunlunxin 算子优化经验，先于一切算子任务检索；查阅要求见「修复经验总纲（强制预读）」。
   - **知识库是 profiling 步骤的唯一来源**：`xpu3/KERNEL_OPT_EXPERIENCE.md`（既有优化经验）+ `xpu3/PROFILING_GUIDE.md`（运行期 profiling 操作手册）。**本步必须校验手册在位**：

     ```bash
     test -s harness/KernelOpt_Knowledge/xpu3/PROFILING_GUIDE.md || echo "MISSING: profiling 手册缺失，Bootstrap 未完成"
     ```

     缺失时按本步的求助流程停下问用户（拉取/权限问题同上），**不得凭记忆自行编造 profiling 流程，也不得跳过该手册继续开工**。
     > 注：该手册**当前仅存在于本工作区、尚未推送到知识库远程**（2026-09-18 用户决定）。因此在**新机器/他人 fresh clone** 上校验失败是**预期行为，不是仓库损坏** —— 照样按上述流程向用户索取（拷本工作区副本 / 授权后 push / 由三篇来源文档重建）。

3. **按元生成器方式初始化本 harness**（对照其「启动指令」序列，任一产物缺失即按模板补齐，不从零写）：
   - 通读 `harness/harness-over-harness/AGENTS.md`：其红线 R1–R8 为本 harness 的通用基座（R2 记忆神圣、R3 不越界写、R6 非明确指示不 commit、R7 不造轮子绕过等默认生效）；**与本文冲突时以本文为准**，并在 `harness/meta/LOG.md` 记录差异。
   - 读 `harness/harness-over-harness/meta/TODO.md` + `meta/LOG.md`（元生成器自身的交接班记录，了解最新演进与踩坑）。
   - 按任务类型打开对应 SOP：搭/改 harness → `meta/GUIDE/harness-construction.md`；subagent/team/skill/command 抉择 → `meta/GUIDE/orchestration.md`。
   - 从 `meta/templates/`（AGENTS/TODO/LOG/ideas/references/command/skill 模板）取模板，对照校验并补齐本仓库 `harness/` 产物——最小必需集：`AGENTS.md`（本文件）、`harness/meta/TODO.md`、`harness/meta/LOG.md`；推荐：`harness/meta/ideas.md`、`harness/meta/references/`。
   - 初始化或同步完成后，在 `harness/meta/LOG.md` 追加一条：拉取版本（commit）、执行了哪些动作、与模板的差异、待决问题。

4. **完成判定（换位测试）**：一个对本仓库一无所知的新 agent，仅凭 `AGENTS.md` + `harness/` 下文件（含第 2 步拉取到的 `harness/KernelOpt_Knowledge/`）能否回答"现在该干什么、怎么跑起来、什么不能碰、**性能任务的 profiling 怎么做**"；任一不能 → 回第 3 步补齐，不宣布完成。注意 profiling 的具体步骤**刻意不在本文件内**，而是由知识库手册承载。

## 本文件依赖的 Skill

本 harness 依赖以下项目级 skill，均位于 `.claude/skills/<name>/SKILL.md`（项目内、随仓库交付），当前已在位。执行对应流程前必须先通过 `Skill` 工具调用；若发现缺失，必须先按 `superpowers:writing-skills` 格式创建补齐（保持 `name`/`description` frontmatter 与正文规范）再执行，不得以缺失为由跳过对应流程：

| Skill | 用途 | 关联章节 |
|---|---|---|
| `/probe-xpu-cards` | `xpu-smi -m` 探测空闲卡、推荐 `CUDA_VISIBLE_DEVICES` | 运行前设备选择、NOC 检查与环境变量；主 Agent（总控） |
| `/probe-reset-xpu` | 检查选定卡 dmesg NOC 状态，确认存在 NOC 后对该卡执行软重置 | 运行前设备选择、NOC 检查与环境变量；主 Agent（总控） |
| `/summarize-gems-speedup` | 按 dtype 等权计算平均 Gems Speedup（先各 dtype 均值，再对 dtype 均值算术平均） | 性能修复验收；性能算子修复：benchmark 验收 |

## 修复经验总纲（强制预读）

- 每个性能或正确性修复子 Agent 在修改、测试或 benchmark 前，必须先检索 Kunlunxin 算子优化知识库 `harness/KernelOpt_Knowledge/`（找与目标算子、瓶颈类型、报错模式相关的既有经验），再阅读 `harness/HARNESS_SUMMARY.md`，再阅读本文件；**性能任务还必须读 `harness/KernelOpt_Knowledge/xpu3/PROFILING_GUIDE.md` 并按其执行运行期 profiling**。主 Agent 派发任务时必须明确此要求。该总纲汇集了已有 Kunlunxin 修复的可复用经验，不能仅凭通用 Triton/CUDA 直觉修改。
- 知识库结论与 `HARNESS_SUMMARY.md` 中的案例一样，是定位线索，不是绕过本文件红线的许可；所有候选仍必须遵守 backend-local 范围、无 fallback、同矩阵 benchmark、dtype 等权 speedup、完整 correctness 和证据记录要求。知识库未覆盖的问题按本文件流程自行定位。
- 开始定位时，先确认 ATen schema、vendor override 导出与 `SpecOpRegistrar` dispatch 是否实际生效；特别检查函数式、in-place 和 `.out` 变体，以及空张量、零尺寸、dtype/bit-cast 和数值边界语义。
- 性能问题先以同矩阵 baseline 和必要的 IR dump 分类：XPU 上优先排查 `@triton.autotune` 多配置重编译、无界 `heur_block_*`/`next_pow2` 导致的 IR 膨胀、`constexpr-N` 离散访存、恒真 mask 的 masked-memory 路径，以及过小 program 导致的 launch-bound 开销。
- 候选优先采用固定且有界的 codegen/tile 配置、连续块 DMA 或直接指针路径；对 `tl.sum` 等归约必须先覆盖大尺寸、边界 shape 和 dtype 正确性，不能把速度提升建立在不稳定的归约语义上。
- `HARNESS_SUMMARY.md` 中的案例是定位线索，不是绕过本文件红线的许可。所有候选仍必须遵守 backend-local 范围、无 fallback、同矩阵 benchmark、dtype 等权 speedup、完整 correctness 和证据记录要求。

## 上层分工

- **所有子 Agent 一律派发到后台异步运行（run_in_background/async 模式）**：主 Agent 不允许同步阻塞等待单个子 Agent 完成，必须以后台方式并发派发全部可并行算子，提高子 Agent 并发运行速度；主 Agent 在派发后立即返回继续调度后续算子，子 Agent 完成结果通过通知机制收口。
- 用户给出一批待修复算子时，每个算子修改都必须派发一个独立子 Agent；主 Agent 负责持续监控、调度和汇总，不能自己代替子 Agent 修复算子。
- **性能算子修复必须由同一个子 Agent 完成完整闭环**：该 Agent 负责修复前功能基线、性能基线、backend 最小修复、修复后性能验证、修复后功能回归、方案证据记录和结果交接；不得将同一算子的功能验证或性能验证拆分给其他子 Agent。主 Agent 只负责选卡、调度、共享文件串行化和最终复核。
- **性能修复前后功能不得回退**：修改任何性能算子前，子 Agent 必须用独立 `FLAGGEMS_CACHE_DIR` 在分配的 XPU 上运行完整对应功能测试并以 `--ref cpu` 对拍，记录命令、测试矩阵、通过/跳过/失败数和证据路径，作为功能基线。完成性能修复和性能验证后，必须以相同测试入口、相同矩阵和独立 cache 再运行一次完整功能回归。post 结果不得引入新的失败、错误、skip、精度/语义差异或原地行为回退；否则性能修复一律标记 `BLOCKED`/失败，禁止记录为 PASS 或 IMPROVED。功能基线或 post 功能回归未完成时，也不得给出性能验收结论。
- 子 Agent 最大并发数固定为 10。主 Agent 先派发至该上限；其他待修复算子保留在队列中。任一子 Agent 结束后，主 Agent 必须立即从队列派发下一个可执行算子，直到队列清空。
- 主 Agent 负责选卡、分配算子、串行化会发生文件重叠的修复任务，并复核结果。没有空闲卡时，不得抢占或使用忙碌卡；等待 1 分钟后重新执行 `xpu-smi -m` 探测，循环等待并在出现空闲卡时继续派发，直到达到 10 个运行中子 Agent 的上限或所有待修复算子均已完成。
- 每个算子修复创建一个子 Agent。子 Agent 必须直接在仓库根目录工作；不得创建或使用隔离 Git worktree。修复、测试与文档记录都必须落在当前 code 工作区，保证用户可立即通过 `git status` 和 `git diff` 审阅。
- 子 Agent 完成后，主 Agent 必须确认其修复改动已汇入当前工作区并在根目录可见；任何隔离环境、临时目录或子 Agent 会话中独有的改动都不得作为完成结果，必须先合并到当前工作区并经根目录 diff 复核。
- 主 Agent 在派发前必须明确每个子 Agent 的算子、允许修改的文件和测试命令。并发 Agent 只能修改互不重叠的文件；若两个任务可能修改同一文件，必须顺序执行。
- 子 Agent 不得提交、重置、还原或覆盖当前工作区的既有改动。完成后必须报告根目录的精确改动、测试摘要和未解决阻塞；主 Agent 负责汇总并决定 PASS 记录。

## 算子修复红线

- 禁止使用 CPU ATen fallback、CPU redispatch、native/composite fallback 或“复用基础算子”绕过 Kunlunxin/XPU kernel 问题来宣称算子修复完成。
- **fallback 判定口径**：在 backend override / kernel 实现里直接调用 torch 算子（如 `torch.add`、`torch.mm`、`torch.xxx.aten` 等 aten 路径）即属于 fallback 行为，一律禁止；`torch.empty` 这类仅做内存分配、不承担被测数值计算的调用不构成 fallback，但需 AI 逐案分析确认其未掩盖真实 kernel 问题后方可保留。
- 通过 fallback 得到的结果不得标记为 `[PASS]`，必须在 `harness/meta/TODO.md` 的独立 `[FallBack]` 区域记录，原算子继续按真实失败或待处理状态跟踪。
- fallback 只能作为诊断、reference 或环境分类手段；不得作为生产 override、correctness solution 或 regression 条目。
- 当 `torch` native/reference 算子报错或未实现时，禁止通过改用 CPU reference 来绕过、掩盖或宣称 Kunlunxin/XPU kernel 通过；CPU reference 仅可用于诊断、对拍 oracle 或环境分类。此类错误必须如实记录为失败或阻塞，不能记为 `[PASS]`。

## fallback 静态检查（机械化门禁，强制）

> 本节把「算子修复红线」的 fallback 判定**机械化**（机械化约束 > 口头约束）。任何 backend 改动在最终判定 PASS 前必须跑一次本脚本；它是静态 grep 启发式**初筛**，不替代人工 / AI 逐条确认，也不放宽红线。

**判定分层**

- **[FAIL] 疑似真实 fallback（硬关键词，脚本 `exit 1`）**：`torch.mm`、`torch.sigmoid` 等直接 torch 计算算子，以及 `torch.xxx.aten.xxx` 计算路径（如 `torch.ops.aten.native_batch_norm` / `adaptive_max_pool2d` / `nonzero` / `linalg_lu_factor_ex`）。命中即视为疑似把被测数值计算落到 torch/aten，须按红线处理。
- **[WARN] 需逐条分析（脚本不阻断）**：`torch.empty` / `torch.zeros`（含 `_like` / `_strided` / `full` / `ones`）等内存分配，以及 `copy_` / `copy` / `_copy_from` / `.contiguous(` / `.cpu(` / `.to(...cpu)` 等数据搬运。这些不一定是 fallback，须逐条确认**是否把被测数值计算或数据落到 aten / CPU 实现**。
- **数据搬运 aten 原语从 [FAIL] 下调为 [WARN]**：`torch.ops.aten._copy_from` / `copy_` / `contiguous` / `reshape` / `slice` / `clone` / `conj` / `as_strided` 不承担被测数值计算。典型如 `ops/pad.py` 直接调 `torch.ops.aten._copy_from` 走 vendor 原生 strided-copy 引擎（gems 只 override `copy_`/`copy`、不 override `_copy_from`），是有意的原生快路径而非 CPU fallback；仍须逐案确认未把计算 / 数据搬到 CPU。

**运行**（从 FlagGems 仓库根目录执行；默认只扫本次 `git` 改动的 backend 文件，`--all` 做全量体检）：

```bash
#!/usr/bin/env bash
set -uo pipefail
BK="src/flag_gems/runtime/backend/_kunlunxin"
if [ "${1:-}" = "--all" ]; then
  files=$(find "$BK" -name '*.py')
else   # 仅本次改动：未提交 + 已暂存 + 新增未跟踪
  files=$( { git diff --name-only -- "$BK"; \
             git diff --name-only --cached -- "$BK"; \
             git ls-files --others --exclude-standard -- "$BK"; } \
           | grep -E '\.py$' | sort -u )
fi
[ -z "$files" ] && { echo "no changed backend .py files (use --all to scan whole backend)"; exit 0; }

HARD_RE='torch\.(mm|bmm|matmul|addmm|sigmoid|softmax|relu|tanh|gelu)\b|torch\.[A-Za-z0-9_]+\.aten\.[A-Za-z0-9_]+'
DATA_MV='torch\.[A-Za-z0-9_]+\.aten\.(_copy_from|copy_|contiguous|reshape|slice|clone|conj|as_strided|narrow|is_non_overlapping_and_dense)\b'
COMMENT=':[0-9]+:[[:space:]]*#'
WARN_RE='torch\.(empty|empty_like|empty_strided|zeros|zeros_like|ones|full)\b|\.copy_\(|\.copy\(|_copy_from|\.contiguous\(|\.cpu\(|\.to\([^)]*cpu'

hard=$(grep -HnE "$HARD_RE" $files 2>/dev/null | grep -vE "$DATA_MV" | grep -vE "$COMMENT" || true)
warn=$(grep -HnE "$WARN_RE" $files 2>/dev/null | grep -vE "$COMMENT" || true)

echo "===== [WARN] torch.empty/zeros/copy 等：逐条分析是否落 aten/CPU ====="; echo "${warn:-<none>}"
echo; echo "===== [FAIL] 疑似真实 fallback（计算落 torch/aten 算子）====="; echo "${hard:-<none>}"; echo
if [ -n "$hard" ]; then echo "FALLBACK CHECK: FAIL"; exit 1; else echo "FALLBACK CHECK: OK"; exit 0; fi
```

**处理流程**

1. 在任何 backend 改动完成、最终判定 PASS 之前运行本脚本（并入下文「最终检查与汇报」第 11 步与「性能算子专项闭环」收尾）。
2. **[FAIL] 命中**：先确认命中行是**可执行代码**——grep 无法区分注释 / docstring / 字符串字面量，须人工排除（脚本已过滤纯 `#` 注释行，但 docstring 与字符串仍可能误报）。确认为真实调用即属红线禁止的 fallback：必须改为真实 XPU kernel；本轮无法解决的按本文件 `[FallBack]` 规则单列到 `harness/meta/TODO.md`，原算子保持真实失败 / 待处理，**不得记 PASS**。
3. **[WARN] 命中**：逐条判定该调用是否把被测语义落到 aten / CPU：
   - 纯内存分配（`torch.empty` / `zeros` 仅分配输出或中间 buffer、`fill_` 初值，不承担被测数值计算）→ 允许保留，记录判定理由。
   - `copy_` / `copy` / `_copy_from` / `.contiguous(` / `.cpu(` / `.to(...cpu)`：确认是走 vendor 原生搬运原语，还是把数据搬到 CPU 或用 aten 承担被测计算 / 语义。若属后者 → 判为 fallback，按红线禁止；若确为原生快路径则保留并记录理由。
4. **记录**：脚本原始输出与逐条判定结论写入对应 `harness/solution/<op>/` 方案 Markdown 与 `harness/meta/LOG.md`；确认的 fallback 按 `[FallBack]` 规则单列，不记 PASS。

## 算子修复记录（强制）

- 每个算子修复任务必须在 `harness/solution/<op>/` 下创建或更新对应的修复方案 Markdown；不得只保留在会话、临时日志或提交信息中。
- 修复方案至少记录：复现条件与基线、根因判断、修改的 Kunlunxin 实现/注册路径、正确性验证、benchmark 前后结果，以及仍存在的回归 shape 或阻塞项。
- 开始修复时和完成或阻塞时，都必须更新 `harness/meta/TODO.md` 中该算子的状态；fallback 仍按本文件的 `[FallBack]` 规则单独记录，不得改为 PASS。
- 每次修复任务结束时，必须在 `harness/meta/LOG.md` 追加叙事型记录，说明实际动作、关键发现、验证命令和最终结果，并链接或引用对应的方案 Markdown。

## 运行前设备选择、NOC 检查与环境变量

每次运行任何 Kunlunxin 测试、benchmark 或 profiling 前，主 Agent 必须按以下顺序执行，子 Agent 不得自行选卡、探测或重置设备：

1. 使用 skill `/probe-xpu-cards` 执行 `xpu-smi -m`，检查各卡的利用率、显存和进程占用；只选择即时空闲的卡。8 张卡（0-7）均可分配。不得复用旧探测结果或固定卡号。
2. 对选定卡使用 skill `/probe-reset-xpu` 检查 NOC。`dmesg` 为系统累积日志，只有明确指向选定卡的 NOC 才触发恢复；其他卡的历史记录不能阻塞空闲卡使用。
3. 若选定卡存在明确 NOC，直接且只对该卡执行 `printf '1' > /proc/kunlun/dev<card_id>/soft_reset`，然后重新检查该卡状态与 NOC；不得重置其他卡。若没有明确 NOC，禁止重置。
4. 仅在选定卡空闲且 NOC 检查或恢复完成后，向子 Agent 分配 `CUDA_VISIBLE_DEVICES=<card_id>`、`XPU_EVENT_KL3_ENABLE=1`、`PYTHONWARNINGS="ignore::UserWarning"` 和本进程唯一的 `FLAGGEMS_CACHE_DIR`。
```bash
# 根据上面的设备占用情况，将 <idle_card_id> 替换为空闲卡号
export CUDA_VISIBLE_DEVICES=<idle_card_id>
export XPU_EVENT_KL3_ENABLE=1
export PYTHONWARNINGS="ignore::UserWarning"
```


## 单算子测试 / dump IR（约定）

### 首选运行方式：`tools/run_tests.py`（统一收拢 log_result）

**快速定位/复现**阶段可以继续用下文各节的 `pytest`（marker、单节点、`-k`），不必删除；但**功能与性能的最终验收统一用** FlagGems 自带调度器 `tools/run_tests.py`——**该脚本一次运行会同时跑功能（accuracy）和性能（performance）测试**，分别产出 `<op>/accuracy_result.json` 与 `<op>/performance_result.json`。不得只凭 `pytest` 结果宣布功能或性能 PASS：

```bash
python tools/run_tests.py --ops "算子名" --gpus <idle_card> --dump-output --output-dir <DIR>
```

- `--ops "算子名"`：逗号分隔的算子 ID（如 `--ops "add,softmax"`），直接指定被测算子。
- `--gpus <idle_card>`：**卡号不固定**，必须由主 Agent 按「运行前设备选择、NOC 检查与环境变量」用 `/probe-xpu-cards` 探测当时空闲且健康的卡（含卡 0）后替换；不得写死 `0`，也不得复用旧探测结果。多卡时可传逗号分隔列表（如 `--gpus 2,3`）。
- `--dump-output`：把每个算子的 stdout/stderr 落盘到 `<output-dir>/<op>/`（`accuracy_stdout.log`、`accuracy_stderr.log`、`performance_*.log` 等）；不加则输出被丢弃。
- `--output-dir <DIR>`（**注意是双横线**）：本次结果输出目录，缺省为 `logs_results_YYYYMMDD_HHMM`。

**AGENT 运行期间所有经本脚本产生的 log_result 必须归拢到同一个文件夹**：本 AGENT 会话固定 `--output-dir harness/run_logs/<op>`（同一算子多轮可加时间戳后缀，如 `harness/run_logs/<op>_<HHMMSS>`），使整段 AGENT 运行的 `summary.json` / 各算子 `accuracy_*`、`performance_*` 全部落在 `harness/run_logs/` 这一个根目录下，便于统一审阅与交接。**不得**让默认时间戳目录散落到 repo 根。该目录属运行产物，不提交 git（见「红线」）。

> `run_tests.py` 是 FlagGems 仓库设施，属已跟踪文件，只调用、不改动（见「红线」）。marker/单节点定位仍可用下文各节的 `pytest` 做快测；但**功能与性能的最终验收一律以本脚本的 `accuracy_result.json` / `performance_result.json` 为准**。本调度器不豁免下文对 `--ref cpu`、独立 `FLAGGEMS_CACHE_DIR`、3 次 fresh-cache clean pass、完整 dtype/shape/layout 矩阵、dtype 等权 Gems Speedup 的约束。

### 功能算子修复：`tests/`

功能任务的标准首轮复现、修复后快速验证和子 Agent 分配命令必须从 `tests/` 目录执行 marker，并使用 CPU reference：

```bash
cd tests
pytest --timeout=900 -m "<算子名>" --ref cpu
```

`--timeout=900` 使用 pytest-timeout 将单个测试用例限制为 15 分钟；所有功能测试、修复验证和子 Agent 分配的 pytest 命令都必须保留该参数。主 Agent 必须在上述命令前完成本文件的选卡和 NOC 流程，并将分配的设备环境与唯一缓存目录传给子 Agent。marker 或单节点可用于定位；最终完整文件与三轮独立进程验收要求仍按下文执行。

```bash
# 最终完整功能文件验收
pytest -m "<算子名>" --ref cpu -svv

# 从功能测试路径 dump IR
pytest 'test_<算子名>.py::<具体测试节点>' --ref cpu -svv

# 从功能测试路径 dump IR
MLIR_ENABLE_DUMP=1 pytest tests/test_<算子名>.py -m "<算子名>" -svv \
  > harness/perf_ir/ir_test_<算子名>.log 2>&1
```

功能修复要求：

- 先在 `tests/` 下建立失败 baseline，记录失败 dtype、shape、参数和首个真实错误，再实施最小 Kunlunxin backend 修复。
- 最小失败节点通过后，必须运行对应完整 `tests/test_<算子名>.py`；最终 PASS 仍需 3 次独立进程、3 个独立 `FLAGGEMS_CACHE_DIR` 的完整 clean pass。
- 如果存在 `benchmark/test_<算子名>.py`，功能修复后必须至少完整运行一次对应 benchmark，确认 benchmark 调用路径、kernel 编译和执行没有因功能修复而回退；不能只验证 `tests/` 路径。
- benchmark 出现编译失败、运行错误或原有可用 case 退化为失败时，功能修复不能直接验收，必须继续定位或撤销候选。

## 性能修复验收

- **达标目标**：功能结果 = Passed（对应完整 `tests/test_<算子名>.py` clean pass）；性能结果 = Passed（对应完整 benchmark 无编译/运行错误、无 case 退化）；性能加速比 >= 0.8（按 dtype 等权 Gems Speedup 计算）。三者同时满足才记 PASS。
- 性能候选只有在与初始 baseline 使用完全相同的 dtype、shape、layout 和参数矩阵下，按 dtype 相等计算的 Gems Speedup 严格优于初始 baseline，才属于有效性能修复；不得将运行波动或未改源码的重复测量视为修复。
- **有效性能修复的源码必须保留在工作区**：候选同等矩阵下严格优于 baseline（无论是否达到 `0.8x`），只要完整正确性回归不回退，就必须把候选源码保留在生产工作区并在 `git status`/`git diff` 可见；不得因未达 `0.8x` 而回退或撤销有效提升。保留的源码须在方案记录中写明改了什么、前后指标并与 JSON 对应。
- 最终性能验收仍要求该候选的按 dtype 相等 Gems Speedup `>= 0.8x`。候选相对 baseline 提升但仍低于 `0.8x` 时，记录为“未达最终性能门槛的 BLOCKED（源码已保留，有效提升）+ 前后指标”，不得记为 PASS；只有**无提升或回退**的候选才需要回退源码。保留有效源码的算子应同步完成完整功能回归与至少一次 fresh-cache 复现，证据同时更新。

### 性能优化必须先 profiling（强制）

> **profiling 的步骤从知识库获取，本文件不重复维护命令细节**（用户 2026-09-18 指令）。
> * 操作手册：`harness/KernelOpt_Knowledge/xpu3/PROFILING_GUIDE.md` —— §0 是 30 秒速查，其余章节是环境变量、analyzer 路径、输出解读，以及**实测与原文不一致的坑**。
> * **这份手册本身就是 Bootstrap 前置工作**：知识库未拉取、或知识库里没有该手册 ⇒ **视为初始化未完成，禁止开工**，按 Bootstrap 第 2 步向用户求助；**不得凭记忆或通用 CUDA/Triton 经验自行编造 profiling 流程**。

性能定位固定用**三种互补手段**，只做其中一种不得进入候选实施：

| # | 手段 | 回答什么问题 | 产物落点 |
|---|---|---|---|
| ① | **检索知识库** | 这类慢源**以前见过没有**、已知解法与收益量级 | `harness/KernelOpt_Knowledge/xpu3/KERNEL_OPT_EXPERIENCE.md` |
| ② | **dump IR** | kernel **长什么样**：`stride`/`isDiscrete`、masked-memory 路径、块大小、访存形态 | `harness/perf_ir/ir_*.log` |
| ③ | **运行期 profiling** | **时间花在哪个 kernel**：各 kernel 的 ms / 调用次数 / 占比 | `harness/perf_ir/prof_*.log` + `kernel_summary_*.md` |

- **手段 ③ 的命令、参数、环境变量与全部已知坑，一律以知识库手册为准**（本机 analyzer 已装于 `~/.comate/skills/p800-profiling-analyzer/`，换机按手册 §1 安装）；本节只规定「必须做」与验收标准。
- 性能任务在动手改 kernel 之前，必须先完成一次 **pre-profiling**，用采集数据指出瓶颈归属（计算 / 访存 / launch / 调度 / 编译开销），不得凭直觉直接堆 tile、warp、stage 参数。③ 必须回答「**目标 kernel 是不是耗时大头**」——实测中 FlagGems `add` kernel 只占 17.6%，80.8% 花在随机初始化 kernel 上，不看 ③ 就会优化错对象。
- 性能候选实施后必须再做一次 **post-profiling**，以同口径前后对比证明瓶颈指标确有改善；对比须同时看**目标 kernel 的 ms 与调用次数**（次数变化可暴露「1 个 kernel 被拆成 3 个」的伪优化）。缺少 profiling 前后对比的性能候选不予验收，也不得标记 PASS。
- profiling 必须覆盖对应 benchmark 定义的完整 dtype / shape / layout 矩阵中的代表性 case，并保留原始采集产物与判读结论（含瓶颈位置、判据指标、改动前后数值）。
- profiling 的选卡、NOC 检查与环境变量要求同「运行前设备选择、NOC 检查与环境变量」；采集产物统一落 `harness/perf_ir/`（或方案记录中注明的路径），不提交 git。
- 手册的来源文档（内部知识库，仅供追溯；**内容冲突时以本地手册中的实测标注为准**）：
  - https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/mQGHFx5Gg3/5Wt-7zwtp0IHpt
  - https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/Lgo_nZo59V/9MRI6XEGRngRdB
  - https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/SPQ7rbIxN3/3TB7La0G2fE-nX

## 多 Agent 协作分工

### 主 Agent（总控）

主 Agent 负责整个算子修复任务的流程控制，不得绕过本文件的验证约束：

- 读取并向子 Agent 明确本文件中的修复步骤、Kunlunxin backend 修改范围、正确性验收和性能验收规则。
- 在每次测试、benchmark 或 profiling 前调用 `/probe-xpu-cards` 探测设备，选择当时空闲的卡（含卡 0）。
- 使用 `/probe-reset-xpu` 检查选定卡的 NOC 状态；只有确认存在 NOC 问题并获得必要确认后，才执行该卡的软重置。
- 为每个待修复算子创建一个新的子 Agent，明确算子范围、baseline、验证命令、独立缓存目录和交付记录要求；不同算子不得复用同一个修复子 Agent 的任务上下文。
- 监控子 Agent 的进度、测试结果和阻塞项；发现违反设备、fallback、代码范围或验收约束时，立即暂停该任务并要求纠正。
- 收集并复核子 Agent 的实现、完整测试结果、benchmark baseline/post、`/summarize-gems-speedup` 汇总结果及 `solution/`、`TODO.md`、`LOG.md` 记录，最后统一判定 PASS、失败或阻塞。
- 本批次全部算子任务判定完成后，按「代码风格 CI 门禁（全部算子任务结束后，强制）」在仓库根目录跑一次 `pre-commit run --show-diff-on-failure --color=always --all-files` 并修到全绿，未通过不得宣布批次完成。

### 修复子 Agent

修复子 Agent 只负责主 Agent 分发的一个算子：

- 先复现并记录 baseline，再定位根因和实施最小 Kunlunxin backend 修复。
- 不负责选卡、探测卡或软重置卡；设备动作由主 Agent 统一执行并通过任务上下文告知。
- 不得修改 `src/flag_gems/ops/`、`src/flag_gems/fused/`，不得使用 CPU/ATen/native/composite fallback 作为修复或 PASS 依据。
- 按要求完成完整正确性测试和 benchmark；性能任务必须先按「性能优化必须先 profiling（强制）」完成 pre/post profiling 并保留采集证据；性能结果必须使用 `/summarize-gems-speedup`，只报告性能算子 PASS 与平均 Gems Speedup。
- 完成或阻塞时向主 Agent 返回修改文件、验证命令、原始结果、失败原因和遗留问题，并更新对应的修复方案及 harness 元数据。

主 Agent 只有在子 Agent 的结果经过独立复核、且所有强制记录和验收条件满足后，才能汇总为最终结果。

### 性能算子修复：benchmark 验收

- 性能验证和结果汇总必须使用 skill `/summarize-gems-speedup`；Gems Speedup 按 dtype 等权计算：先分别计算每种 dtype 的平均 Gems Speedup，再对 dtype 均值做算术平均。
- 性能算子修复最终只看两个指标：一是性能算子 PASS，二是性能算子的平均 Gems Speedup。固定 Gems latency、单个 shape 的 speedup 或未按 dtype 等权的汇总值不得作为额外验收指标。
- 性能修复仍必须保留完整 benchmark baseline 和同矩阵 post，并先完成对应完整 `tests/test_<算子名>.py` 的正确性验证；benchmark 编译失败、运行错误或性能算子未 PASS 时，性能修复不能验收。

### 性能算子修复：`benchmark/`

性能任务的标准首轮 baseline、修复后快速验证和子 Agent 分配命令必须从 `benchmark/` 目录执行 marker：

```bash
cd benchmark
pytest -m "<算子名>"
```

主 Agent 必须在上述命令前完成本文件的选卡和 NOC 流程，并将分配的设备环境与唯一缓存目录传给子 Agent。该命令用于启动性能任务和定位热点；性能验收仍必须覆盖该文件定义的完整 dtype、shape、layout 和参数矩阵，并保留同矩阵 baseline/post。

```bash

# marker 或单 case 用于定位热点
pytest -m "<算子名>" -svv -s

# 从性能 benchmark 路径 dump IR
MLIR_ENABLE_DUMP=1 pytest benchmark/test_<算子名>.py -m "<算子名>" -svv -s \
  > harness/perf_ir/ir_benchmark_<算子名>.log 2>&1
```

性能修复要求：

- 修改前保存完整 benchmark baseline；候选通过后使用相同设备、模式和完整 case 矩阵运行 post，并按性能专项闭环计算 dtype-balanced speedup 和固定 Gems latency。
- 每个性能候选修改后都必须立即运行对应完整 `tests/test_<算子名>.py`，确认功能、dtype、layout、索引、随机状态、alias 等语义没有回退；benchmark clean pass 或 speedup 提升不能替代功能验收。
- 性能候选最终保留前，对应完整功能文件必须完成 3 次独立 fresh-cache clean pass；任何功能回退都必须先修复或撤销性能候选。

### 双向不回退原则

- 功能修复和性能修复都必须同时验证 `tests/` 与存在的对应 `benchmark/` 路径：功能修复不能破坏 benchmark，性能修复不能破坏功能测试。
- `tests/` 是功能正确性的唯一 PASS 判据；`benchmark/` 是性能收益和 benchmark 路径可用性的判据，两者不能互相替代。
- 诊断时允许使用 marker、`-k` 或单节点缩小范围；最终功能验收和完整性能验收禁止用这些选项缩小目标矩阵。
- IR 里看性能信号：`triton_xpu.load`/`store` 的 `stride`（1=连续块 DMA，-1=离散 gather/scatter）与 `isDiscrete`；gm2lm/lm2gm 创建阶段的 `offsetState=-1` 是默认值，不是最终判据。
- dump 产物统一落 `harness/perf_ir/`，不提交 git。

## 回归测试规范

### 回归清单

- 回归入口是 `harness/run_regression.sh`，测试清单是 `harness/meta/regression.txt`。
- `regression.txt` 每个非空、非注释行是一条相对 `tests/` 的 pytest 规格：
  - 整个测试文件 clean pass：`test_log2.py`
  - 只有 marker 子集 clean pass：`test_floor_divide.py -m floor_divide_scalar_`
  - 需要 CPU reference：`test_norm.py --ref cpu`
- 只允许加入 flag_gems 侧 clean pass 的目标。参考侧 non_bug、全 skip 或仍有失败的目标不能作为 PASS 加入。
- 每修复一个新算子，目标测试完成 3 次独立 fresh-cache clean pass 后，立即追加回归项，并同步更新 `meta/TODO.md` 和对应中文报告。
- 每次对算子进行修复后，都必须在 `harness/meta/LOG.md` 写入记录，说明修复的算子、问题、改动、验证结果及回归情况；不能只更新 TODO 或正确性报告。

### 什么时候跑回归

- 修改目标算子实现：运行对应目标测试，并完成 3 次独立 fresh-cache clean pass。
- 默认不运行完整 `harness/run_regression.sh`；只有用户当次明确要求时才运行。
- 修改相关公共代码、注册逻辑或多个算子共享的 kernel/helper 时，必须扩大受影响目标测试范围，并在最终汇报中说明回归风险；是否运行完整 regression 仍以用户当次要求为准。
- 只修改测试状态、文档、报告、`meta/TODO.md` 或 `meta/regression.txt`，没有修改算子实现及相关公共代码时，只需运行受影响的目标测试。
- 新修复算子的有效验证必须使用 3 个独立进程和 3 个独立 `FLAGGEMS_CACHE_DIR`；不能用同一编译缓存重复计数。

### 单算子修复整体流程

给出一个算子后，必须完成以下闭环，不能只修改 kernel 或只跑一次测试：

1. **确认任务范围**
   - 确认算子名称、实际 ATen schema、测试文件、pytest marker，以及是否需要 `--ref cpu`。
   - 检查 `harness/meta/TODO.md` 中的当前状态、已有报告、历史修复和回归项。
   - 判断问题属于通用实现还是 Kunlunxin/XPU backend override。

2. **检查环境和设备**
   - 激活 `.venv` 并 `source harness/env.sh`。
   - 每次使用 XPU 前使用skill/probe-xpu-cards选卡，使用 skill /probe-reset-xpu 检查卡是否 noc，只使用本次检查中空闲且健康的卡。
   - 必要时先做最小设备探测，设备异常结果不能计入算子验证。

3. **建立 baseline**
   - 使用独立 `FLAGGEMS_CACHE_DIR` 运行原始完整测试目标，并输出 JSON report。
   - 记录 passed/failed/skipped 数、失败 dtype/shape/参数/schema 和首个明确错误。
   - 区分 flag_gems 错误、参考侧错误、环境问题和设备问题；不能依赖旧记录代替本次 baseline。

4. **定位根因**
   - 沿测试 API、实际 ATen schema、`_FULL_CONFIG`、通用实现、backend override、backend import/`__all__` 和 Triton kernel 逐层确认真实调用链。
   - 必须证明测试实际执行了准备修改的实现，不能只根据函数名推断分派路径。

5. **分类处理**
   - 真实算子问题：实施修复。
   - 参考侧或测试环境问题：记录为 `NON_BUG`，不能伪装成算子 PASS。
   - 设备不支持或目标全 skip：记录为 `SKIP`，不能加入 PASS 回归。
   - 当前已经 clean pass、仅状态滞后：不修改 kernel，只完成三次验证和元数据闭环。
   - 尚未闭环：保留 `DEEP/DEFER`，不能提前标记 PASS。

6. **实施最小修复**
   - 优先做作用域最小的 Kunlunxin backend 修复，避免无关重构和其他后端行为变化。
   - 复用项目已有 kernel、helper、pointwise codegen 和配置，不为假设需求增加抽象。
   - backend override 必须同时检查实现模块、`ops/__init__.py` import、`__all__` 和 ATen schema 注册。
   - 保留工作树中已有的无关修改，不覆盖、不回退用户改动。

7. **验证最小失败用例**
   - 先用独立 fresh cache 运行 baseline 中最小且明确的失败节点。
   - 确认根因修复后，再运行完整文件或完整 marker；局部节点通过不能直接标记 PASS。

8. **三次完整目标验证**
   - 同一个完整目标执行 3 次，每次使用独立进程、独立 `FLAGGEMS_CACHE_DIR` 和独立结果目录。
   - 三次都必须满足 pytest 退出码为 0、没有 failed/error，并且不是全 skip。
   - 任一次失败都不能更新为 PASS；必须继续定位并重新完成三次验证。

9. **更新修复归档**
   - 更新 `harness/meta/TODO.md`：移动状态、同步 PASS 计数和简要说明。
   - 更新 `harness/meta/regression.txt`：追加经过三次 clean pass 的可执行 pytest 规格。
   - 写入对应中文正确性报告：记录 baseline、根因、改动、三次验证和已知限制。
   - 写入 `harness/meta/LOG.md`：记录算子、问题、实现改动、验证结果和回归情况。

10. **按要求执行完整回归**
    - 默认不运行 `harness/run_regression.sh`；只有用户当次明确要求时才运行。
    - 用户要求完整回归时，参数解析失败需要正确引用后手工复验，不能直接忽略或算作算子失败。
    - 未运行完整回归时，最终汇报必须明确说明已完成的目标测试范围和剩余回归风险。

11. **最终检查与汇报**
    - 判定结果前，先按「收尾 AB 测试（强制）」跑一次 A（改动前）/ B（改动后）对比，确认 B 相对 A 无回退。
    - 运行「fallback 静态检查（机械化门禁）」脚本；[FAIL] 命中未澄清或未修复前不得判 PASS，[WARN] 命中须逐条给出是否落 aten/CPU 的结论。
    - 对修改的 Python 文件运行 `python -m py_compile`，检查 backend import 和 `__all__`。
    - 核对三次目标结果、TODO 状态和计数、回归项、中文报告及 `LOG.md`。
    - 不提交日志、JSON report、缓存、core dump 等运行产物。
    - 默认不 commit、不 push；只有用户明确要求时才执行。
    - 最终汇报实现改动、三次目标测试、按要求执行的回归、归档更新和仍存在的限制。

### 性能算子专项闭环

性能修复除上述正确性流程外，还必须完成以下门禁；只归档 `DEEP/DEFER`、只跑局部 case 或只观察 speedup 都不算完成。

1. **确认范围和保护项**
   - 优先只修改 `src/flag_gems/runtime/backend/_kunlunxin/`；确需修改其他目录时，先在 `harness/meta/TODO.md` 登记原因和范围。
   - 开工前检查 `harness/meta/blacklist.txt`，不得修改、恢复 dispatch 或加入 regression 的黑名单算子。
   - 先检查当前 `git diff`，只撤销自己本轮的失败候选，不覆盖用户或其他并发修改。

2. **建立完整 baseline**
   - 修改前运行对应完整 `tests/` 功能文件，记录 passed/failed/skipped、首个真实错误及失败 dtype/shape/参数。
   - 运行对应完整 kernel 和 operator benchmark，覆盖 benchmark 定义的全部 dtype、shape、layout 和参数矩阵，并保存原始 JSON 或完整日志。
   - 单 case、`-k` 和 marker 只能用于定位，不能替代 baseline 或最终验收。

3. **实施单变量候选**
   - 动手改 kernel 前，先按「性能优化必须先 profiling（强制）」完成 pre-profiling，用数据锁定瓶颈再实施候选，并保留采集产物。
   - 一次只修改一个 kernel、config、tile、warp、stage、循环或 dispatch 路径，确保收益可归因。
   - 禁止 CPU/native/ATen redispatch/Composite fallback，禁止新增 skip、deselect、放宽容差或借助基础算子绕过真实 XPU kernel 问题。
   - 优先修复明确瓶颈：错误 mask/索引/位宽、长动态循环、重复 reduction、中间 tensor、额外 launch、broadcast/clone 或不合适的 XPU codegen 配置。

4. **即时功能门禁**
   - 候选修改后先运行最小失败节点；仍失败、编译崩溃或 lowering 超时时立即撤销，不进入 benchmark。
   - 最小节点通过后，立即运行对应完整 `tests/` 文件。benchmark 候选必须做功能回归，benchmark clean pass 不能单独作为通过条件。
   - 最终功能验收不能使用 `-k`、marker、skip 或 deselect 缩小完整目标范围。

5. **完整性能门禁**
   - 功能无回退后，按 baseline 相同口径运行完整 kernel 和 operator benchmark。
   - 聚合规则固定为：先对每个 dtype 的所有 case speedup 求算术平均，再对各 dtype 均值求算术平均，得到 dtype-balanced speedup。
   - 除 speedup 外，必须逐 case 和逐 dtype 比较固定 Gems latency；Torch baseline 抖动导致的 speedup 假提升不能保留。
   - 首次提升必须重复完整 benchmark；结果波动时执行时间相邻、参数完全一致的受控 baseline/post A/B。
   - 只有功能无回退、完整矩阵成功、dtype-balanced speedup 提升且 Gems 固定延迟可复现改善时才能保留候选。

6. **失败候选和最终验收**
   - 任一功能或性能门禁不通过，立即撤销本轮候选，记录命令、结果、失败原因和后续方向，不在失败候选上继续叠加参数。
   - 候选通过性能门禁后，使用 3 个独立进程和 3 个独立 `FLAGGEMS_CACHE_DIR` 完整运行对应功能文件，三次都必须 clean pass 且不能全 skip。
   - 最终判定前按「收尾 AB 测试（强制）」跑一次 A（改动前，`git stash` 还原）/ B（改动后）受控对比，用 `/summarize-gems-speedup` 确认 B 相对 A 严格不回退且达标。
   - 完成后立即更新 `harness/meta/TODO.md`、`harness/meta/regression.txt`、`harness/meta/LOG.md`，并在 `harness/solution/performance/` 写中文证据，包含根因、profiling 前后对比、改动、三轮功能结果、完整 benchmark、dtype 聚合、固定 Gems latency 和已知限制。

### 运行前条件

1. 在仓库根目录激活环境并加载 Kunlunxin harness 环境：

   ```bash 
   source harness/env.sh
   ```

2. 每次使用 XPU 前必须先检查设备：

   ```bash
   xpu-smi
   ```

3. 只选择本次检查中空闲且健康的卡；不能仅依据历史卡池判断。
4. 卡状态不确定时使用单张已确认健康的卡，例如 `DEVICES=2 PARALLEL=1`；只有多张卡都确认健康和空闲时才提高并行度。

### 目标算子验证

单次 fresh-cache 测试示例：

```bash
CUDA_VISIBLE_DEVICES=2 \
FLAGGEMS_CACHE_DIR=$(mktemp -d) \
python -m pytest tests/test_aminmax.py -q \
  --record json --output /tmp/aminmax_regression/report.json
```

> ⚠️ `--output` 是 **JSON 报告文件路径**（`tests/conftest.py` 直接 `open(REPORT_FILE, "a+")`），不是目录；传目录会以 `IsADirectoryError` 在 `pytest_terminal_summary` 阶段把一次全绿运行变成 exit 1（2026-09-18 冒烟实测）。

新修复算子需要把同一目标以独立 `FLAGGEMS_CACHE_DIR` 完整执行 3 次。每次都必须满足：pytest 退出码为 0、没有 failed/error，并且不是全 skip。

### 执行完整回归

保守单卡执行：

```bash 
source harness/env.sh
xpu-smi
DEVICES=2 PARALLEL=1 bash harness/run_regression.sh
```

使用 skill /probe-xpu-cards  选卡和 skill  /probe-reset-xpu 检查卡是否 noc，后并行执行：

```bash
DEVICES=<select card> PARALLEL=3 bash harness/run_regression.sh
```

脚本会：

- 读取 `meta/regression.txt` 的全部有效规格。
- 每条规格启动独立 pytest 进程，并按 round-robin 分配到 `DEVICES`。
- 为每条规格创建独立 `FLAGGEMS_CACHE_DIR`、日志和 JSON report。
- 用 `PARALLEL` 控制并发数。
- 任意规格退出码非 0 或摘要包含 failed/error 时记为 FAIL；最终只在 FAIL 数为 0 时返回成功。
- 在结束时打印 OK/FAIL 数和临时日志目录。失败时必须检查对应日志，不能只看总数。

### 已知参数解析限制

当前 `run_regression.sh` 使用 shell 字符串拆分规格，`regression.txt` 中带引号的多词 `-k` 表达式可能被错误拆分并产生 pytest exit 4。当前已知受影响的规格包括：

```text
test_conv_transpose2d.py -m conv_transpose2d --ref cpu -k 'not unsupported_dtype_raise'
test_rnn_relu.py -m rnn_relu -k 'direct or large_hidden or dtype1'
test_tril.py -m tril_ -k 'not expanded and not overlapping'
```

若完整回归只有这些规格因参数拆分失败，必须使用正确引用手工复验，例如：

```bash
CUDA_VISIBLE_DEVICES=2 FLAGGEMS_CACHE_DIR=$(mktemp -d) \
python -m pytest tests/test_rnn_relu.py \
  -m rnn_relu -k "direct or large_hidden or dtype1" -q
```

只有所有真实测试都 clean pass 后，才可判定本轮回归有效通过。参数解析失败本身不能记为算子失败，也不能在未手工复验时忽略。

## 收尾 AB 测试（强制，任何算子改动结束后必做）

> **硬规则**：任何算子改动（功能或性能）在闭环流程走完、准备判定结果之前，**必须再跑一次受控 AB 测试**作为最终验收门禁。A = 改动前版本，B = 改动后版本；缺少这次 AB 对比，一律不得判定 PASS / IMPROVED。这是独立于上文各闭环步骤的**通用收尾步骤**，无论算子属于功能还是性能任务都适用。

**A / B 定义**

- **A（baseline，改动前）**：改动结束后，用 `git stash` 临时贮藏本轮改动（或对本轮改动的文件 `git checkout --` 还原到改动前状态），在这份「干净树」上跑一次；A 只用于对比，不是最终工作区状态。
- **B（改动后）**：`git stash pop`（或撤销还原）把本轮改动完整恢复到工作区后再跑一次。**AB 测试结束后工作区必须停在 B 状态**：本轮有效改动完整可见于 `git status` / `git diff`，不得因贮藏/还原丢失。

**AB 采集要求**

- **只贮藏 / 还原本轮自己的改动**：AB 前先 `git diff` 确认改动范围；`git stash` 只对本轮改动生效，不得贮藏或覆盖用户或其他并发 Agent 的既有改动。恢复后必须再次 `git diff` 核对改动无缺失、无多余，与 AB 前一致。
- **A、B 必须时间相邻、环境一致**：同一张卡、同一套环境变量（`CUDA_VISIBLE_DEVICES` / `XPU_EVENT_KL3_ENABLE` / `PYTHONWARNINGS`）、各自独立的 `FLAGGEMS_CACHE_DIR`（A、B 不复用同一编译缓存）、完全相同的 dtype / shape / layout / 参数矩阵；选卡与 NOC 检查按「运行前设备选择、NOC 检查与环境变量」执行。
- **性能算子**：A、B 均跑对应完整 `benchmark/test_<算子名>.py` 全矩阵，用 skill `/summarize-gems-speedup` 按 dtype 等权计算 Gems Speedup，逐 dtype 与整体对比 B vs A；B 相对 A 必须严格不回退，且满足上文性能验收门槛（dtype 等权 Gems Speedup `>= 0.8x`）。
- **功能算子**：A、B 均跑对应完整 `tests/test_<算子名>.py`，对比 passed / failed / skipped 与精度 / 语义 / 原地行为；B 不得相对 A 引入任何新的 failed / error / skip 或语义回退。
- **B 不得回退**：AB 对比中 B 若相对 A 出现任何回退（性能变慢、功能变差、benchmark 编译/运行错误、case 退化），本轮改动一律标记 `BLOCKED` / 失败，不得记为 PASS / IMPROVED。

**AB 记录要求**

- AB 的 A、B 原始产物（benchmark JSON / 完整日志、speedup 汇总、tests 结果）统一落 `harness/perf_ir/` 或方案记录中注明的路径，不提交 git。
- 在 `harness/solution/<op>/`（性能任务用 `harness/solution/performance/`）对应方案 Markdown 中写明：A/B 各自命令、卡号与缓存目录、完整矩阵、A vs B 的逐 dtype 与整体指标、结论（B 是否无回退 / 达标）。
- 在 `harness/meta/LOG.md` 追加一条 AB 记录：本轮算子、A/B 版本来源（git stash / 还原点 commit）、对比指标与最终结论，并链接方案 Markdown。

## 代码风格 CI 门禁（全部算子任务结束后，强制）

> FlagGems 仓库用 pre-commit 做 code style CI。**本批次全部算子任务修复 / 验证完成后**，主 Agent 必须在 FlagGems 仓库根目录跑一次全量 pre-commit，作为交付前的代码风格门禁；未全绿不得宣布批次完成。这是**批次级收尾**（全部算子结束后跑一次），不替代单算子的功能 / 性能 / 收尾 AB / fallback 门禁，单算子门禁也不替代它。

```bash
pre-commit run --show-diff-on-failure --color=always --all-files
```

- 该命令覆盖 `.pre-commit-config.yaml` 的全部 hook（flake8 / clang-format / isort / black / end-of-file-fixer / trailing-whitespace / check-yaml 等），检查范围是**整个仓库全部文件**，不是仅本次改动。
- 部分 hook（end-of-file-fixer、trailing-whitespace、isort、black）会**自动改写文件**：首次运行若有改写会以非 0 退出并留下 diff。必须 `git diff` 复核这些自动格式化改动，确认只落在本批次算子文件上，再重跑直到全绿（exit 0、无剩余 diff）。
- **不擅改无关文件**：若自动格式化触及本批次之外的既有文件（仓库历史风格债），按红线不擅自改动仓库设施 / 无关文件——用 `git checkout -- <file>` 还原这些无关改动并向用户报告，不把它们混入本批次交付。
- flake8 / clang-format 报错须真实修复源码，禁止用 `# noqa`、放宽 `--ignore`、`--no-verify` 或跳过 hook 绕过。
- 结果（通过 / 修复了哪些风格问题 / 是否有无关文件被还原）记入 `harness/meta/LOG.md`。

## 红线
- 不要提交 `results*/`、`logs_results*/`、`harness/run_logs/`、`*.log`、`*.xlsx`、`*.tar.gz` 等产物到 git；`harness/harness-over-harness/` 与 `harness/KernelOpt_Knowledge/` 是外部拉取的目录，一律不提交。
- 不改动 FlagGems 自身的 `tools/run_tests.py` 等已跟踪文件——那是仓库设施，不是本 harness。
- 脚本里默认注释掉的行是历史筛选状态，放开前先确认，别一次性全跑爆机器。
- 改完任务务必更新 `meta/TODO.md` 与 `meta/LOG.md`（硬规则）。
