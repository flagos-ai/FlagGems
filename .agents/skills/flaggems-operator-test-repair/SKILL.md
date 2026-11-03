---
name: flaggems-operator-test-repair
description: 在当前后端机器上修复 FlagGems 算子自身的测试、benchmark 和 Definition，处理 override 绕过、case 列举、输入构造及 reference 部分参数不支持。reference 整体不可用时记录并暂不处理；不承担 kernel 开发或跨机基线采集。
---

# FlagGems 算子测试修复

Agent 已运行在目标后端机器及所需容器/环境中，从当前仓库执行。只验收当前后端，不配置 SSH、不调度其他机器、不采集或接入跨机性能基线。该 skill 不扩大用户授权；只读问题只调查，发布与合并遵循当前用户要求。

目标是让原有测试按原契约实际运行，并通过 override 命中 candidate。ref-as-candidate 证明测试链路可运行，dummy 证明触达；均不独立证明 kernel 数学正确或性能达标。

## 修改范围

默认只改目标算子自身的三件套：

- 实际对应的 `tests/test_*.py`。
- 实际对应的 `benchmark/test_*.py`。
- `definitions/<op>.json`。

共享文件只改该算子相关部分，保留其他算子函数和行为。复用已有 helper；需要新增局部辅助函数时放自身测试文件。不得顺带修改公共框架/helper（包括 conftest、accuracy_utils、benchmark/base 和 runner）、kernel、注册/导出、CI、conf/operators.yaml 或其他算子的 Definition。确需出界先报告路径、原因与影响；没有明确额外授权则保留阻断。

自身 Definition 可以修正名称、参数、输出及 mutation/alias，但须依据实际 API 契约；不得删参数、换 overload 或弱化语义来通过测试。缺失时明确说明根据源码派生，不能冒称恢复原始文件。

## 执行流程

1. **固定原版本。** 读仓库指令，记录用户指定算子、源码提交、本机后端/运行时/设备及交付粒度。使用独立 feature 分支/工作树，不覆盖其他任务或升级公共运行时。检查实际加载的源码路径，避免测到其他安装副本。
2. **固定原覆盖。** 保存完整正确性参数组合、core catalog 和原始失败。若构造/listing 已失败，从原 builder 与配置独立还原覆盖；不能拿修后的输出证明原覆盖。核对 Definition、marker、benchmark op_name、override key、candidate 实际调用和 native API/overload。
3. **分类修复。** 阅读 [故障分类](references/repairs.md)。输入/签名/注入问题只在自身三件套修；reference 部分参数不支持时依据真实失败加精确局部 skip。**reference 整个 API 不可用时记录并暂不处理该算子**，不新增 CPU oracle、不采集静态基线、不用全量 skip 冒充修复。其他可处理算子继续。
4. **本机完整验收。** 按 [命令与验收](references/workflow.md) 执行默认和 CPU reference 两种模式完整参数集的 ref-as-candidate、两种模式 dummy、core dummy、core list-cases、真实 KGS inspect 和实际 core benchmark。CPU 模式保留用于数值比较，不代表引入新 CPU reference 方案。
5. **交付。** 每个算子的必过项全部通过后才能放入 PR；环境阻断、整体 reference 缺失、全 skip/零 case 等单列未完成。按当前用户要求分批（默认每 PR 最多 10 个算子），每个 PR 至少两轮独立 subagent review；资源不可用时说明 review 缺口，不冒称完成。只推 feature 分支并提 PR 到指定分支（默认 kernelgen-dev），不直接推目标分支；合并需用户明确授权。

正确性先断言 candidate 的 device/dtype/shape、mutation/alias 和必要布局，再搬 CPU 做数值比较；不能通过搬运/cast/裁剪 candidate 掩盖契约错误。不得扩大容差、删原参数组合或掩盖合法负例。

benchmark 使用 `--override` 时，不向 `OperatorBenchmark` / `GenericBenchmark` 显式传 `gems_op=flag_gems.XXX`：它可能在注入前触发缺失属性错误。保留正确 op_name、原生 torch_op、builder 和其他配置，由既有 runner 执行时解析 override；清理因此无用的 import。

提交和 review 前检查 `git diff --name-only BASE...HEAD`，并检查共享文件只改自身部分。报告记录源码/工具提交、完整命令、环境、逐模式 pass/fail/skip、case 数及 ID、原始报告、review 和阻断项。保留失败证据与待 review 源码，清理已确认无用且无进程依赖的本任务中间产物，不触及其他人的任务。
