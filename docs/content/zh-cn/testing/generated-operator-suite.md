# 从生成分支迁移的算子测试

本分支从官方 master `fb69018199b2212757e4ab293e01eddac90360d4` 创建，仅迁移 gumptao 新增测试的最终修订内容，不合并旧 KernelGen adapter 历史，也不包含厂商 kernel。

## 来源与范围

原始提交为 `346afc7904f9fea33b615faaadd46db41fbdb161`（72 组正确性与 benchmark 测试）。迁移快照为 `kernelgen-dev@0d9c3a11a33c69bdfe663cdbaa0b34cf235041e6`，包含 gumptao 后续语义、dtype、shape、梯度、alias 和稀疏输入修正，以及之后的 sparse case 序列化修正。

后续明确删除的 `_fw_primal_copy`、`_make_dual_copy`、`_indices_copy`、`_values_copy` 不复活，因此当前是 **68 个 correctness 文件 + 68 个 benchmark 文件**。正确性文件及 `tests/test_utils.py` 与源快照逐字节一致；benchmark 的测例生成逻辑经 AST 比较保持一致，只调整测试辅助类导入及过时注释。

## 与 master 的边界

`src/flag_gems/`、`benchmark/base.py`、`benchmark/conftest.py` 完全沿用 master。候选注入使用 master 的 `--override <op>:<path>:<function>` / `DynamicOpOverride`，不迁移旧的 testing registry、resolver、case ContextVar 或修改全局注册表的算子桥接代码。

必要的测试辅助代码为：

- `tests/test_utils.py`：值域、shape、dtype 和正确性比较工具。
- `tests/conftest.py` 的单个收集 hook：quick 模式的空参数集确实产生零 case，而不是制造多条 skip；其余配置和 override hook 不改动。
- `benchmark/generated_operator_utils.py`：测试专用 `OperatorBenchmark`，仅处理算子专用 shape 和原地算子的 fresh-input 计时/采样，不重写候选选择、Preflight 覆盖报告或 override 调用统计。普通算子 Profile 直接使用 master，stateful 输入在每次采样前恢复，恢复操作不计入采样范围。
- `tests/core/test_generated_operator_inputs.py`：验证上述辅助行为及继承 master 候选入口的边界。

旧分支的 `test_operator_validation.py` 混合了旧注入机制的回归测试，未整文件搬入；算子自身的全部正确性测试仍保留。

## 验证口径

迁移验证包括 helper/override/Preflight 回归、68 个 correctness 文件的默认与 quick 收集，以及 68 个 benchmark 的 `--list-cases --level core`。收集或 list-case 成功不表示已有实现、数值通过或性能达标；没有实现的算子仍需要通过标准 override 注入候选再执行。

2026-09-24 在 H20 现有官方容器运行的结果：79 项 helper/override/Preflight 回归通过；默认收集 54,347 个 correctness case、quick 收集 10,160 个；68 个 benchmark 文件中的 80 个 pytest 入口全部完成 case-list 生成，共 1,618 个唯一 case ID。使用现有 Torch 2.7.0a0 / Triton 3.2.0，没有安装或替换运行时。本轮没有执行全部算子的正确性、性能或 KGS E2E，不能将这些收集结果称为全部算子通过。

原始本地记录：`/data/akg_kernel_bench_lite/kernelgen/runs/generated-operator-pytests-20260924/ce0451ead1e3.json`。测试源码 tree 为 `ce0451ead1e31b06232631dfdab6d5e1a980a076`，随后只补充本段验证说明；远端对应目录保留 JUnit、日志和 case-list JSON。

状态可变算子的多次 Profile capture 还要求消费 hook 的外部 profiler 支持重复采样范围；本次测试文件迁移不修改 KGS 的 capture 次数约束，也不宣称完成这类 KGS 集成验证。
