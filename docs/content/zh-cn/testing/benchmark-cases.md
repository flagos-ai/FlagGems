---
title: 枚举与精确重放 Benchmark Workload
weight: 30
---

# Benchmark Case API

`pytest --collect-only` 只能枚举测试节点，看不到 benchmark 内部的 shape、dtype、参数组合。Case API 将其拆为两步：`get_case_iter(dtype)` 生成不含输入 Tensor 的 workload 描述，`build_inputs(case)` 只为实际选择的 case 构造输入。

本功能独立于候选注入：不包含 `--override`、resolver、候选覆盖报告、Preflight 或 Profile 模式，不修改现有直接 `gems_op`、`use_gems()`、正确性 pytest 和计时方法。

## 枚举与重放

在已有 FlagGems 测试环境、仓库根目录执行：

```bash
python -m pytest benchmark/test_softmax.py::test_softmax --level core --dtypes float32 --list-cases --output /tmp/softmax-cases.json
```

输出 Schema 为 `flaggems.benchmark-case-list/v2`，顶层 `benchmarks` 数组每项包含 `op_name`、`phase="timing"`、`level`、`cases`；每个 case 包含 `case_id`、`ordinal`、`dtype`、`shape`、`params`。私有 builder 状态不进入 JSON。枚举报告每次覆盖指定文件，不合并上次运行残留的 case。

从输出中复制一个完整 ID，再执行：

```bash
python -m pytest benchmark/test_softmax.py::test_softmax --level core --dtypes float32 --case-id 'benchmark/test_softmax.py::test_softmax::core::float32::0' --record json --output /tmp/softmax-replay.json
```

`--case-id` 可重复指定，只构造并测试选中的输入，正常 benchmark 指标中会记录 `case_id`。重放仍按原路径测 reference 和 Gems，不是 candidate-only 执行。未知 ID、选中却未执行的 ID（包括 pytest deselect/skip）使 session 失败；重复指定 ID，以及 `--list-cases` 与 `--case-id` 或 `--query` 混用，属于参数错误。

ID 格式为 `pytest_nodeid::level::dtype::ordinal`，不是内容哈希。枚举和重放必须固定 checkout、pytest 根目录、shape 文件、dtype 筛选、level、case 顺序及环境相关过滤条件；修改 workload 后旧序号可能指向不同配置。枚举保存的是输入生成配方，不是冻结的随机 Tensor 数值。

规划阶段不构造输入 Tensor，不执行算子、reference 或计时；但 FlagGems 和第三方 benchmark 模块的导入仍可能依赖设备运行时，因此不承诺任意模块都能在纯 CPU 环境导入。测试函数也不得在 `bench.run()` 之前提前创建 Tensor。

## 支持范围与迁移方式

通用 unary、binary、scalar-binary、unary-out、reduction、BLAS 和 TransformerEngine GLU family 已提供 case 描述。子类若覆盖原输入循环，必须自己提供对应规划和构造方法，不能仅凭继承 family 就宣称支持枚举。BLAS 的输入 factory 每个 shape/layout 必须恰好产出一个输入 tuple，多产出或零产出都会报错，不能静默丢 workload。

尚未迁移的 `GenericBenchmark(input_fn=...)` 和自定义循环仍可正常 benchmark；请求枚举或精确选择时明确报不支持。本 PR 不批量迁移所有算子文件。

GenericBenchmark 新增 `case_fn(shape, dtype)` 和 `build_inputs_fn(plan, dtype, device)` 配对接口，替代旧 `input_fn`，不能同时传入两套。前者 yield `BenchmarkCasePlan(shape=..., params=..., builder_args=...)`；后者直接返回一个与旧实现相同的输入 tuple（包括 kwargs 字典），不能返回 generator。完整示例见[英文说明](../../en/testing/benchmark-cases.md#supported-providers-and-legacy-benchmarks)。迁移时保持全部原始 shape、dtype、参数、布局和顺序，不能通过执行旧 Tensor generator 再丢弃数据来伪装成轻量枚举。

## 测试

无需 Torch/Triton 的序列化契约测试：

```bash
PYTHONPATH=. python -m pytest --confcutdir=tests/core tests/core/test_benchmark_case_contract.py
```

在标准 FlagGems 环境执行 `python -m pytest benchmark/test_benchmark_case_api.py` 验证 provider，再在目标设备验证真实枚举和单 case 重放。候选注入需在对应功能就绪后独立验收。
