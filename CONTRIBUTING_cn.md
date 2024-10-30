[English](./CONTRIBUTING.md)

# FlagGems 贡献者指南

感谢您对 FlagGems 的兴趣！我们使用 GitHub 来托管代码、管理问题和处理拉取请求。在贡献之前，请阅读以下指南。

## 1. 错误报告
请使用 GitHub 的 Issues 来报告错误。在报告错误时，请提供：

- 简单摘要，
- 复现步骤，
- 确保描述具体且准确！
- 如果可以提供一些示例代码将会很有帮助。

## 2. 代码贡献
在提交拉取请求时，贡献者应描述所做的更改以及更改的原因。如果可以设计测试用例，请提供相应测试。拉取请求在合并前需要 __一位__ 成员的批准，而且需要通过代码的持续集成检查。

目前持续集成检查设有四条流水线:

### 2.1 代码格式检查
Code Format Check 使用 FlagGems 的 pre-commit git 钩子，可以在调用 git commit 命令时格式化 Python 源代码并执行基本的代码预检查。

```bash
pip install pre-commit
pre-commit install
pre-commit
```

### 2.2 算子单元测试
Op Unit Test 检查算子的正确性，如果新增算子，需要在 `tests` 目录的相应文件下增加测试用例；如果新增了测试文件，则需要在 `tools/coverage.sh` 文件中的 `cmd` 变量新增测试命令。
对于算子类的单元测试，请在测试函数前装饰 @pytest.mark.{OP_NAME}，这样可以通过 `pytest -m` 选择运行指定 OP 的单元测试函数。一个单元测试函数可装饰多个自定义 mark。

### 2.3 模型测试
Model Test 检查模型的正确性，新增模型的流程与新增算子的流程类似。

### 2.4 代码覆盖率检查
Python Coverage 检查当前 PR 新增代码的覆盖率, 其运行依赖`算子单元测试`和`模型测试`皆运行成功，否则会被跳过执行，代码覆盖率表示为：

>    PR 新增代码且测试能够覆盖的行数 / PR 总新增代码行数

注意：因为 Triton JIT 函数体不会运行，计算覆盖率时将会被去除。

代码合入要求 覆盖率达到 __90%__ 以上，代码覆盖率的详细信息可以在 log 中查看。

本地复现需要安装 `lcov`, `coverage` 以及 `PyGithub` 等工具。

```bash
cd $FlagGemsROOT
PR_ID=your_pr_id
bash tools/op-unit-test.sh
bash tools/model-test.sh
tools/code_coverage/coverage.sh PR_ID
```

当前流水线尚未对算子的性能进行检查，可以在 `benchmark` 目录下撰写性能测试，查看自己的优化效果。


### 2.5 算子性能测试

`Op Benchmark` 用于评估算子的性能。如果新增了算子，需要在 `benchmark` 目录下的相应文件中添加对应的测试用例。建议按照以下步骤完成算子用例的添加：

1. **选择合适的测试文件**
   根据算子的类别，选择 `benchmark` 目录下对应的文件：
   - 对于 reduction 类算子，可以添加到 `test_reduction_perf.py` 文件。
   - 对于 tensor constructor 类算子，可以添加到 `test_tensor_constructor_perf.py` 文件。
   - 如果算子难以归类，可以放到 `test_special_perf.py` 文件，或者创建一个新文件来表示新的算子类别。

2. **检查现有测试类**
   确认所需添加的文件后，查看该文件下已有的继承了 `Benchmark` 结构的各类（Class）。检查是否有适合你算子的测试场景，主要考虑以下两点：
   - **Metric 采集是否合适**。
   - **输入构造函数（`input_generator` 或 `input_fn`）是否合适**。

3.    **添加测试用例**
   根据测试场景的需求，选择以下方式添加测试用例：

   3.1 **使用现有的 metric 和输入构造函数**
   如果现有的 metric 采集和输入构造函数满足算子的要求，可以按照文件内的代码组织形式，直接添加一行 `pytest.mark.parametrize`。例如，可以参考 `test_binary_pointwise_perf.py` 文件中的所有算子用例。

   3.2 **自定义输入构造函数**
   如果现有的 metric 采集符合要求，但输入构造函数不满足算子需求，可以实现自定义的 `input_generator`。具体可参考 `test_special_perf.py` 文件中的 `topk_input_fn` 函数，它是为 `topk` 算子编写的输入构造函数。

   3.3 **自定义 metric 和输入构造函数**
   如果现有的 metric 采集和输入构造函数都不满足需求，可以新建一个 `Class`，为该类设置算子特化的 metric 采集逻辑和输入构造函数。此类场景可以参考 `benchmark` 目录下各种 `Benchmark` 子类的写法。

## 3. 项目结构

```
FlagGems
├── src: 源码
│   └──flag_gems
│       ├──utils: 自动代码生成的工具
│       ├──ops: 单个算子
│       ├──fused: 融合算子
│       ├──testing: 测试工具
│       └──__init__.py
├── tests: 精度测试文件
├── benchmark: 性能测试文件
├── examples: 模型测试文件
├── LICENSE
├── README.md
├── README_cn.md
├── OperatorList.md
├── CONTRIBUTING.md
├── CONTRIBUTING_cn.md
├── pyproject.toml
└── pytest.ini
```

## 4. 许可证
FlagGems 使用 [Apache 许可证](https://github.com/FlagOpen/FlagGems/blob/master/LICENSE)
