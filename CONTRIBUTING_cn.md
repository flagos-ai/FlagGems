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
