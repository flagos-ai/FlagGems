[中文版](./CONTRIBUTING_cn.md)

# FlagGems Contributor's Guide

Thank you for your interest in FlagGems! We use github to host code, manage issues and pull requests. Before contributing, please read the following guidelines.

## 1. Bug Reports
Please report bugs using Github's issues. When reporting bugs, please provide

- a brief summary,
- steps to reproduce,
- and be specific!
- Some sample codes will be helpful too.

## 2. Code Contribution
In pull requests, contributor should describe what changed and why. Please also provide test cases if applicable.
Pull requests require approvals from __one members__ before merging. Additionally, they must pass continuous integration checks.

Currently, continuous integration checks include four pipelines:

### 2.1 Code Format Check
Using pre-commit git hooks with FlagGems, you can format source Python code and perform basic code pre-checks when calling the git commit command

```bash
pip install pre-commit
pre-commit install
pre-commit
```

### 2.2 Op Unit Test
Operator Unit Tests check the correctness of operators. If new operators are added, you need to add test cases in the corresponding file under the `tests` directory. If new test files are added, you should also add the test commands to the `cmd` variable in the `tools/coverage.sh` file.
For operator testing, decorate @pytest.mark.{OP_NAME} before the test function so that we can run the unit test function of the specified OP through `pytest -m`. A unit test function can be decorated with multiple custom marks.

### 2.3 Model Test
Model Tests check the correctness of models. Adding a new model follows a process similar to adding a new operator.

### 2.4 Python Coverage
Python Coverage checks the coverage of the new code added in the current PR. It depends on the successful execution of both `Op Unit Test` and `Model Tests`; otherwise, it will be skipped. Code coverage is represented as:

>    Lines added in the PR and covered by the tests  / Total lines of new code in the PR

Note: Since Triton JIT functions do not actually run, they will be excluded from the coverage calculation.

The code merging requirement is a coverage rate of __90%__ or above. Detailed information about code coverage can be viewed in the log.

To reproduce locally, you need to install tools like `lcov`, `coverage` and `PyGithub`.

```bash
cd $FlagGemsROOT
PR_ID=your_pr_id
bash tools/op-unit-test.sh
bash tools/model-test.sh
tools/code_coverage/coverage.sh PR_ID
```

Currently, the pipeline does not check the performance of operators. You can write performance tests in the `benchmark` directory to evaluate your optimization results.

## 3. Project Structure

```
FlagGems
├── src: source code for library
│   └──flag_gems
│       ├──utils: utilities for automatic code generation
│       ├──ops: single operators
│       ├──fused: fused operators
│       ├──testing: testing utility
│       └──__init__.py
├── tests: accuracy test files
├── benchmark: performance test files
├── examples: model test files
├── LICENSE
├── README.md
├── README_cn.md
├── OperatorList.md
├── CONTRIBUTING.md
├── CONTRIBUTING_cn.md
├── pyproject.toml
└── pytest.ini
```

## 4. License
Any contributions you make will be under the [Apache License](https://github.com/FlagOpen/FlagGems/blob/master/LICENSE).
