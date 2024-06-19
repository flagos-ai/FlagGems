# FlagGems Contributor's Guide

Thank you for your interest in FlagGems! We use github to host code, manage issues and pull requests. Before contributing, please read the following guidelines.

## Bug Reports
Please report bugs using Github's issues. When reporting bugs, please provide

- a brief summary,
- steps to reproduce,
- and be specific!
- Some sample codes will be helpful too.

## Code Contribution
In pull requests, contributor should describe what changed and why. Please also provide test cases if applicable.
Pull requests require approvals from two members before merging.

Using pre-commit git hooks with FlagGems, you can format source Python code and perform basic code pre-checks when calling the git commit command

```bash
pip install pre-commit
pre-commit install
pre-commit
```

## Project Structure

```
FlagGems
├── src: source code for library
│   ├──flag_gems
│   │   ├──utils: utilities for automatic code generation
│   │   ├──ops: single operators
│   │   ├──fused: fused operators
│   │   ├──__init__.py
├── tests: accuracy test files
├── benchmark: performance test files
├── examples: model test files
├── LICENSE
├── README.md
├── README_cn.md
├── OperatorList.md
├── CONTRIBUTING.md
└── pyproject.toml
```

## License
Any contributions you make will be under the [Apache License](https://github.com/FlagOpen/FlagGems/blob/master/LICENSE).
