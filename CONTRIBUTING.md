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

## Project Structure

```
FlagGems
├── src: source code for library
│   ├──flag_gems
│   │   ├──utils: utilities for automatic code generation
│   │   ├──ops: single operators
│   │   ├──fused: fused operators
│   │   ├──__init__.py
├── tests
│   ├──flag_gems
│   │   ├──model_bert_test.py: test for BERT model running with flag_gems
│   │   ├──op_accu_test.py: test for accuracy of operators
│   │   ├──op_perf_test.py: test for performance of operators
├── LICENSE
├── README.md
├── README_cn.md
├── OperatorList.md
├── CONTRIBUTING.md
└── pyproject.toml
```

## License
Any contributions you make will be under the [Apache License](https://github.com/FlagOpen/FlagGems/blob/master/LICENSE).
