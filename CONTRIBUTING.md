[中文版](./CONTRIBUTING_cn.md)

# FlagGems Contributor's Guide

Thank you for your interest in FlagGems! We use github to host code, manage issues and pull requests. Before contributing, please read the following guidelines.

## 1. Bug Report

Please report bugs using Github's issues. When reporting bugs, please provide

- a brief summary
- steps to reproduce
- and be specific!
- Some sample codes will be helpful too

## 2. Code Contribution

In pull requests, contributor should describe what changed and why. Please also provide test cases if applicable.
Pull requests require approvals from **one member** before merging. Additionally, they must pass continuous integration checks. More details see [Code Contribution](docs/code_countribution.md)

Currently, continuous integration checks include four pipelines:

## 3. Documentation Supplement

The documentation for FlagGems is stored in the `docs` directory and is currently built and deployed using [MkDocs](https://www.mkdocs.org/). The documentation undergoes nightly builds in the CI pipeline.

# Project Structure

```cpp
FlagGems
├── src                  // python source code
│   └──flag_gems
│       ├──csrc          // C source
│       ├──utils         // python utilities
│       ├──ops           // python single operators
│       ├──fused         // python fused operators
│       ├──modules       // python modules
│       ├──patches       // patching scripts
│       ├──testing       // python testing utility
├── tests                // python accuracy test files
├── benchmark            // python performance test files
├── examples             // examples
├── cmake                // c++ cmake files for C-extension
├── include              // c++ headers
├── lib                  // c++ source code for operator lib
├── ctest                // c++ testing files
├── triton_src           // triton jit functions src temporary
├── docs                 // docs for flag_gems
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── ...
```

# License

Any contributions you make will be under the [Apache License](https://github.com/FlagOpen/FlagGems/blob/master/LICENSE).
