---
title: Project Structure
weight: 40
---

# Project Structure

```none
FlagGems
├── src                  // python source code
│   └──flag_gems
│       ├──utils         // python automatic code generation utilities
│       ├──ops           // python single operators
│       ├──fused         // python fused operators
│       `──testing       // python testing utility
├── tests                // python accuracy test files
├── benchmark            // python performance test files
├── examples             // python model test files
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
