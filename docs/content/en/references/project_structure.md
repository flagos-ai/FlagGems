---
title: Project Structure
weight: 40
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->


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
