---
title: 硬件平台支持
weight: 20
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


<!--
# Platforms Supported
-->
# 支持的硬件平台

*FlagGems* 已支持 10 余种不同的加速硬件后端。目前正式支持及开发中的硬件平台及其数据类型支持状态汇总如下：

| 厂商 / 架构 | 后端标识符 (`backend`) | 支持状态 | float16 | float32 | bfloat16 |
| :--------- | :--------------------- | :------: | :-----: | :-----: | :------: |
| AIPU       | `aipu`                 | ☑️       | ✅      | ✅      | ✅       |
| AMD        | `amd`                  | 🚧       | -       | -       | -        |
| ARM (CPU)  | `arm`                  | ☑️       | -       | -       | -        |
| Ascend     | `ascend`               | ✅       | ✅      | ✅      | ✅       |
| Cambricon  | `cambricon`            | ✅       | ✅      | ✅      | ✅       |
| Hygon      | `hygon`                | ✅       | ✅      | ✅      | ✅       |
| Iluvatar   | `iluvatar`             | ✅       | ✅      | ✅      | ✅       |
| Kunlunxin  | `kunlunxin`            | ✅       | ✅      | ✅      | ✅       |
| MetaX      | `metax`                | ✅       | ✅      | ✅      | ✅       |
| Mthreads   | `mthreads`             | ✅       | ✅      | ✅      | ✅       |
| NVIDIA     | `nvidia`               | ✅       | ✅      | ✅      | ✅       |
| Sunrise    | `sunrise`              | ☑️       | ✅      | ✅      | ✅       |
| TsingMicro | `tsingmicro`           | 🚧       | -       | -       | -        |

**状态说明**：

- ✅ - 全面支持（Fully Supported）
- ☑️ - 部分支持（Partially Supported）
- 🚧 - 正在开发适配中（Under Development）
