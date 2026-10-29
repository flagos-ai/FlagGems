---
title: 在非英伟达（NVIDIA）硬件上使用
weight: 50
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

# 在非英伟达（NVIDIA）硬件上使用 FlagGems

*FlagGems* 秉承“一次编写，跨端运行”的设计哲学，不仅在 NVIDIA GPU 上表现出色，更对国内外主流异构 AI 加速芯片提供了深度适配。

## 1. 支持的硬件平台

*FlagGems* 现已支持包括昇腾（Ascend）、摩尔线程（Mthreads）、海光（Hygon）、沐曦（MetaX）、天数智芯（Iluvatar）、昆仑芯（Kunlunxin）等在内的 10 余种加速芯片后端。各厂商的详细支持程度及数据类型支持清单请参阅[硬件平台支持](../overview/platforms.md)。

## 2. 统一简洁的使用体验

无论底层是何种加速硬件，*FlagGems* 面向上层应用提供**完全一致的 Python API 接口**。从 NVIDIA 迁移至其他芯片架构时，应用层模型代码无需做出任何修改：

```python
import flag_gems

# 统一的全局启用入口
flag_gems.enable()
```

一旦启用，*FlagGems* 的内部调度机制会自动识别物理硬件后端，并将算子透明路由到匹配的底层实现，为多芯片异构集群带来无缝一致的开发与部署体验。

## 3. 运行依赖与环境准备

在非 NVIDIA 平台上运行 *FlagGems* 时，需确保目标系统已就绪以下基础组件：

1. **硬件厂商定制的 PyTorch 与驱动**：通常由芯片厂商提供专有的 PyTorch 插件或编译版本（如 `torch_npu`、`torch_musa` 等），确保基础 Tensor 运算与设备内存交互正常。
2. **Triton / FlagTree 编译器**：
   - 方式一：使用厂商官方提供的适配分支；
   - 方式二（推荐）：采用 [FlagTree](https://github.com/flagos-ai/FlagTree) 统一多后端编译器。FlagTree 将各厂商的后端代码生成与优化补丁整合进统一的开源编译器框架中，大幅简化了编译链维护与多端兼容工作。

## 4. 后端自动识别与手动覆盖

默认情况下，*FlagGems* 会在导入时自动嗅探当前环境的物理加速卡类型，并绑定到对应的后端。

若在复杂的容器化、异构混部或开发测试环境中自动探测未生效，可通过环境变量显式指定目标厂商标识：

```shell
export GEMS_VENDOR=<vendor_name>
```

合法的 `vendor_name` 取值（如 `ascend`、`hygon`、`metax`、`mthreads` 等）请查阅[硬件平台支持](../overview/platforms.md)。

在 Python 运行时中，可通过 `flag_gems.vendor_name` 确认当前生效的后端名称：

```python
import flag_gems

print("Active backend:", flag_gems.vendor_name)
```
