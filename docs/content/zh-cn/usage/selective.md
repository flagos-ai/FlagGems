---
title: 选择性启用算子
weight: 30
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

# 选择性启用算子

在集成 *FlagGems* 时，您可以通过精细化的过滤参数灵活控制加速算子的生效范围。这一能力在性能对照基准测试、逐算子精度排查或复杂多模块工作流中尤为实用。

目前 *FlagGems* 提供了三种方式实现算子的选择性启用或排除：

## 1. 全局排除特定算子：`enable(unused=[...])`

API 接口 `flag_gems.enable()` 提供了 `unused` 参数，用于配置算子黑名单：

| 参数名称 | 数据类型 | 描述 |
| :------- | :------- | :--- |
| `unused` | `List[str]` | 指定不启用的算子名称列表（黑名单） |

当您发现个别算子在特定硬件架构或极端 Shape 下收益不及预期时，可将其加入排除列表。此时未被排除的算子均由 *FlagGems* 加速，而列表中的算子将自动回退（Fallback）至 PyTorch 原生实现：

```python
import flag_gems

# 启用除 sum 和 add 之外的所有 FlagGems 算子
flag_gems.enable(unused=["sum", "add"])
```

## 2. 仅启用特定算子：`only_enable(include=[...])`

若只需对指定的几个热点算子开启加速，可以使用 `only_enable()` 接口：

| 参数名称 | 数据类型 | 描述 |
| :------- | :------- | :--- |
| `include` | `List[str]` | 显式指定需要启用的算子白名单 |

此时仅有列表中指定的算子会被 *FlagGems* 接管，其余所有算子继续保持 PyTorch 原生实现：

```python
import flag_gems

# 仅接管 rms_norm 和 softmax 算子
flag_gems.only_enable(include=["rms_norm", "softmax"])
```

> [!WARNING]
> `only_enable()` 属于实验性接口，未来版本可能会与 `enable()` 进一步合并优化。

## 3. 上下文管理器范围过滤：`use_gems(include=..., exclude=...)`

在局部代码块中，使用 `flag_gems.use_gems()` 上下文管理器同样支持包含与排除控制：

| 参数名称 | 数据类型 | 描述 |
| :------- | :------- | :--- |
| `include` | `List[str]` | 作用域内显式启用的算子白名单 |
| `exclude` | `List[str]` | 作用域内需要排除的算子黑名单 |

示例 1：仅在上下文中启用 `sum` 与 `add` 算子：

```python
import flag_gems

with flag_gems.use_gems(include=["sum", "add"]):
    # 代码块中仅有 sum 与 add 会调用 FlagGems 加速版本
    ...
```

示例 2：在上下文中启用除 `mul` 与 `div` 之外的所有算子：

```python
import flag_gems

with flag_gems.use_gems(exclude=["mul", "div"]):
    # 除 mul 和 div 之外的所有兼容算子均获得加速
    ...
```

> [!TIP]
> `include` 参数的优先级高于 `exclude`。当同时传入两者时，系统将优先以 `include` 白名单为准，`exclude` 将被忽略。
