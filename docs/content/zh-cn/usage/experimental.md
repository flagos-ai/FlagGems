---
title: 使用实验性算子
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

# 使用实验性算子

`flag_gems.experimental_ops` 模块为社区孵化中的前沿算子提供了独立的命名空间。这些算子通常处于精度对齐、性能极限压测或 API 设计收敛阶段，尚未正式合入稳定版发布中。

虽然处于实验阶段，但这些算子均严格遵循与核心算子一致的代码规范与接口设计。开发者可按如下方式显式调用以尝鲜体验：

## 1. 直接显式导入调用

```python
from flag_gems import experimental_ops as exp_ops

# 调用实验性 RMSNorm 算子
result = exp_ops.rmsnorm(*args)
```

## 2. 在加速上下文中调用

在 `with flag_gems.use_gems():` 上下文中，亦可通过模块全路径直接调用实验性算子：

```python
import flag_gems

with flag_gems.use_gems():
    result = flag_gems.experimental_ops.rmsnorm(*args)
```

当前所有处于实验阶段的算子列表可参阅[实验性算子清单](../references/experimental.md)。
