---
title: 调试支持
weight: 45
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
# Enable Debug Logging
-->
# 调试与调用日志追踪

为了方便算子开发、执行路径排查以及性能 Profiling 分析，*FlagGems* 在 `enable()` 和 `only_enable()` 接口中提供了算子调用追踪参数：

| 参数名称 | 数据类型 | 描述 |
| :------- | :------- | :--- |
| `record` | `bool`   | 是否开启算子调用日志追踪，用于调试与性能排查 |
| `path`   | `str`    | 日志文件输出路径（仅在 `record=True` 时生效） |

若希望在模型运行期间记录算子的实际命中与调用情况，可在启用时指定 `record=True` 及目标日志路径：

```python
import flag_gems

flag_gems.enable(
    record=True,
    path="./gems_debug.log",
)
```

程序执行完毕后，可通过查看生成的日志文件（如 `./gems_debug.log`），确认哪些算子成功通过 *FlagGems* 进行了加速分发：

日志输出示例如下：

```shell
$ cat ./gems_debug.log
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.mm: GEMS MM
[DEBUG] flag_gems.fused.reshape_and_cache: GEMS RESHAPE_AND_CACHE
```

> [!WARNING]
> **性能损耗提醒**
>
> 开启调用日志会在算子关键路径上引入同步磁盘 I/O 操作，会对工作负载的端到端吞吐产生一定影响。尤其在小算子高频调用的场景下，额外开销更为明显。因此建议仅在排查问题与验证算子分发时开启，生产环境请保持关闭。
