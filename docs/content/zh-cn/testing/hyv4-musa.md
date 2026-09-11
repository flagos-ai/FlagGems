---
title: HyV4 MUSA 算子验证
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

# HyV4 MUSA 算子验证

本文定义 FlagGems 实现被认定为覆盖 HyV4 MUSA 推理路径前所需的证据。模型侧
blacklist 或 fallback 只是观察结果，不能单独证明对应 FlagGems 算子存在错误。

## 算子族

HyV4 涉及以下高优先级算子族：

| 算子族 | 代表性输入契约 | 验证重点 |
|---|---|---|
| MXFP8 MoE | hidden 6144、256 experts、top-k 8、intermediate 2048、group size 128 | scale 语义、量化、grouped GEMM、路由和累加 dtype |
| DSA prefill | paged KV、index top-k 2048 | 稀疏注意力精度和 page-table transform |
| DSA KV gather/dequant | paged FP8 KV、selected indices `[T, 2048]`、BF16 输出 | 索引边界、scale layout 和输出精度 |
| DSA top-k | logits `[T, S]`、k=2048、int64 indices | values、indices 和相同值时的确定性 |
| mHC post | input `[T, 6144]`、residual `[T, 4, 6144]`、post `[T, 4]` | 广播顺序和 FP32 累加 |
| clamped SwiGLU | gate/up `[T, 4096]`、output `[T, 2048]` | clamp、sigmoid 和输出 dtype |
| non-contiguous BMM | 从模型提取的真实 shape 和 stride | 不强制 contiguous copy 时的 stride 正确性 |
| repeat/index/copy/reduction | 动态 token、expert 和 page metadata | dispatch owner、动态 shape 和整数边界 |

精确的 token 维 `T`、stride、dtype 和 scale tensor 必须从有效模型运行时提取。
上述代表性维度不能被扩展成所有动态 shape 均已支持的结论。

## Issue 和 PR 的必要证据

每个可独立修复的算子问题都应包含：

1. FlagGems、编译器、PyTorch 和设备栈的完整 commit；
2. 保存的真实 shape 输入，或带 checksum 的确定性生成器；
3. 实际 dispatch owner 和 kernel 名；
4. expected/actual，以及最大和平均误差；
5. eager 与编译路径覆盖；
6. 修复前失败、修复后通过的测试；
7. 单算子时延和模型端到端 A/B；
8. 不受影响后端的回归测试；
9. 回滚说明。

如果实际 owner 是 PyTorch-MUSA、编译器后端、SGLang 或通信库，应将问题提交到
对应项目，而不是在 FlagGems 中增加 workaround。

## 安全的选择性启用

模型集成使用选择性算子启用时，应保留运行时 dispatch 记录。记录为空必须明确
说明；这通常意味着已有 PrivateUse1 实现持有该调用，FlagGems 跳过了重复注册。

以下任一情况都不能单独作为 FlagGems 已覆盖模型的证据：

- 源码树中存在同名算子；
- 算子名称出现在 allowlist 或 blacklist；
- 服务返回 HTTP 200；
- 模型 benchmark 通过，但 FlagGems dispatch 记录为空。

## 端到端回归门禁

算子验证完成后，使用相同模型 fingerprint 重新执行：

- 有界确定性生成；
- 重复和并发请求；
- 已选择的精度任务；
- profiler 关闭的性能测试；
- worker、scheduler、OOM、NaN 和设备错误检查。

任何算子、精度、dispatch、graph 或 cache 变化都会使旧模型证据失效，必须重新
通过上述门禁。

