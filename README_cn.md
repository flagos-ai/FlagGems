[English](./README.md)

## 介绍

FlagGems 是一个使用 OpenAI 推出的[Triton 编程语言](https://github.com/openai/triton)实现的高性能通用算子库，
旨在为大语言模型提供一系列可应用于 PyTorch 框架的算子，加速模型面向多种后端平台的推理与训练。

FlagGems 通过对 PyTorch 的后端 ATen 算子进行覆盖重写，实现算子库的无缝替换，
一方面使得模型开发者能够在无需修改底层 API 的情况下平稳地切换到 Triton 算子库，
使用其熟悉的 PyTorch API 同时享受新硬件带来的加速能力，
另一方面对 kernel 开发者而言，Triton 语言提供了更好的可读性和易用性，可媲美 CUDA 的性能，
因此开发者只需付出较低的学习成本，即可参与 FlagGems 的算子开发与算子库建设。

## 特性

- 支持的算子数量规模较大
- 部分算子已经过深度性能调优
- 可直接在 Eager 模式下使用, 无需通过 `torch.compile`
- Pointwise 自动代码生成，灵活支持多种输入类型和内存排布
- Triton kernel 调用优化
- 灵活的多后端支持机制
- 代码库已集成十余种后端
- C++ Triton 函数派发 (开发中)

## 更多特性细节

### 多后端硬件支持

FlagGems 支持更多的硬件平台并且在不同硬件上进行了充分的测试。

### 自动代码生成

FlagGems 提供了一套自动代码生成的机制，开发者可以使用它来便捷地生成 pointwise 类型的单算子与融合算子。自动代码生成可以处理常规的对位计算、非张量参数、指定输出类型等多种需求。详细信息参考 [pointwise_dynamic](docs/pointwise_dynamic.md)

### LibEntry

FlagGems 构造了 `LibEntry` 独立维护 kernel cache, 绕过 `Autotuner`、`Heuristics` 和 `JitFunction` 的 runtime, 使用时仅需在 triton kernel 前装饰即可。`LibEntry` 支持 `Autotuner`、`Heuristics`、`JitFunction` 的直接包装，不影响调参功能的正常使用，但是无需经过运行时类型的嵌套调用，节省了重复的参数处理，无需绑定和类型包装，简化了 cache key 格式，减少不必要的键值计算。

### C++ 运行时

FlagGems 可以作为纯 Python 包安装，也可以作为带有 C++ 扩展的包安装。C++ 运行时旨在解决 python 运行时开销昂贵的问题, 提高整个端到端的性能。

## 更新日志

### v4.2（即将发布）

- 计划支持 216 个算子，并与最新的 [Operator List](./docs/operators.md) 保持一致
- 计划新增算子：`tan`、`tan_`、`baddbmm`、`avg_pool2d`、`clamp_min`、`clamp_min_`、`std`、`trace`、`max_pool2d`、`bitwise_left_shift`、`bitwise_right_shift`
- 原有的 `upsample` 算子将拆分为 `upsample_nearest2d` 与 `upsample_bicubic2d_aa`

### v4.1

- 面向 RWKV 模型的定制版本，共计支持 204 个算子
- 包含针对 RWKV 推理加速场景优化的融合算子：`rwkv_mm_sparsity`、`rwkv_ka_fusion`
- 已被 RWKV 项目采用，具体见 [BlinkDL/Albatross:faster_251101](https://github.com/BlinkDL/Albatross/tree/main/faster_251101)

### v4.0

- 共计支持 202 个算子
- 新增通用算子：`addcdiv`、`addcmul`、`addmv`、`addr`、`atan`、`atan_`、`celu`、`celu_`、`elu_`、`exp2`、`exp2_`、`get_scheduler_metadata`、`index_add_`、`logspace`、`moe_align_block_size`、`softplus`、`sqrt_`、`topk_softmax`
- Triton JIT runtime 新增支持的算子：`add`、`addmm`、`argmax`、`bmm`、`cat`、`contiguous`、`embedding`、`exponential_`、`fill`、`flash_attn_varlen_func`、`fused_add_rms_norm`、`max`、`mm`、`nonzero`、`reshape_and_cache_flash`、`rms_norm`、`rotary_embedding`、`softmax`、`sum`、`topk`、`zeros`

### v3.0

- 共计支持 184 个算子，包括大模型推理使用的定制算子
- 支持更多的硬件平台，新增 Ascend、AIPU 等
- 兼容 vllm 框架，DeepSeek 模型推理验证通过

### v2.1

- 支持 Tensor 类算子：where, arange, repeat, masked_fill, tile, unique, index_select, masked_select, ones, ones_like, zeros, zeros_like, full, full_like, flip, pad
- 支持神经网络类算子：embedding
- 支持基础数学算子：allclose, isclose, isfinite, floor_divide, trunc_divide, maximum, minimum
- 支持分布类算子：normal, uniform\_, exponential\_, multinomial, nonzero, topk, rand, randn, rand_like, randn_like
- 支持科学计算算子：erf, resolve_conj, resolve_neg

### v2.0

- 支持 BLAS 类算子: mv, outer
- 支持 pointwise 类算子: bitwise_and, bitwise_not, bitwise_or, cos, clamp, eq, ge, gt, isinf, isnan, le, lt, ne, neg, or, sin, tanh, sigmoid
- 支持 reduction 类算子: all, any, amax, argmax, max, min, prod, sum, var_mean, vector_norm, cross_entropy_loss, group_norm, log_softmax, rms_norm
- 支持融合算子: fused_add_rms_norm, skip_layer_norm, gelu_and_mul, silu_and_mul, apply_rotary_position_embedding

### v1.0

- 支持 BLAS 类算子：addmm, bmm, mm
- 支持 pointwise 类算子：abs, add, div, dropout, exp, gelu, mul, pow, reciprocal, relu, rsqrt, silu, sub, triu
- 支持 reduction 类算子：cumsum, layernorm, mean, softmax

## 快速入门

参考文档 [开始使用](./docs/getting-started.md) 快速安装使用 FlagGems。

## 支持算子

算子将按照文档 [OperatorList](./docs/operators.md) 的顺序逐步实现。

## 支持模型

- Bert-base-uncased
- Llama-2-7b
- Llava-1.5-7b

## 支持平台

| vendor     | state                  | float16 | float32 | bfloat16 |
| ---------- | ---------------------- | ------- | ------- | -------- |
| aipu       | ✅ （Partial support） | ✅      | ✅      | ✅       |
| ascend     | ✅ （Partial support） | ✅      | ✅      | ✅       |
| cambricon  | ✅                     | ✅      | ✅      | ✅       |
| hygon      | ✅                     | ✅      | ✅      | ✅       |
| iluvatar   | ✅                     | ✅      | ✅      | ✅       |
| kunlunxin  | ✅                     | ✅      | ✅      | ✅       |
| metax      | ✅                     | ✅      | ✅      | ✅       |
| mthreads   | ✅                     | ✅      | ✅      | ✅       |
| nvidia     | ✅                     | ✅      | ✅      | ✅       |
| arm(cpu)   | 🚧                     |         |         |          |
| tsingmicro | 🚧                     |         |         |          |

## 性能表现

FlagGems 相比 Torch Eager 模式下 ATen 算子库的加速比如下图所示。其中，每个算子的加速比综合了多个形状测例的数据，代表该算子的整体性能。

![算子加速比](./docs/assets/speedup-20250423.png)

## 贡献代码

欢迎大家参与 FlagGems 的算子开发并贡献代码，详情请参考[CONTRIBUTING.md](./CONTRIBUTING_cn.md)。

## 引用

欢迎引用我们的项目：

```bibtex
@misc{flaggems2024,
    title={FlagOpen/FlagGems: FlagGems is an operator library for large language models implemented in the Triton language.},
    url={https://github.com/FlagOpen/FlagGems},
    journal={GitHub},
    author={BAAI FlagOpen team},
    year={2024}
}
```

## 联系我们

如有疑问，请提交 issue，或发送邮件至<a href="mailto:contact@flagos.io">contact@flagos.io</a>。
我们为 FlagGems 创建了微信群。扫描二维码即可加入群聊！第一时间了解我们的动态和信息和新版本发布，或者有任何问题或想法，请立即加入我们！

<img width="204" height="180" alt="开源小助手" src="https://github.com/user-attachments/assets/4e9a8566-c91e-4120-a011-6b5577c1a53d" />

## 许可证

本项目采用 [Apache License (version 2.0)](./LICENSE) 授权许可。
