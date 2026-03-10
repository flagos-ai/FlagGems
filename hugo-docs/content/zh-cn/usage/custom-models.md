---
title: 构造定制的模型
weight: 80
---

# Building Custom Models Using Gems Operators

In some scenarios, users may wish to build their own models from scratch
or to adapt existing ones to better suit their specific use cases.
To support this, *FlagGems* provides a growing collection of high-performance modules
commonly used in large language models (LLMs).

These components are implemented using *FlagGems*-accelerated operators
and can be used in the way you use any standard `torch.nn.Module`.
You can seamlessly integrate them into your architecture to benefit from kernel-level acceleration
without writing custom CUDA or Triton code.

Modules can be found in
[flag_gems/modules](https://github.com/flagos-ai/FlagGems/tree/master/src/flag_gems/modules).

## Modules Available

| Module                 | Description                                           | Supported Features                         |
| ---------------------- | ----------------------------------------------------- | ------------------------------------------ |
| `GemsRMSNorm`          | RMS LayerNorm                                         | Fused residual add, `inplace` & `outplace` |
| `GemsRope`             | Standard rotary position embedding                    | `inplace` & `outplace`                     |
| `GemsDeepseekYarnRoPE` | RoPE with extrapolation for DeepSeek-style LLMs       | `inplace` & `outplace`                     |
| `GemsSiluAndMul`       | Fused SiLU activation with elementwise multiplication | `outplace` only                            |

We encourage users to use these as drop-in replacements for equivalent PyTorch layers.
More components such as fused attention, MoE layers, and transformer blocks are under development.
