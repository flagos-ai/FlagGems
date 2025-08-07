![img_v3_02gp_8115f603-cc89-4e96-ae9d-f01b4fef796g](https://github.com/user-attachments/assets/97950fc6-62bb-4b6a-b8d5-5751c14492fa)

## About

FlagGems is a high-performance general operator library implemented in [OpenAI Triton](https://github.com/openai/triton). It aims to provide a suite of kernel functions to accelerate LLM training and inference.

By registering with the ATen backend of PyTorch, FlagGems facilitates a seamless transition, allowing users to switch to the Triton function library without the need to modify their model code. Users can still utilize the ATen backend as usual while experiencing significant performance enhancement. The Triton language offers benefits in readability, user-friendliness and performance comparable to CUDA. This convenience allows developers to engage in the development of FlagGems with minimal learning investment.

## Features

### Multi-Backend Hardware Support

FlagGems supports a wide range of hardware platforms and has been extensively tested across different hardware configurations.

### Automatic Codegen

FlagGems provides an automatic code generation mechanism that enables developers to easily generate both pointwise and fused operators.
The auto-generation system supports a variety of needs, including standard element-wise computations, non-tensor parameters, and specifying output types.
For more details, please refer to pointwise_dynamic(pointwise_dynamic.md).

### LibEntry

FlagGems introduces `LibEntry`, which independently manages the kernel cache and bypasses the runtime of `Autotuner`, `Heuristics`, and `JitFunction`. To use it, simply decorate the Triton kernel with LibEntry.

`LibEntry` also supports direct wrapping of `Autotuner`, `Heuristics`, and `JitFunction`, preserving full tuning functionality. However, it avoids nested runtime type invocations, eliminating redundant parameter processing. This means no need for binding or type wrapping, resulting in a simplified cache key format and reduced unnecessary key computation.

### C++ Runtime

FlagGems can be installed either as a pure Python package or as a package with C++ extensions. The C++ runtime is designed to address the overhead of the Python runtime and improve end-to-end performance.


## Changelog

### v1.0

- support BLAS operators: addmm, bmm, mm
- support pointwise operators: abs, add, div, dropout, exp, gelu, mul, pow, reciprocal, relu, rsqrt, silu, sub, triu
- support reduction operators: cumsum, layernorm, mean, softmax

### v2.0

- support BLAS operators: mv, outer
- support pointwise operators: bitwise_and, bitwise_not, bitwise_or, cos, clamp, eq, ge, gt, isinf, isnan, le, lt, ne, neg, or, sin, tanh, sigmoid
- support reduction operators: all, any, amax, argmax, max, min, prod, sum, var_mean, vector_norm, cross_entropy_loss, group_norm, log_softmax, rms_norm
- support fused operators: fused_add_rms_norm, skip_layer_norm, gelu_and_mul, silu_and_mul, apply_rotary_position_embedding

### v2.1

- support Tensor operators: where, arange, repeat, masked_fill, tile, unique, index_select, masked_select, ones, ones_like, zeros, zeros_like, full, full_like, flip, pad
- support neural network operator: embedding
- support basic math operators: allclose, isclose, isfinite, floor_divide, trunc_divide, maximum, minimum
- support distribution operators: normal, uniform\_, exponential\_, multinomial, nonzero, topk, rand, randn, rand_like, randn_like
- support science operators: erf, resolve_conj, resolve_neg

## Get Start

For a quick start with installing and using flag_gems, please refer to the documentation [Get Start](get_start_with_flaggems.md).

## Supported Operators

Operators will be implemented according to [OperatorList](operator_list.md).

## Supported Models

- Bert-base-uncased
- Llama-2-7b
- Llava-1.5-7b

## Supported Platforms

|  Platform  | float16 | float32 | bfloat16 |
| :--------: | :-----: | :-----: | :------: |
| Nvidia GPU |    ✓    |    ✓    |    ✓     |

## Performance

The following chart shows the speedup of FlagGems compared with PyTorch ATen library in eager mode. The speedup is calculated by averaging the speedup on each shape, representing the overall performance of the operator.

![Operator Speedup](assets/speedup-20250423.png)

## Contributions

If you are interested in contributing to the FlagGems project, please refer to [Countributing Guide](code_countribution.md). Any contributions would be highly appreciated.

## Contact us

If you have any questions about our project, please submit an issue, or contact us through <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

We also created WeChat group for FlagGems. Scan the QR code to join the group chat! To get the first hand message about our updates and new release, or having any questions or ideas, join us now!

<p align="center">
 <img src="https://github.com/user-attachments/assets/69019a23-0550-44b1-ac42-e73f06cb55d6" alt="bge_wechat_group" class="center" width="200">
</p>
## License

The FlagGems project is based on [Apache 2.0](https://github.com/FlagOpen/FlagGems/blob/master/LICENSE).
