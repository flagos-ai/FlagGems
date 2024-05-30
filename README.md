[中文版](https://github.com/FlagOpen/FlagGems/blob/master/README_cn.md)

## Introduction

FlagGems is a high-performance general operator library implemented in [OpenAI Triton](https://github.com/openai/triton). It aims to provide a suite of kernel functions to accelerate LLM training and inference.  

By registering with the ATen backend of PyTorch, FlagGems facilitates a seamless transition, allowing users to switch to the Triton function library without the need to modify their model code. Users can still utilize the ATen backend as usual while experiencing significant performance enhancement. The Triton language offers benefits in readability, user-friendliness and performance comparable to CUDA. This convenience allows developers to engage in the development of FlagGems with minimal learning investment.  


## Changelog

### v1.0
- support BLAS operators: addmm, bmm, mm  
- support pointwise operators: abs, add, div, dropout, exp, gelu, mul, pow, reciprocal, relu, rsqrt, silu, sub, triu  
- support reduction operators: cumsum, layernorm, mean, softmax  

### v2.0
- support BLAS operator: mv, outer  
- support pointwise operators: bitwise_and, bitwise_not, bitwise_or, cos, clamp, eq, ge, gt, isinf, isnan, le, lt, ne, neg, or, sin, tanh, sigmoid  
- support reduction operators: all, any, amax, argmax, max, min, prod, sum, var_mean, vector_norm, cross_entropy_loss, group_norm, log_softmax, rms_norm  
- support fused operators: skip_rms_norm, skip_layer_norm, gelu_and_mul, silu_and_mul, apply_rotary_position_embedding  

## Quick Start

### Requirements

1. Triton >= 2.2.0  
2. PyTorch >= 2.1.2  
3. Transformers >= 4.40.2  

### Installation  

```shell
git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems
pip install .
```

## Usage  

### Import

1. Enable permanently  
    ```python
    import flag_gems
    flag_gems.enable()
    ```

2. Enable temporarily  
    ```python
    import flag_gems
    with flag_gems.use_gems():
        pass
    ```

3. Example  
    ```python
    import torch
    import flag_gems

    M, N, K = 1024, 1024, 1024
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    B = torch.randn((K, N), dtype=torch.float16, device="cuda")
    with flag_gems.use_gems():
        C = torch.mm(A, B)
    ```

### Execute

1. Test Operator Accuracy  
    - Run reference on cuda  
        ```shell
        cd tests
        pytest test_xx_ops.py
        ```
    - Run reference on cpu  
        ```shell
        cd tests
        pytest test_xx_ops.py --device cpu
        ```

2. Test Model Accuracy  
    ```shell
    cd examples
    pytest model_xx_test.py
    ```

3. Test Operator Performance  
    - Test CUDA performance  
        ```shell
        cd benchmark
        pytest test_xx_perf.py -s
        ```
    - Test end-to-end performance  
        ```shell
        cd benchmark
        pytest test_xx_perf.py -s --mode cpu
        ```

4. Run tests with logging infomation  
    ```shell
    pytest program.py --log-cli-level debug
    ```
    Not recommended in performance testing.  

## Supported Operators

Operators will be implemented according to [OperatorList.md](https://github.com/FlagOpen/FlagGems/blob/master/OperatorList.md).

## Supported Models

- Bert-base-uncased  
- Llama-2-7b  

## Supported Platforms

| Platform | float16 | float32 | bfloat16 |
| :---: | :---: | :---: | :---: |
| Nvidia A100 | ✓ | ✓ | ✓ |

## Contributions

If you are interested in contributing to the FlagGems project, please refer to [CONTRIBUTING.md](https://github.com/FlagOpen/FlagGems/blob/master/CONTRIBUTING.md). Any contributions would be highly appreciated.

## Contact us

If you have any questions about our project, please submit an issue, or contact us through <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

## License

The FlagGems project is based on [Apache 2.0](https://github.com/FlagOpen/FlagGems/blob/master/LICENSE).
