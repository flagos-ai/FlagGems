[中文版](https://github.com/FlagOpen/FlagGems/blob/master/README_cn.md)

## Introduction

FlagGems is a high-performance general operator library implemented in [OpenAI Triton](https://github.com/openai/triton). It aims to provide a suite of kernel functions to accelerate LLM training and inference.  

By registering with the ATen backend of PyTorch, FlagGems facilitates a seamless transition, allowing users to switch to the Triton function library without the need to modify their model code. Users can still utilize the ATen backend as usual while experiencing significant performance enhancement. The Triton language offers benefits in readability, user-friendliness and performance comparable to CUDA. This convenience allows developers to engage in the development of FlagGems with minimal learning investment.  


## Changelog

### v1.0
- released in April 2024  
- support BLAS operators: addmm, bmm, mm  
- support pointwise operators: abs, add, div, dropout, exp, gelu, mul, pow, reciprocal, relu, rsqrt, silu, sub, triu  
- support reduction operators: cumsum, layernorm, mean, softmax  

## Quick Start

### Requirements

1. Triton >= 2.2.0  
2. PyTorch >= 2.1.2  
3. Transformers >= 4.31.0  

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

1. Run Tests  
    - Operator Accuracy  
        ```shell
        cd tests/flag_gems
        pytest op_accu_test.py
        ```
    - Model Accuracy  
        ```shell
        cd tests/flag_gems
        pytest model_bert_test.py
        ```
    - Operator Performance  
        ```shell
        cd tests/flag_gems
        python -O op_perf_test.py
        ```

2. Run without printing Flag infomation  
    ```shell
    python -O program.py
    ```

## Supported Operators

Operators will be implemented according to [OperatorList.md](https://github.com/FlagOpen/FlagGems/blob/master/OperatorList.md).

## Supported Models

| Model | float16 | float32 | bfloat16 |
| :---: | :---: | :---: | :---: |
| Bert_base | ✓ | ✓ | ✓ |

## Supported Platforms

| Platform | float16 | float32 | bfloat16 |
| :---: | :---: | :---: | :---: |
| Nvidia A100 | ✓ | ✓ | ✓ |

## Contributions

If you are interested in contributing to the FlagGems project, please refer to [Contributing.md](https://github.com/FlagOpen/FlagGems/blob/master/Contributing.md). Any contributions would be highly appreciated.

## Contact us

If you have any questions about our project, please submit an issue, or contact us through <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

## License

The FlagGems project is based on [Apache 2.0](https://github.com/FlagOpen/FlagGems/blob/master/LICENSE).
