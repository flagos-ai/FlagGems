[English](https://github.com/FlagOpen/FlagGems/blob/master/README.md)

## 介绍

FlagGems是一个使用OpenAI推出的[Triton编程语言](https://github.com/openai/triton)实现的高性能通用算子库，旨在为大语言模型提供一系列可应用于PyTorch框架的算子，加速模型的推理与训练。  

FlagGems通过对PyTorch的后端aten算子进行覆盖重写，实现算子库的无缝替换，使用户能够在不修改模型代码的情况下平稳地切换到triton算子库。FlagGems不会影响aten后端的正常使用，并且会带来良好的性能提升。Triton语言为算子库提供了更好的可读性和易用性，同时保持了不逊于CUDA的算子性能，因此开发者只需付出较低的学习成本，即可参与FlagGems的算子开发与建设。  

## 更新日志

### v1.0
- 支持BLAS类算子：addmm, bmm, mm  
- 支持pointwise类算子：abs, add, div, dropout, exp, gelu, mul, pow, reciprocal, relu, rsqrt, silu, sub, triu  
- 支持reduction类算子：cumsum, layernorm, mean, softmax  

### v2.0
- 支持BLAS类算子: mv, outer  
- 支持pointwise类算子: bitwise_and, bitwise_not, bitwise_or, cos, clamp, eq, ge, gt, isinf, isnan, le, lt, ne, neg, or, sin, tanh, sigmoid  
- 支持reduction类算子: all, any, amax, argmax, max, min, prod, sum, var_mean, vector_norm, cross_entropy_loss, group_norm, log_softmax, rms_norm  
- 支持融合算子: skip_rms_norm, skip_layer_norm, gelu_and_mul, silu_and_mul, apply_rotary_position_embedding  

## 快速入门

### 依赖

1. Triton >= 2.2.0  
2. PyTorch >= 2.1.2  
3. Transformers >= 4.40.2  

### 安装  

```shell
git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems
pip install .
```

## 使用  

### 导入

1. 在进程中永久启用  
    ```python
    import flag_gems
    flag_gems.enable()
    ```

2. 暂时启用  
    ```python
    import flag_gems
    with flag_gems.use_gems():
        pass
    ```

3. 示例  
    ```python
    import torch
    import flag_gems

    M, N, K = 1024, 1024, 1024
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    B = torch.randn((K, N), dtype=torch.float16, device="cuda")
    with flag_gems.use_gems():
        C = torch.mm(A, B)
    ```

### 执行

1. 算子正确性测试  
    - 在CUDA上运行参考实现  
        ```shell
        cd tests/flag_gems
        pytest op_accu_test.py
        ```
    - 在CPU上运行参考实现  
        ```shell
        cd tests
        pytest test_xx_ops.py --device cpu
        ```
2. 模型正确性测试  
    ```shell
    cd examples
    pytest model_xx_test.py
    ```

3. 算子性能测试  
    - 测试CUDA性能  
        ```shell
        cd benchmark
        pytest test_xx_perf.py -s
        ```
    - 测试端到端性能  
        ```shell
        cd benchmark
        pytest test_xx_perf.py -s --mode cpu
        ```

2. 运行时打印日志信息  
    ```shell
    pytest program.py --log-cli-level debug
    ```
    测试性能时不建议打开。  

## 支持算子

算子将按照文档[OperatorList.md](https://github.com/FlagOpen/FlagGems/blob/master/OperatorList.md)的顺序逐步实现。

## 支持模型

- Bert-base-uncased  
- Llama-2-7b  

## 支持平台

| Platform | float16 | float32 | bfloat16 |
| :---: | :---: | :---: | :---: |
| Nvidia A100 | ✓ | ✓ | ✓ |

## 贡献代码

欢迎大家参与FlagGems的算子开发并贡献代码，详情请参考[CONTRIBUTING.md](https://github.com/FlagOpen/FlagGems/blob/master/CONTRIBUTING.md)。

## 联系我们

如有疑问，请提交issue，或发送邮件至<a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>。

## 证书

本项目基于[Apache 2.0](https://github.com/FlagOpen/FlagGems/blob/master/LICENSE)。
