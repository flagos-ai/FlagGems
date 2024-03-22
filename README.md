## Introduction

FlagGems is an operator library implemented in [Triton Language](https://github.com/openai/triton). It is designed for large language models to provide a set of operators that can be used in PyTorch framework.

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
    import gems
    gems.enable()
    ```

2. Enable temporarily  
    ```python
    import gems
    with gems.use_gems():
        pass
    ```

3. Example  
    ```python
    import torch
    import gems

    M, N, K = 1024, 1024, 1024
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    B = torch.randn((K, N), dtype=torch.float16, device="cuda")
    with gems.use_gems():
        C = torch.mm(A, B)
    ```

### Execute

1. Run Tests  
    - Operator Accuracy  
        ```shell
        cd tests/gems
        pytest op_accu_test.py
        ```
    - Model Accuracy  
        ```shell
        cd tests/gems
        pytest model_bert_test.py
        ```
    - Operator Performance  
        ```shell
        cd tests/gems
        python op_perf_test.py
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

## Contact us

If you have any questions about our project, please submit an issue, or contact us through <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

## License

The FlagGems project is based on [Apache 2.0](https://github.com/FlagOpen/FlagGems/blob/master/LICENSE).
