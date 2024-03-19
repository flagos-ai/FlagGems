# FlagGems

## Description

1. Requirements  
    - Triton >= 2.2.0  
    - PyTorch >= 2.1.2  

2. Installation  
    ```shell
    git clone https://gitee.com/flagir/flag-gems.git
    cd flag-gems
    pip install .
    ```

3. Usage  
    - Enalbe permanently  
        ```python
        import flag_gems
        flag_gems.enable()
        ```
    - Enable temporarily  
        ```python
        import flag_gems
        with flag_gems.Context():
            pass
        ```

4. Disable Flag Info  
    ```shell
    python -O program.py
    ```

5. Tests  
    - Operator Accuracy  
        ```shell
        cd tests/flag_gems
        pytest op_accu_test.py
        python -m pytest -k test_accuracy_addmm op_accu_test.py
        ```
    - Operator Performance  
        ```shell
        cd tests/flag_gems
        python op_perf_test.py
        ```
    - Model Accuracy  
        ```shell
        cd tests/flag_gems
        python model_bert_test.py
        ```

## Operators

- addmm  
    - support torch.float16, torch.float32 and torch.bfloat16  

- bmm  
    - support torch.float16, torch.float32 and torch.bfloat16  

- cumsum  
    - support torch.float16, torch.float32 and torch.bfloat16  
    - support high dimension  

- dropout  
    - support torch.float16, torch.float32 and torch.bfloat16  
    - support inference and training  

- gelu  
    - support torch.float16, torch.float32 and torch.bfloat16  
    - support high dimension  

- layernorm  
    - support torch.float16, torch.float32 and torch.bfloat16  
    - support high dimension  

- mm  
    - support torch.float16, torch.float32 and torch.bfloat16  

- relu  
    - support torch.float16, torch.float32 and torch.bfloat16  
    - support high dimension 

- silu  
    - support torch.float16, torch.float32 and torch.bfloat16  
    - support high dimension 

- softmax  
    - support torch.float16, torch.float32 and torch.bfloat16  
    - support high dimension 

- triu
    - support torch.float16, torch.float32 and torch.bfloat16  
    - support high dimension 
