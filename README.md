# FlagGems

## Description

1. Requirements  
    - Triton >= 2.2.0  
    - PyTorch >= 2.1.2  

2. Installation  
    ```shell
    pip install .
    ```

3. Usage  
    ```python
    import flag_gems
    flag_gems.enable()
    ```

## Operators

- addmm  
    - only support torch.float16

- bmm  
    - only support torch.float16

- cumsum  
    - only support torch.float32  
    - support synamic dimension  

- dropout  

- gelu  
    - support torch.float16 and torch.float32
    - support dynamic dimension  

- layernorm  
    - support torch.float16 and torch.float32
    - only support 2d tensor  

- mm  
    - only support torch.float16

- relu  
    - support torch.float16 and torch.float32
    - support dynamic dimension 

- silu  
    - support torch.float16 and torch.float32
    - support dynamic dimension 

- softmax  
    - support torch.float16 and torch.float32
    - support dynamic dimension 
