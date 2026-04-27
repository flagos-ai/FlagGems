---
title: Enabling FlagGems
weight: 20
---

# Enabling FlagGems

To use the operators from the *FlagGems* operator library, import `flag_gems` and enable acceleration
before running your program. You can enable it globally or for a specific code block.
Besides these, you can invoke operators from the `flag_gems.ops` package directly.

## 1. Global Enablement

To apply *FlagGems* optimizations across your entire program or your interaction session:

```python
import flag_gems

# Enable FlagGems operators globally
flag_gems.enable()
```

Once enabled, all supported operators in your code will be replaced automatically
by the optimized *FlagGems* implementations — no further changes needed.
This means the supported `torch.*` / `torch.nn.functional.*` calls will be dispatched
to FlagGems implementations automatically. For example:

```python
import torch
import flag_gems

flag_gems.enable()

x = torch.randn(4096, 4096, device=flag_gems.device, dtype=torch.float16)
y = torch.mm(x, x)
```

## 2. Scoped Enablement

When needed, you can enable *FlagGems* only within a specific code block
using a `with...` statement:

```python
import flag_gems
import torch

# Enable flag_gems for specific operations
with flag_gems.use_gems():

    # Code inside this block will use FlagGems-accelerated operators
    x = torch.randn(4096, 4096, device=flag_gems.device, dtype=torch.float16)
    y = torch.mm(x, x)
```

This scoped usage is useful when you want to:

- perform performance benchmarks, or
- compare correctness between implementations, or
- apply acceleration selectively in complex workflows.

## 3. Direct invocation

You can bypass the PyTorch dispatch process and directly invoke operators from
the `flag_gems.ops` package.

```python
import torch
from flag_gems import ops
import flag_gems

a = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
b = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
c = ops.mm(a, b)
```

For fused operators, you can import and use them directly from the `flag_gems.fused` package:

```python
import torch
import flag_gems
from flag_gems.fused.moe_align_block_size import moe_align_block_size

# Example usage of moe_align_block_size
num_tokens = 4096
topk = 2
num_experts = 128

# topk_ids should be expert indices for each token
topk_ids = torch.randint(
    low=0,
    high=num_experts,
    size=(num_tokens, topk),
    device=flag_gems.device,
    dtype=torch.int32,
)

sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
    topk_ids=topk_ids,
    block_size=128,
    num_experts=num_experts,
)
```

## 4. Query Registered Operators

After having enabled *FlagGems*, you can check the operators registered:

```python
import flag_gems

flag_gems.enable()

# Get list of registered function names
registered_funcs = flag_gems.all_registered_ops()
print("Registered functions:", registered_funcs)

# Get list of registered operator keys
registered_keys = flag_gems.all_registered_keys()
print("Registered keys:", registered_keys)
```

This is useful for debugging or verifying which operators are active.

## 5. Advanced Usage

For advanced usage scenarios, check the following related documentation:

- [Selective enablement](/FlagGems/usage/selective/)
- [Using experimental operators](/FlagGems/usage/experimental/)
- [Enabling logging](/FlagGems/usage/debugging/) for debugging
- [Using FlagGems on non-NVIDIA hardware](/FlagGems/usage/non-nvidia/)
- [Using FlagGems in a multi-GPU or distributed environment](/FlagGems/usage/distributed/)
- [Integrating FlagGems with a popular framework](/FlagGems/usage/frameworks/)
- [Building your own models using FlagGems modules](/FlagGems/usage/modules/)
- [Enable pre-tuning for better performance](/FlagGems/usage/tuning/)
- [Using C++ wrapped operators for optimal performance](/FlagGems/usage/cpp/)
