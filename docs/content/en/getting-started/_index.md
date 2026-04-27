---
title: Getting Started
weight: 30
---

# Get Started With FlagGems

## 1. Install FlagGems

*FlagGems* can be installed either as a pure python package
or a package with C-extensions for better runtime performance.
See [installation](/FlagGems/getting-started//install/) for some detailed guidance
on different installation options.

## 2. Verify the installation

After having installed the `flag_gems` package and its dependencies,
you may want to verify if they work as expected.

### 2.1 Verify PyTorch environment

The first thing you want to check is that you can import
`torch` in your working environment:

```shell
python -c "import torch; print(torch.__version__)"
```

If you are using a [non-NVIDIA platform](/FlagGems/usage/non-nvidia/),
you may have a PyTorch plugin that is provided by the backend vendor.
You can perform a similar verification against this plugin.
For example, on a MooreThreads GPU platform, you can verify if the plugin
works using the following command:

```shell
python -c "import torch_musa; print(torch_musa.__version__)"
```

### 2.2 Verify the Triton setup

The next verification is against the `triton` package.
You can check if `triton` is working as expected using the following
command:

```shell
python -c "import torch, triton; print(triton.__version__)"
```

Note that if you are using a [non-NVIDIA platform](/FlagGems/usage/non-nvidia/),
you have to consult your platform vendor to see if they have
a customized version.

> [!WARNING]
> **Warning**
>
> Usually, the vendor-customized version of `triton` has the same name
> with the upstream package. Please **double confirm** that you are using
> the correct package before proceeding.

### 2.3 Verify the FlagGems installation

You can do a similar verification for the `flag_gems` package,
using the following command:

```shell
python -c "import flag_gems; print(flag_gems.__version__)"
```

## 3. Start using FlagGems

You can enable the accelerated operators from `flag_gems` in many ways.
The following code snippet enables `flag_gems` globally:

```python
import flag_gems

flag_gems.enable()

# your code goes here
```

You can also enable `flag_gems` in a specific context for a certain
section of your code, as shown below:

```python
# your code goes here ..

# enable flag_gems in a specific context
with flag_gems.use_gems():
    # use the accelerated operators here
    # ...
```

For example:

```python
import torch
import flag_gems

M, N, K = 1024, 1024, 1024
A = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
B = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
with flag_gems.use_gems():
    C = torch.mm(A, B)
```

You can bypass the PyTorch dispatch process and directly invoke operators from
the `flag_gems.ops` & `flag_gems.fused` package.

```python
import flag_gems

a = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
b = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
c = flag_gems.ops.mm(a, b)

num_tokens = 4096; block_size = 64; topk = 2; num_experts = 128;
topk_ids = torch.randint(
    low=0,
    high=num_experts,
    size=(num_tokens, topk),
    device=flag_gems.device,
    dtype=torch.int32,
)
sorted_ids, expert_ids, num_tokens_post_pad = flag_gems.fused.moe_align_block_size.moe_align_block_size(topk_ids, block_size, num_experts)
```

Check the [using FlagGems](/FlagGems/usage/) section for more detailed
documentation on the various usage patterns about *FlagGems*.
