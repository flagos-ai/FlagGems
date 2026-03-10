---
title: 使用非英伟达（NVIDIA）硬件
weight: 50
---

# Running FlagGems on Non-NVIDIA Hardware

## Supported Platforms

FlagGems supports a range of AI chips beyond NVIDIA.
For an up-to-date list of validated platforms, please refer to
[Supported Platforms](./features.md#platforms-supported)

## Unified Usage Interface

Regardless of the underlying hardware, the usage of `flag_gems` remains exactly the same.
There is no need to modify application code when switching from NVIDIA to non-NVIDIA platforms.

Once you call `import flag_gems` and enable acceleration via `flag_gems.enable()`,
operator dispatch will be automatically routed to the correct backend.
This provides a consistent developer experience across different environments.

## Backend Requirements

Although the usage pattern is unchanged, there are some prerequisites when running *FlagGems* on non-NVIDIA platforms.
The **PyTorch** and the **Triton compiler** have to be installed and properly configured on the target platform.

There are two common ways to obtain compatible builds:

1. **Consult your hardware vendor**

   Hardware vendors typically maintain custom builds of PyTorch and Triton tailored to their chips.
   Contact the vendor to request the appropriate versions.

2. **Explore the FlagTree project**

   The [FlagTree](https://github.com/flagos-ai/flagtree) project offers a unified Triton compiler
   that supports a range of AI chips, including NVIDIA and non-NVIDIA platforms.
   It consolidates vendor-specific patches and enhancements into a shared, open-source backend,
   simplifying compiler maintenance and enabling multi-platform compatibility.

   > [!Note]
   > FlagTree provides Triton only. A matching PyTorch build is still required separately.

> [!Note]
> Some platforms may require additional setup or patching.

## Backend Auto-Detection and Manual Setting

By default, *FlagGems* automatically detects the current hardware backend at runtime
and selects the corresponding implementation.
In most cases, no manual configuration is required, and everything works out of the box.

However, if auto-detection fails or there are compatibility issues in your environment,
you can manually set the target backend to ensure correct runtime behavior.
To do this, set the following environment variable before running your code:

```shell
export GEMS_VENDOR=<your_vendor_name>
```

> ⚠️  This setting should match the actual hardware platform.
> Manually setting an incorrect backend may result in runtime errors.

You can verify the active backend at runtime using:

```python
import flag_gems
print(flag_gems.vendor_name)
```
