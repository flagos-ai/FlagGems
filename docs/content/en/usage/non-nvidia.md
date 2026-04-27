---
title: Running on Non-NVIDIA Hardware
weight: 50
---

# Running FlagGems on Non-NVIDIA Hardware

## 1. Supported platforms

FlagGems supports a range of AI chips/platforms beyond NVIDIA.
For an up-to-date list of validated platforms, please refer to
[Supported Platforms](/FlagGems/overview//features/#platforms-supported)

## 2. Unified usage interface

Regardless of the underlying hardware, the usage of `flag_gems` remains exactly the same.
There is no need to modify application code when switching from NVIDIA to non-NVIDIA platforms.

Once you have imported `flag_gems` into your code and
[enabled *FlagGems* acceleration](/FlagGems/usage/basic/#enabling-flaggems),
the operator dispatch mechanism will automatically route operator invocations
to the correct implementation for the backend.
This provides a consistent developer experience across different environments.

## 3. Platform requirements

Although the usage pattern remains unchanged, there are some prerequisites
when running *FlagGems* on non-NVIDIA platforms.
The *PyTorch* and the *Triton* compiler have to be installed and
properly configured on the target platform.

There are two common ways to obtain compatible builds:

1. **Consult your hardware vendor**

   Some platforms may require additional setup or patching.
   Hardware vendors typically maintain custom builds of *PyTorch* and *Triton* customized
   or tailored to their chips. Contact the vendor to request the appropriate versions.

2. **Explore the FlagTree project**

   The [FlagTree](https://github.com/flagos-ai/FlagTree) project offers a unified Triton compiler
   that supports a wide range of AI chips, including NVIDIA and non-NVIDIA platforms.
   It consolidates vendor-specific patches and enhancements into a shared, open-source backend,
   simplifying compiler maintenance and enabling multi-platform compatibility.

   Note that *FlagTree* only provides a compiler framework at the Triton language layer.
   A matching PyTorch build is still required separately.

## 4. Backend auto-detection and manual setting

By default, *FlagGems* automatically detects the current hardware backend during runtime
and selects the corresponding implementation.
In most cases, no manual configuration is required because everything just works
out of the box.

If the builtin auto-detection mechanism fails or there are compatibility issues
in your environment, you can manually set the target backend to ensure correct runtime behaviors.
To do this, set the following environment variable before running your code:

```shell
export GEMS_VENDOR=<vendor_name>
```

For the list of valid `vendor_name`s, please check the
[supported platforms](/FlagGems/overview/features/#platforms-supported) documentation
for details.

> [!WARNING]
> **Warning**
>
> This setting should match the actual hardware platform.
> Manually setting an incorrect backend may result in runtime errors.

You can verify the active backend at runtime by checking `flag_gems.vendor_name`:

```python
import flag_gems
print(flag_gems.vendor_name)
```
