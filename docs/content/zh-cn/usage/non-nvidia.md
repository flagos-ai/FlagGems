---
title: 使用非英伟达（NVIDIA）硬件
weight: 50
---

<!--
# Running FlagGems on Non-NVIDIA Hardware

## 1. Supported platforms
-->
# 在非英伟达（NVIDIA）硬件上使用 FlagGems

## 1. 支持的平台

<!--
FlagGems supports a range of AI chips/platforms beyond NVIDIA.
For an up-to-date list of validated platforms, please refer to
[Supported Platforms](/FlagGems/overview//features/#platforms-supported)
-->
*FlagGems* 在 NVIDIA 芯片之外支持若干不同类型的 AI 芯片或平台。
请参阅[平台支持](/FlagGems/zh-cn/overview/features/#platforms-supported)
文档了解已经验证过的平台的最新列表。

<!--
## 2. Unified usage interface

Regardless of the underlying hardware, the usage of `flag_gems` remains exactly the same.
There is no need to modify application code when switching from NVIDIA to non-NVIDIA platforms.
-->
## 2. 统一的使用接口

无论下层使用的是哪种硬件，`flag_gems` 的用法始终保持不变。
从 NVIDIA 平台切换到非 NVIDIA 平台时，一般而言不需要更改应用代码。

<!--
Once you have imported `flag_gems` into your code and
[enabled *FlagGems* acceleration](/FlagGems/usage/basic/#enabling-flaggems),
the operator dispatch mechanism will automatically route operator invocations
to the correct implementation for the backend.
This provides a consistent developer experience across different environments.
-->
当你在代码中导入了 `flag_gems` 包，并且
[启用了 *FlagGems* 加速](/FlagGems/zh-cn/usage/basic/#enabling-flaggems)之后，
算子的派发机制会自动将对算子的调用指向针对当前后端的正确实现上。
这一派发机制为开发者提供了跨不同环境的体验一致性。

<!--
## 3. Platform requirements

Although the usage pattern remains unchanged, there are some prerequisites
when running *FlagGems* on non-NVIDIA platforms.
The *PyTorch* and the *Triton* compiler have to be installed and
properly configured on the target platform.
-->
## 3. 平台需求

尽管 *FlagGems* 的使用模式保持不变，在非 NVIDIA 平台上运行 *FlagGems*
时仍然需要满足一些前置条件。你必须在目标平台上安装了 *PyTorch* 和 *Triton*
编译器。

<!--
There are two common ways to obtain compatible builds:

1. **Consult your hardware vendor**

   Some platforms may require additional setup or patching.
   Hardware vendors typically maintain custom builds of *PyTorch* and *Triton* customized
   or tailored to their chips. Contact the vendor to request the appropriate versions.
-->
通常而言，有两种方式可以实现兼容的软件构建：

1. **咨询你的硬件厂商**

   某些平台需要一些额外的安装配置或者补丁操作。
   硬件厂商通常会针对自己的芯片开发维护 *PyTorch* 和 *Triton* 的定制构建版本。
   你需要与厂商取得联系，以获得这些软件包的正确版本。

<!--
2. **Explore the FlagTree project**

   The [FlagTree](https://github.com/flagos-ai/FlagTree) project offers a unified Triton compiler
   that supports a wide range of AI chips, including NVIDIA and non-NVIDIA platforms.
   It consolidates vendor-specific patches and enhancements into a shared, open-source backend,
   simplifying compiler maintenance and enabling multi-platform compatibility.

   Note that *FlagTree* only provides a compiler framework at the Triton language layer.
   A matching PyTorch build is still required separately.
-->
2. **尝试 FlagTree 项目**

   [FlagTree](https://github.com/flagos-ai/FlagTree) 项目提供一种统一的 Triton 编译器，
   能够支持多种不同的 AI 芯片，包括 NVIDIA 和非 NVIDIA 平台。
   FlagTree 能够将特定于厂商的补丁和增强聚合在一起，形成一个共享的、开源的编译器后端，
   从而简化编译器的维护工作，保证跨多个平台的兼容性。

   需要注意的是 *FlagTree* 仅提供在 Triton 语言层的编译器框架。
   你仍然需要安装部署一个合适的 *PyTorch* 发行版本。

<!--
## 4. Backend auto-detection and manual setting

By default, *FlagGems* automatically detects the current hardware backend during runtime
and selects the corresponding implementation.
In most cases, no manual configuration is required because everything just works
out of the box.
-->
## 4. 后端自动检测与手动设置

默认情况下，*FlagGems* 能够在运行时自动检测当前使用的硬件后端，为之选择对应的算子实现。
很多时候，所有的组件都能够直接工作，不需要手动的配置。

<!--
If the builtin auto-detection mechanism fails or there are compatibility issues
in your environment, you can manually set the target backend to ensure correct runtime behaviors.
To do this, set the following environment variable before running your code:
-->
如果内置的自动后端检测机制失败，或者在你的环境中出现了兼容性相关的问题，
你可以手动设置目标后端，以确保运行时算子的行为是正确的。
你可以通过设置下面的环境变量来指定后端，之后再运行你的代码：

```shell
export GEMS_VENDOR=<厂商名称>
```

<!--
For the list of valid `vendor_name`s, please check the
[supported platforms](/FlagGems/overview/features/#platforms-supported) documentation
for details.
-->
参阅[平台支持](/FlagGems/zh-cn/overview/features/#platforms-supported)文档，
了解不同厂商对应的符号名。

<!--
> [!WARNING]
> **Warning**
>
> This setting should match the actual hardware platform.
> Manually setting an incorrect backend may result in runtime errors.
-->
> [!WARNING]
> **警告**
>
> 手动指定的后端名称要与实际的硬件平台匹配。
> 手动设置一个错误的后端符号名可能会导致软件运行错误。

<!--
You can verify the active backend at runtime by checking `flag_gems.vendor_name`:
-->
你可以通过在运行时查看 `flag_gems.vendor_name` 的取值来检查当前使用的后端。

```python
import flag_gems
print(flag_gems.vendor_name)
```
