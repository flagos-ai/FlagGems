---
title: 概述
weight: 5
---

<!--
# Overview
-->
# 概述

<!--
FlagGems supports several usage patterns. You can choose from these patterns based on your
requirements. These patterns are not mutually exclusive. You can use different patterns
in different contexts.
-->
*FlagGems* 支持多种使用模式。你可以根据自己的需要作出选择。
这些使用模式并非全部是互斥的。在不同的上下文，你可以使用不同的模式。

<!--
- **[Enable FlagGems](/FlagGems/usage/basic/)**

  Before using operators from the *FlagGems* library, you need to enable the package in your program.
  You can do a global enablement and you can choose a scoped enablement using contextual manager.
-->
- **[启用 FlagGems](/FlagGems/zh-cn/usage/basic/)**

  在使用 *FlagGems* 中的算子之前，你需要在你的程序中启用 `flag_gems` 包。
  *FlagGems* 支持全局性的启用，也支持基于上下文管理器的局部启用。

<!--
- **[Selective enablement](/FlagGems/usage/selective/)**:

  When you have a specific requirements regarding enabling or disabling certain
  operators in your program, you can enable operators selectively.
  This is usually done based on the operator performance on your hardware platform and/or
  the workloads you are running.
-->
- **[选择性启用](/FlagGems/zh-cn/usage/selective/)**:

  如果你在自己的程序中对启用、禁用某些算子有一些特殊的需求，可以选择性地启用、禁用它们。
  通常，这类选择的依据之一是算子在你所使用的硬件平台上展现出来的性能，
  另一个依据则是你所运行的工作负载的性质。

<!--
- **[Enable logging](/FlagGems/usage/debugging/)**:

  When you want to dump operator invocation traces, you can choose to enable
  FlagGems with debugging options.
-->
- **[启用日志输出](/FlagGems/zh-cn/usage/debugging/)**:

  如果希望检视算子调用的轨迹，你可以在启用 `flag_gems` 时添加日志输出选项。

<!--
- **[Enable experimental operators](/FlagGems/usage/experimental/)**

  There are some operators in the *FlagGems* library that are still in experiment
  stage. You can enable them in you workflow nevertheless if you want to try them out.
-->
- **[启用实验性算子](/FlagGems/zh-cn/usage/experimental/)**

  *FlagGems* 算子库中有一些算子仍然处于实验性阶段，尚未经过严格的产品环境检验。
  即便如此，如果你希望尝试这些算子，也可以在你的工作流中启用它们。

<!--
- **[Using FlagGems on non-NVIDIA hardware](/FlagGems/usage/non-nvidia/)**

  If you are running your workloads on some non-NVIDIA hardware, you can still
  check if the hardware is supported by FlagGems. One of the benefits of using
  *FlagGems* is that you don't need to worry about platform portability.
-->
- **[在非 NVIDIA 平台上使用 FlagGems](/FlagGems/zh-cn/usage/non-nvidia/)**

  如果你在使用非 NVIDIA 的硬件平台运行自己的工作负载，你仍然可以查看 *FlagGems*
  是否已经支持你所使用的硬件平台。
  使用 *FlagGems* 的好处之一是你不需要过度担心跨平台的可移植性问题。

<!--
- **[Running in a multi-GPU or distributed environment](/FlagGems/usage/distributed/)**

  If you are running your application in a multi-GPU or distributed environment
  such as a distributed inference platform backed by vLLM, you can check how to
  enable *FlagGems* in these environments. You may need to do some environment preparation
  before enabling FlagGems
-->
- **[运行多 GPU 或分布式环境](/FlagGems/zh-cn/usage/distributed/)**

  <!--
  If you are running your application in a multi-GPU or distributed environment
  such as a distributed inference platform backed by vLLM, you can check how to
  enable *FlagGems* in these environments. You may need to do some environment preparation
  before enabling FlagGems.
  -->
  如果你在多 GPU 或者分布式环境下运行自己的 AI 应用，例如基于 vLLM 的分布式推理平台，
  你可以查阅如何在这类环境中启用 *FlagGems*。你可能需要在启用 *FlagGems*
  之前完成一些环境准备工作。

<!--
- **[Integration with a popular framework](/FlagGems/usage/frameworks/)**

  The *FlagGems* library can be integrated with popular training or inference frameworks like
  [Hugging Face Transformers](https://huggingface.co/docs/transformers/index),
  [vLLM](https://docs.vllm.ai/en/latest/), [Metatron-LM](https://github.com/NVIDIA/Megatron-LM),
  and so on.
-->
- **[与常用框架集成](/FlagGems/zh-cn/usage/frameworks/)**

  *FlagGems* 算字库可以与常见的训练或推理框架集成，例如
  [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)、
  [vLLM](https://docs.vllm.ai/en/latest/)、
  [Metatron-LM](https://github.com/NVIDIA/Megatron-LM)
  等等。

<!--
- **[Building your own models](/FlagGems/usage/modules/)**

  The *FlagGems* project provides a growing collection of modules that are ready to
  be integrated into your models, be it a new one or a adapted one.
-->
- **[构建自己的模型](/FlagGems/zh-cn/usage/modules/)**

  *FlagGems* 项目提供一组可以被直接集成到你的模型当中的模块，这个集合仍在持续扩大。
  这里所说的模型可以是一个从头开发的模型，也可以是一个适配的模型。

<!--
- **[Enable pre-tuning for better performance](/FlagGems/usage/tuning/)**

  *FlagGems* provides [`LibTuner`](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/utils/libentry.py#L139),
  a lightweight enhancement to Triton’s autotuning system.
  It helps mitigate runtime overhead in Triton's default autotuning process.
-->
- **[启用预优化获得更好的性能](/FlagGems/zh-cn/usage/tuning/)**

  *FlagGems* 提供 [`LibTuner`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/src/flag_gems/utils/libentry.py#L206) 类，
  作为对 Triton 的自动调优系统的轻量级增强。
  这个类可以帮助你优化 Triton 自动调优过程中存在的运行时开销。

<!--
- **[Using C++ wrapped operators for optimal performance](/FlagGems/usage/cpp/)**

  *FlagGems also provides a growing set of operators which are deeply optimized
  using C++ language. You may want to give them a try if they are applicable to your scenario.
-->
- **[使用 C++ 封装的算子实现更佳性能](/FlagGems/zh-cn/usage/cpp/)**

  *FlagGems* 还提供一个使用 C++ 语言深度优化的算子集合，这个集合也在持续增长中。
  如果其中的算子适合于你的应用场景，不妨尝试使用这类算子。
