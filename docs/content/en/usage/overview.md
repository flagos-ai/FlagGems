---
title: Overview
weight: 10
---

# Overview

FlagGems supports several usage patterns. You can choose from these patterns based on your
requirements. These patterns are not mutually exclusive. You can use different patterns
in different contexts.

- **[Enable FlagGems](/FlagGems/usage/basic/)**

  Before using operators from the *FlagGems* library, you need to enable the package in your program.
  You can do a global enablement and you can choose a scoped enablement using contextual manager.

- **[Selective enablement](/FlagGems/usage/selective/)**:

  When you have a specific requirements regarding enabling or disabling certain
  operators in your program, you can enable operators selectively.
  This is usually done based on the operator performance on your hardware platform and/or
  the workloads you are running.

- **[Enable logging](/FlagGems/usage/debugging/)**:

  When you want to dump operator invocation traces, you can choose to enable
  FlagGems with debugging options.

- **[Enable experimental operators](/FlagGems/usage/experimental/)**

  There are some operators in the *FlagGems* library that are still in experiment
  stage. You can enable them in you workflow nevertheless if you want to try them out.

- **[Using FlagGems on non-NVIDIA hardware](/FlagGems/usage/non-nvidia/)**

  If you are running your workloads on some non-NVIDIA hardware, you can still
  check if the hardware is supported by FlagGems. One of the benefits of using
  *FlagGems* is that you don't need to worry about platform portability.

- **[Running in a multi-GPU or distributed environment](/FlagGems/usage/distributed/)**

  If you are running your application in a multi-GPU or distributed environment
  such as a distributed inference platform backed by vLLM, you can check how to
  enable *FlagGems* in these environments. You may need to do some environment preparation
  before enabling FlagGems.

- **[Integration with a popular framework](/FlagGems/usage/frameworks/)**

  The *FlagGems* library can be integrated with popular training or inference frameworks like
  [Hugging Face Transformers](https://huggingface.co/docs/transformers/index),
  [vLLM](https://docs.vllm.ai/en/latest/), [Metatron-LM](https://github.com/NVIDIA/Megatron-LM),
  and so on.

- **[Building your own models](/FlagGems/usage/modules/)**

  The *FlagGems* project provides a growing collection of modules that are ready to
  be integrated into your models, be it a new one or a adapted one.

- **[Enable pre-tuning for better performance](/FlagGems/usage/tuning/)**

  *FlagGems* provides [`LibTuner`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/src/flag_gems/utils/libentry.py#L206),
  a lightweight enhancement to Triton’s autotuning system.
  It helps mitigate runtime overhead in Triton's default autotuning process.

- **[Using C++ wrapped operators for optimal performance](/FlagGems/usage/cpp/)**

  *FlagGems* also provides a growing set of operators which are deeply optimized
  using C++ language. You may want to give them a try if they are applicable to your scenario.
