---
title: 简介
weight: 10
bookCollapseSection: true
---

[<img width="2182" height="602" alt="github+banner-20260130" src="/FlagGems/images/banner-20260130.png" />](https://flagos.io/)

<!--
# FlagGems Overview

## About FlagGems

*FlagGems* is a high-performance general operator library implemented
in the [Triton](https://github.com/openai/triton) language.
It aims to provide a suite of kernel functions to accelerate LLM training and inference.
-->
# FlagGems 简介

## 关于 FlagGems

*FlagGems* 是一个高性能的通用算子库，使用 [Triton](https://github.com/openai/triton)
编程语言实现。项目的目标是提供一套核心（kernel）函数来加速 LLM 训练与推理。

<!--
By registering with the ATen backend of PyTorch, *FlagGems* facilitates a seamless transition,
allowing users to switch to the Triton function library without the need to modify their model code.
-->
通过将自身算子实现注册到 PyTorch 的 ATen 后端，*FlagGems* 可以实现无缝的衔接，
方便用户在不需要修改自身模型代码的前提下迁移到 Triton 函数库。

<!--
FlagGems is supported by the OpenAI Triton compiler (for NVIDIA and AMD) and
[FlagTree compiler](https://github.com/flagos-ai/flagtree/) for different AI hardware platforms.
Users can continue to use the ATen backend as usual while enjoying significant performance enhancement.
The Triton language offers benefits in readability, user-friendliness and performance comparable to CUDA.
This convenience allows developers to engage in the development of *FlagGems* with minimal learning effort.
-->
*FlagGems* 可以被 OpenAI 的 Triton 编译器（针对 NVIDIA 和 AMD 芯片）和
[FlagTree 编译器](https://github.com/flagos-ai/FlagTree) 支持；
后者可以支持多种不同的 AI 硬件平台。
用户可以像往常一样使用 ATen 后端，同时藉由 *FlagGems* 算子库实现性能上的提升。
Triton 编程语言在代码可读性、用户友好性上都有很好表现，并且所获得的性能与 CUDA
原生平台具有可比较性。Triton 所提供的这种便利使得开发者能够很快学会并参与到
*FlagGems* 算子的开发工作中。

<!--
## Next step

- Review [features highlighted](/FlagGems/overview/features/)
- Review [platforms supported](/FlagGems/overview/platforms/)
- [Getting started with FlagGems](/FlagGems/getting-started/)
- Check the project [changelog](/FlagGems/references/changelog/)
- Review the list of [operators suppored](/FlagGems/references/changelog/)
-->
## 下一步

- 阅读[功能特性概览](/FlagGems/zh-cn/overview/features/)
- 了解[FlagGems 所支持的平台](/FlagGems/zh-cn/overview/platforms/)
- 尝试[开始使用 FlagGems](/FlagGems/zh-cn/getting-started/)
- 查看 FlagGems 项目的[变更历史](/FlagGems/zh-cn/references/changelog/)
- 了解 FlagGems 目前支持的[算子集合](/FlagGems/zh-cn/references/changelog/)

<!--
## Supported models
-->
## 模型支持

- Bert-base-uncased
- Llama-2-7b
- Llava-1.5-7b

<!--
## Contribution

If you are interested in contributing to the FlagGems project, please refer to
the [contributing guide](/FlagGems/contribution/) page.
Any kind of contributions would be highly appreciated.
-->
## 参与开发

如果你对 *FlagGems* 项目的愿景感兴趣，愿意参与其开发活动，
请阅读[贡献指南](/FlagGems/zh-cn/contribution/) 小节。
我们欢迎任何形式的贡献。

<!--
## Contact us

If you have any questions about FlagGems, please submit an issue, or contact us through
<a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.
-->
## 联系我们

如果你对 *FlagGems* 有任何问题，请在 GitHub 代码仓库上登记你的问题，
或者通过 <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>
邮箱与我们联系。

<!--
We also created WeChat group for FlagGems. Scan the QR code to join the group chat!
To get the first hand message about project updates and new release,
or having any questions or ideas, join us now!
-->
我们还为 *FlagGems* 创建了微信群。你可以微信扫描下面的二维码，参与群聊。
要想了解项目的最新进展、新版本的规划特性，或者提出任何问题与建议，
请考虑加入我们！

<p align="center">
 <img src="https://github.com/user-attachments/assets/69019a23-0550-44b1-ac42-e73f06cb55d6" alt="bge_wechat_group" class="center" width="200">
</p>

<!--
## License

The *FlagGems* project is licensed under
[Apache 2.0](https://github.com/flagos-ai/FlagGems/blob/master/LICENSE).
-->
## 许可协议

*FlagGems* 项目使用
[Apache 2.0](https://github.com/flagos-ai/FlagGems/blob/master/LICENSE)
许可协议。
