---
title: Overview
weight: 10
bookCollapseSection: true
---

[<img width="2182" height="602" alt="github+banner-20260130" src="/FlagGems/images/banner-20260130.png" />](https://flagos.io/)

# FlagGems Overview

## About FlagGems

*FlagGems* is a high-performance general operator library implemented
in the [Triton](https://github.com/openai/triton) language.
It aims to provide a suite of kernel functions to accelerate LLM training and inference.

By registering with the ATen backend of PyTorch, *FlagGems* facilitates a seamless transition,
allowing users to switch to the Triton function library without the need to modify their model code.

FlagGems is supported by the OpenAI Triton compiler (for NVIDIA and AMD) and
[FlagTree compiler](https://github.com/flagos-ai/flagtree/) for different AI hardware platforms.
Users can continue to use the ATen backend as usual while enjoying significant performance enhancement.
The Triton language offers benefits in readability, user-friendliness and performance comparable to CUDA.
This convenience allows developers to engage in the development of *FlagGems* with minimal learning effort.

## Next step

- Review [features highlighted](/FlagGems/overview/features/)
- Review [platforms supported](/FlagGems/overview/platforms/)
- [Getting started with FlagGems](/FlagGems/getting-started/)
- Check the project [changelog](/FlagGems/references/changelog/)
- Review the list of [operators suppored](/FlagGems/references/changelog/)

## Supported models

- Bert-base-uncased
- Llama-2-7b
- Llava-1.5-7b

## Contribution

If you are interested in contributing to the FlagGems project, please refer to
the [contributing guide](/FlagGems/contribution/) page.
Any kind of contributions would be highly appreciated.

## Contact us

If you have any questions about our project, please submit an issue, or contact us through
<a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

We also created WeChat group for FlagGems. Scan the QR code to join the group chat!
To get the first hand message about our updates and new release,
or having any questions or ideas, join us now!

<p align="center">
 <img src="https://github.com/user-attachments/assets/69019a23-0550-44b1-ac42-e73f06cb55d6" alt="bge_wechat_group" class="center" width="200">
</p>

## License

The *FlagGems* project is licensed under
[Apache 2.0](https://github.com/flagos-ai/FlagGems/blob/master/LICENSE).
