<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

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

*FlagGems* 是一个高性能、跨平台的通用算子库，采用 [Triton](https://github.com/openai/triton) 语言实现，旨在提供一套高效的算子核心（Kernel）函数以加速大语言模型（LLM）的训练与推理任务。

通过注册到 PyTorch 的 ATen 后端，*FlagGems* 实现了与 PyTorch 生态的无缝衔接。用户无需修改任何模型代码，即可透明切换至 Triton 算子实现并享受极致性能加速。

*FlagGems* 不仅支持 OpenAI 原生 Triton 编译器（面向 NVIDIA 与 AMD 平台），还全面适配支持多硬件平台的 [FlagTree 编译器](https://github.com/flagos-ai/FlagTree)。用户沿用熟悉的 ATen 后端工作流即可获得显著性能收益。Triton 语言兼具极高的代码可读性与易用性，且生成的 Kernel 性能足以媲美原生 CUDA。这使得开发者能够以极低的学习门槛快速参与到 *FlagGems* 算子的研发与优化中。

## 探索指南

- 阅读[功能特性概览](overview/features.md)
- 了解[支持的硬件平台](overview/platforms.md)
- 快速[开始使用 FlagGems](getting-started/_index.md)
- 查看版本[变更历史](references/changelog.md)
- 查阅已支持的[算子清单](references/operators.md)

## 支持的模型

- Bert-base-uncased
- Llama-2-7b
- Llava-1.5-7b

## 参与贡献

如果您对 *FlagGems* 感兴趣并希望参与社区共建，请参阅[贡献指南](contribution/_index.md)。我们由衷欢迎并感谢任何形式的贡献！

## 联系我们

如对 *FlagGems* 有任何疑问或建议，欢迎在 GitHub 仓库提交 Issue，或发送邮件至 <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>。

我们还建立了 FlagGems 官方微信交流群。欢迎扫描下方二维码加入群聊，第一时间获取项目最新动态、版本发布通知并与开发者深入交流：

<p align="center">
 <img src="https://github.com/user-attachments/assets/69019a23-0550-44b1-ac42-e73f06cb55d6" alt="bge_wechat_group" class="center" width="200">
</p>

## 开源许可

*FlagGems* 项目基于 [Apache 2.0](https://github.com/flagos-ai/FlagGems/blob/master/LICENSE) 许可证开源。
