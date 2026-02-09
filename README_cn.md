[<img width="2182" height="602" alt="github+banner-20260130" src=".github/assets/banner-20260130.png" />](https://flagos.io/)

[中文版|[English](./README.md)]

## 介绍

FlagGems 是 [FlagOS](https://flagos.io/) 的一部分。
FlagOS 是一个面向多元AI芯片的开源、统一系统软件栈，旨在打通模型、系统与芯片层，
培育开放协作的生态系统。它支持“一次开发，多芯运行”的工作流，兼容多样化的 AI 加速芯片，
释放硬件性能潜力，消除各类 AI 芯片专用软件栈之间的碎片化问题，
并大幅降低大模型在多种 AI 硬件移植与维护的成本。

FlagGems 是一个使用 OpenAI 推出的[Triton 编程语言](https://github.com/openai/triton)实现的高性能通用算子库，
旨在为大语言模型提供一系列可应用于 PyTorch 框架的算子，加速模型面向多种后端平台的推理与训练。

FlagGems 通过对 PyTorch 的后端 ATen 算子进行覆盖重写，实现算子库的无缝替换，
一方面使得模型开发者能够在无需修改底层 API 的情况下平稳地切换到 Triton 算子库，
使用其熟悉的 PyTorch API 同时享受新硬件带来的加速能力，
另一方面对 kernel 开发者而言，Triton 语言提供了更好的可读性和易用性，可媲美 CUDA 的性能，
因此开发者只需付出较低的学习成本，即可参与 FlagGems 的算子开发与算子库建设。

## 特性

- 支持的算子数量规模较大
- 部分算子已经过深度性能调优
- 可直接在 Eager 模式下使用, 无需通过 `torch.compile`
- Pointwise 自动代码生成，灵活支持多种输入类型和内存排布
- Triton kernel 调用优化
- 灵活的多后端支持机制
- 代码库已集成十余种后端
- C++ Triton 函数派发 (开发中)

更多特性细节可参阅 [./docs/features.md] 文档。

## 快速入门

- 参考文档[开始使用](./docs/getting-started.md)快速安装使用 FlagGems。
- 参考文档[使用方法](./docs/usage_cn.md)了解详细用法。
- 参考文档[用法指南](./docs/how_to_use_flaggems.md)了解使用注意事项及高级选项。

## 供测试的模型

- Bert-base-uncased
- Llama-2-7b
- Llava-1.5-7b

## 贡献代码

<!--TODO(Qiming): Sync this with English version.-->
- 欢迎大家参与 FlagGems 的算子开发并贡献代码，详情请参考[CONTRIBUTING.md](./CONTRIBUTING_cn.md)。
- 欢迎提交问题报告（Issue）或者特性请求（Feature Request）
- 关于项目的疑问或建议，可发送邮件至<a href="mailto:contact@flagos.io">contact@flagos.io</a>。
- 我们为 FlagGems 创建了微信群。扫描二维码即可加入群聊！第一时间了解我们的动态和信息和新版本发布，
  或者有任何问题或想法，请立即加入我们！

  <img width="204" height="180" alt="开源小助手" src="https://github.com/user-attachments/assets/4e9a8566-c91e-4120-a011-6b5577c1a53d" />

## 引用

欢迎引用我们的项目：

```bibtex
@misc{flaggems2024,
    title={FlagOpen/FlagGems: FlagGems is an operator library for large language models implemented in the Triton language.},
    url={https://github.com/FlagOpen/FlagGems},
    journal={GitHub},
    author={BAAI FlagOpen team},
    year={2024}
}
```

## 许可证

本项目采用 [Apache License (version 2.0)](./LICENSE) 授权许可。
