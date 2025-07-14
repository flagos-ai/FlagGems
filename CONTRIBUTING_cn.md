[English](./CONTRIBUTING.md)

# FlagGems 贡献者指南

感谢您对 FlagGems 的兴趣！我们使用 GitHub 来托管代码、管理问题和处理拉取请求。在贡献之前，请阅读以下指南。

## 1. 错误报告

请使用 GitHub 的 Issues 来报告错误。在报告错误时，请提供：

- 简单摘要
- 复现步骤
- 确保描述具体且准确！
- 如果可以提供一些示例代码将会帮助开发者快速定位问题

## 2. 代码贡献

在提交拉取请求时，贡献者应描述所做的更改以及更改的原因。如果可以设计测试用例，请提供相应测试。拉取请求在合并前需要 **一位** 成员的批准，而且需要通过代码的持续集成检查。 详细信息见 [代码贡献指南](docs/code_countribution.md)

## 3. 文档补充

FlagGems 的文档存放在 `docs` 目录下，并且当前使用 [MkDocs](https://www.mkdocs.org/) 进行构建和部署，文档 在 CI 流水线中进行 nightly build

# FlagGems 项目结构

```cpp
FlagGems
├── src                  // python 源码
│   └──flag_gems
│       ├──csrc          // C 源文件
│       ├──utils         // python utils
│       ├──ops           // python 独立算子
│       ├──fused         // python 融合算子
│       ├──modules       // python 模块
│       ├──patches       // 布丁脚本
│       ├──testing       // python 测试
├── tests                // python 算子精度测试
├── benchmark            // python 算子性能测试
├── examples             // python 模型性能测试
├── cmake                // c++ cmake 文件构建 C-extension
├── include              // c++ 头文件
├── lib                  // c++ 源码构建 算子库
├── ctest                // c++ 测试文件
├── triton_src           // triton jit_function 源文件
├── docs                 // flag_gems 文档
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── ...
```

# FlagGems 许可证

FlagGems 使用 [Apache 许可证](https://github.com/FlagOpen/FlagGems/blob/master/LICENSE)
