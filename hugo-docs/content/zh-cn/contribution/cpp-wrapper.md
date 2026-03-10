---
title: 添加 C++ 封装的算子
weight: 20
---
<!--
# Add a C++ wrapper

To add a C++ wrapped operator, you need to first build FlagGems with C++ extensions enabled.
Please refer to [Installation](./installation.md).
-->
# 添加 C++ 封装的算子

要添加一个 C++ 封装的算子，你需要首先在安装 FlagGems 时启用 C++ 扩展能力特性。
请参阅[FlagGems 安装](/FlagGems/zh-cn/installation/)文档。

<!--
## Write the wrapper

Follow the following steps to add a new C++ wrapped operator:

- Add a function prototype for the operator in the `include/flag_gems/operators.h` file.
- Add the operator function implementation in the `lib/op_name.cpp` file.
- Change the cmakefile `lib/CMakeLists.txt` accordingly.
- Add python bindings in `src/flag_gems/csrc/cstub.cpp`
- Add the `triton_jit` function in `triton_src`.
-->
## 编写封装层 {#write-the-wrapper}

按照如下步骤添加一个新的 C++ 封装的算子：

- 在 `include/flag_gems/operators.h` 文件中为算子添加函数原型声明；
- 在 `lib/<op_name>.cpp` 文件中添加算子的函数实现；
- 修改 `lib/CMakeLists.txt` 文件，包含新的算子；
- 在 `src/flag_gems/csrc/cstub.cpp` 文件中为算子添加 Python 绑定逻辑；
- 在 `triton_src/` 下面为算子添加 `triton_jit` 函数；

  > [!TIP]
  > **提示**
  >
  > 目前我们使用一个专门的目录来存放 `triton_jit` 函数。
  > 将来我们会复用 `flag_gems` 目录下 Python 代码中的 `triton_jit` 函数。

<!--
## Write test cases

FlagGems uses `ctest` and `googletest` for C++ unit tests.
After having finished the C++ wrapper, a corresponding C++ test case should be added.
Add your unit test in `ctests/test_triton_xxx.cpp` and `ctests/CMakeLists.txt`.
Finally, build your test source and run it with [C++ Tests](./ctest_in_flaggems.md).
-->
## 编写测试用例  {#write-test-cases}

FlagGems 使用 `ctest` 和 `googletest` 来执行 C++ 代码的单元测试。
在完成 C++ 封装的算子实现之后，你需要为其添加对应的 C++ 测试用例。
你的测试用例应该添加到 `ctests/test_triton_<xxx>.cpp` 文件中，
并且在 `ctests/CMakeLists.txt` 中列出测试用例文件。
最后，构建你的测试代码并使用 [C++ 测试](/FlagGems/zh-cn/testing/ctests/)文档中所给方法来执行测试。

<!--
## Create a PR for your code

When submitting a PR, it's desirable to provide end-to-end performance data
in your PR description.
-->
## 为你的代码提交 PR

在提交 PR 时，我们希望你能够在 PR 描述中包含端到端的性能测试数据以方便评审。
