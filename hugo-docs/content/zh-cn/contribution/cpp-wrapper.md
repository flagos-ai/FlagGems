---
title: 添加 C++ 封装的算子
weight: 20
---
<!--
# Add a C++ Wrapped Operator

To add a C++ wrapped operator, you need to first build FlagGems with C++ extensions enabled.
Please refer to [installation](/FlagGems/getting-started/installation/) section
for detailed instructions on setting up `flag_gems` with C++ extensions enabled.
-->
# 添加 C++ 封装的算子

要添加一个 C++ 封装的算子，你需要首先在安装 FlagGems 时启用 C++ 扩展能力特性。
请参阅[FlagGems 安装](/FlagGems/zh-cn/getting-started/installation/)文档，
了解如何在安装 `flag_gems` 时启用 C++ 扩展的详细步骤。

<!--
## 1. Write the wrapper

Follow the following steps to add a new C++ wrapped operator:

- Add a function prototype for the operator in the `include/flag_gems/operators.h` file.
- Add the operator function implementation in the `lib/op_name.cpp` file.
- Change the cmakefile `lib/CMakeLists.txt` accordingly.
- Add python bindings in `src/flag_gems/csrc/cstub.cpp`
- Add the `triton_jit` function in `triton_src`.
-->
## 1. 编写封装层 {#write-the-wrapper}

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
## 2. Write test cases

FlagGems uses `ctest` and `googletest` for C++ unit tests.
After having finished the C++ wrapper, a corresponding C++ test case should be added.
Add your unit test in `ctests/test_triton_xxx.cpp` and `ctests/CMakeLists.txt`.
Finally, build your test source and run it with
[C++ Tests](/FlagGems/testing/ctests/).
-->
## 2. 编写测试用例  {#write-test-cases}

FlagGems 使用 `ctest` 和 `googletest` 来执行 C++ 代码的单元测试。
在完成 C++ 封装的算子实现之后，你需要为其添加对应的 C++ 测试用例。
你的测试用例应该添加到 `ctests/test_triton_<xxx>.cpp` 文件中，
并且在 `ctests/CMakeLists.txt` 中列出测试用例文件。
最后，构建你的测试代码并使用 [C++ 测试](/FlagGems/zh-cn/testing/ctests/)文档中所给方法来执行测试。

<!--
## 3. Running the C++ test cases

If you build FlagGems with C++ extensions with cmake option `FLAGGEMS_BUILD_CTESTS` set to `ON`,
you can run the ctest in the dir `FlagGems/build/cpython-3xx` with the following command:
-->
## 3. 运行 C++ 测试用例

如果你在构造 FlagGems 时将 cmake 选项 `FLAGGEMS_BUILD_CTESTS` 设置为 `ON`，
进而启用了 C++ 扩展，那么你就可以使用下面的命令在 `FlagGems/build/cpython-3xx`
目录下运行 ctest。

```shell
ctest .
```

<!--
This will run all the test files under `ctests/`.
You can also use the following command to run a specific test with log info:
-->
上面这条命令会运行 `ctests/` 目录下的所有测试用例文件。
你也可以使用下面的命令来运行一个指定的测试用例，同时启用日志信息输出：

```shell
ctest -V —R <regex>
```

<!--
where

- `-R <regex>`: runs only the tests whose names match the given regular expression.
- `-V`: enables the verbose mode, printing detailed output for each test,
  including any messages sent to stdout/stderr.

For example:
-->
其中

- `-R <regex>`：运行名字与所给的正则表达式匹配的所有测试用例；
- `-V`：启用详尽（verbose）输出模式，打印每个测试的详细输出，
  测试用例写入到标准输出（stdout）或标准错误输出（stderr）的所有消息。

例如：

```shell
TORCH_CPP_LOG_LEVEL=INFO ctest -V -R test_triton_reduction
```

<!--
We use PyTorch Aten log as well, so you need set the env `TORCH_CPP_LOG_LEVEL=INFO`
for more logs in `libtorch_example`.
-->
我们也使用 PyTorch 的 Aten 日志机制，所以你需要设置环境变量
`TORCH_CPP_LOG_LEVEL=INFO` 才能获得 `libtorch_example` 中的更多日志信息。

<!--
## Create a PR for your code

When everything works as expected, it's time to submit a pull request (PR).
It's desirable to provide some end-to-end performance data in your PR description,
in addition to a brief summary about what the operator does.
-->
## 4. 为你的代码提交 PR

当所有一切都工作正常时，该到提交拉去请求（Pull Request，PR）的时候了。
在提交 PR 时，我们希望你在针对算子所做的工作提供简要描述之外，
能够在 PR 描述中包含端到端的性能测试数据以方便评审。
