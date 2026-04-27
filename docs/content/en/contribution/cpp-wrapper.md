---
title: Add a C++ Wrapped Operator
weight: 20
---
# Add a C++ Wrapped Operator

To add a C++ wrapped operator, you need to first build FlagGems with C++ extensions enabled.
Please refer to [installation](/FlagGems/getting-started/install/) section
for detailed instructions on setting up `flag_gems` with C++ extensions enabled.

## 1. Write the wrapper

Follow the following steps to add a new C++ wrapped operator:

- Add a function prototype for the operator in the `include/flag_gems/operators.h` file.
- Add the operator function implementation in the `lib/<op_name>.cpp` file.
- Change the cmakefile `lib/CMakeLists.txt` accordingly.
- Add Python bindings in `src/flag_gems/csrc/cstub.cpp`
- Add the `triton_jit` function in `triton_src`.

  > [!TIP]
  > **TIP**
  >
  > Currently we use a dedicated directory to store the `triton_jit` functions.
  > In the future, we will reuse the `triton_jit` functions in Python code under `flag_gems`.

## 2. Write test cases

FlagGems uses `ctest` and `googletest` for C++ unit tests.
After having finished the C++ wrapper, a corresponding C++ test case should be added.
Add your unit test in `ctests/test_triton_xxx.cpp` and `ctests/CMakeLists.txt`.
Finally, build your test source and run it with
[C++ Tests](/FlagGems/testing/ctests/).

## 3. Running the C++ test cases

If you build FlagGems with C++ extensions with cmake option `FLAGGEMS_BUILD_CTESTS` set to `ON`,
you can run the ctest in the dir `FlagGems/build/cpython-3xx` with the following command:

```shell
ctest .
```

This will run all the test files under `ctests/`.
You can also use the following command to run a specific test with log info:

```shell
ctest -V —R <regex>
```

where

- `-R <regex>`: runs only the tests whose names match the given regular expression.
- `-V`: enables the verbose mode, printing detailed output for each test,
  including any messages sent to stdout/stderr.

For example:

```shell
TORCH_CPP_LOG_LEVEL=INFO ctest -V -R test_triton_reduction
```

We use PyTorch Aten log as well, so you need set the env `TORCH_CPP_LOG_LEVEL=INFO`
for more logs in `libtorch_example`.

## 4. Create a PR for your code

When everything works as expected, it's time to submit a pull request (PR).
It's desirable to provide some end-to-end performance data in your PR description,
in addition to a brief summary about what the operator does.
