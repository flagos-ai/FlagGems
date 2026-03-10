# Add a C++ wrapper

To add a c++ wrapper, you need to first build FlagGems with C++ extensions enabled.
Please refer to [Installation](./installation.md).

## Write the wrapper

Follow the following steps to add a new C++ wrapped operator:

- Add a function prototype for the operator in the `include/flag_gems/operators.h` file.
- Add the operator function implementation in the `lib/op_name.cpp` file.
- Change the cmakefile `lib/CMakeLists.txt` accordingly.
- Add python bindings in `src/flag_gems/csrc/cstub.cpp`
- Add the `triton_jit` function in `triton_src`.
  Currently we use a dedicated directory to store the `triton_jit` functions
  In the future, we will reuse the `triton_jit` functions in Python code under `flag_gems`.

## Write test case

FlagGems uses `ctest` and `googletest` for C++ unit tests.
After having finished the C++ wrapper, a corresponding C++ test case should be added.
Add your unit test in `ctests/test_triton_xxx.cpp` and `ctests/CMakeLists.txt`.
Finally, build your test source and run it with [C++ Tests](./ctest_in_flaggems.md).

## Create a PR for your code

When submitting a PR, it's desirable to provide end-to-end performance data
in your PR description.
