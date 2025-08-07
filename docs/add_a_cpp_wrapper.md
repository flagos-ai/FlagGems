# Add A C++ wrapper

To add a c++ wrapper, you need to build FlagGems with C extensions first. Please refer to [Installation](./installation.md).

## Write the Wrapper

Add function declaration of the operator in `include/flag_gems/operators.h`

Add implement of the function in `lib/op_name.cpp`

Change the cmakefile `lib/CMakeLists.txt`

Add python bindings in `src/flag_gems/csrc/cstub.cpp`

Add the triton_jit function in `triton_src`, currently we use a new dir to store the triton_jit functions and later will reuse the triton_jit functions in flag_gems python code.

## Write the Test

FlagGems uses ctest and googletest for c++ unnit test.
After finish the c++ wrapper, a corresponding c++ test should be added as well.
Add your test in `ctests/test_triton_xxx.cpp` and `ctests/CMakeLists.txt`
Just build your test and run it with [C++ Tests](ctest_in_flaggems.md)

## Push your code

Its better to have an end-to-end performance data in the PR description.
