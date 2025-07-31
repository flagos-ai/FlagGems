# C++ Tests in FlagGems

If you build FlagGems with C extensions with `FLAGGEMS_BUILD_CTESTS` cmake option `ON`, you can run the ctest in the dir `FlagGems/build/cpython-3xx` with command:

```bash
ctest .
```

This will run all the test files under `FlagGems/ctests`

Use `ctest -V —R xxx_test` for a specific test with log info, where

- `-R <regex>`: Runs only the tests whose names match the given regular expression.
- `-V`: Enables verbose mode, printing detailed output for each test, including any messages sent to stdout/stderr.

For example:

```bash
TORCH_CPP_LOG_LEVEL=INFO ctest -V -R test_triton_reduction
```

we use pytorch aten log as well, so you need set the env `TORCH_CPP_LOG_LEVEL=INFO` for more logs in `libtorch_example`.
