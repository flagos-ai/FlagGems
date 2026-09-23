---
title: Precision Checking (Experimental)
weight: 46
---

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


# Precision Checking (Experimental)

FlagGems provides an experimental precision-checking mechanism that
automatically compares the output of FlagGems operators against native
PyTorch (CPU) results, and logs any discrepancies to a file.
This is useful for verifying numerical correctness during development.

## How to Enable

Enabling precision checking requires two steps:

1. Call `enable_precision_check()` from `flag_gems.logging_utils` to
   configure the precision logger.
2. Pass `PrecisionCheckRegister` as the `registrar` parameter to
   `enable()` or `only_enable()`, so that operators are wrapped with
   precision-checking logic.

```python
import flag_gems
from flag_gems.logging_utils import enable_precision_check
from flag_gems.runtime.precision_register import PrecisionCheckRegister

# Step 1: Configure precision checking (initialize the precision logger)
enable_precision_check()

# Step 2: Register all operators with PrecisionCheckRegister
flag_gems.enable(registrar=PrecisionCheckRegister)

# Run your model or operators as usual
output = model(input)
```

You can also use it with `only_enable()` to check specific operators:

```python
from flag_gems.logging_utils import enable_precision_check
from flag_gems.runtime.precision_register import PrecisionCheckRegister

enable_precision_check(rtol=1e-3, atol=1e-4)
flag_gems.only_enable(
    include=["mm", "add", "softmax"],
    registrar=PrecisionCheckRegister,
)
```

## Configuration

You can customize the precision checking behavior by passing parameters
to `enable_precision_check()`.

| Parameter      | Type        | Default                        | Description                                         |
| -------------- | ----------- | ------------------------------ | --------------------------------------------------- |
| `rtol`         | `float`     | `1e-4`                         | Relative tolerance                                  |
| `atol`         | `float`     | `1e-5`                         | Absolute tolerance                                  |
| `max_checks`   | `int`       | `10`                           | Max checks per operator before skipping             |
| `log_once`     | `bool`      | `True`                         | Only log the first failure per operator             |
| `path`         | `str`       | `~/.flaggems/precision.log`    | Log file path                                       |

```python
from flag_gems.logging_utils import enable_precision_check

enable_precision_check(
    rtol=1e-3,
    atol=1e-5,
    max_checks=20,
    path="./my_precision.log",
)
```

## Disabling

To disable precision checking at runtime:

```python
from flag_gems.logging_utils import disable_precision_check

disable_precision_check()
```

## Log Output

Precision check results are written to `~/.flaggems/precision.log` by default.
Only operators that fail the tolerance check will be logged.

Sample log content:

```shell
$ cat ~/.flaggems/precision.log
2025-05-19 10:00:01 [WARNING] Op: add.Tensor | FAIL | in: [(2, 3):torch.float16] | out: (2, 3):torch.float16 | max_abs: 1.200000e-03 | max_rel: 2.500000e-02 | rtol=0.01, atol=0.01
```

## Behavior Details

The precision checker has several built-in safeguards to minimize
performance impact:

- Only the first N calls per operator are checked (controlled by `max_checks`)
- Tensors larger than 1M elements are skipped to avoid copy overhead
- Once an operator logs a failure, it will not be checked again
- Pure layout/memory ops (`clone`, `view`, `copy_`, etc.) are automatically skipped
- Random sampling ops (`uniform_`, `normal_`, etc.) are automatically skipped
- `.out` variant operators are skipped
- For `float16` / `bfloat16` inputs, tolerance is automatically relaxed to at least `1e-2`

## How It Works

When `PrecisionCheckRegister` is used as the registrar, each operator
is wrapped with a precision-checking decorator. The wrapper:

1. Executes the FlagGems (GPU) implementation normally.
2. Copies inputs to CPU and runs the native `aten` operator as reference.
3. Compares the two results using the configured tolerance.
4. Logs a warning if the results diverge beyond tolerance.

> [!WARNING]
> Precision checking copies GPU tensors to CPU and runs native PyTorch
> computation as a reference. This incurs significant performance overhead.
> This feature is intended for development and debugging only — do not
> enable it in production.
