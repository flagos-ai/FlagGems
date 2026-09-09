---
title: Testing with Dynamic Operator Override
weight: 30
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


# Testing with Dynamic Operator Override

Most accuracy tests in `tests/` call *FlagGems* operators directly, for
example `flag_gems.softmax(x)` or `flag_gems._list_to_tensor(...)`. When you
are iterating on a new kernel implementation, or comparing several candidate
implementations side by side, it is often inconvenient to edit the operator
source file under `src/flag_gems/ops/` for every variant you want to try.

The `flag_gems.dynamic_registry` and `flag_gems.cli_override` modules solve
this by letting a test process swap out the implementation bound to a
`flag_gems.<op_name>` attribute at runtime, without touching the operator's
source code. This makes it possible to:

- Point a test run at an implementation living in an arbitrary `.py` file.
- Drive the override from the command line or from a YAML/JSON config file,
  so no test code needs to change between runs.
- Run several such test processes concurrently, each overriding the same
  operator with a different implementation, without interfering with each
  other.

## 1. Overriding an operator from Python

`DynamicOpOverride` tracks the original implementation of every operator it
touches, so it can restore it later. The recommended usage is as a context
manager, which restores all overrides automatically on exit — including when
the test body raises:

```python
import torch
import flag_gems
from flag_gems.dynamic_registry import DynamicOpOverride

def my_softmax(input, dim=-1, dtype=None):
    return torch.softmax(input, dim=dim, dtype=dtype)

with DynamicOpOverride() as registry:
    registry.override("softmax", my_softmax)
    x = torch.randn(10, 10, device=flag_gems.device)
    result = flag_gems.softmax(x)  # uses my_softmax
# original flag_gems.softmax is restored here
```

Without the context manager, call `restore()` or `restore_all()` explicitly:

```python
registry = DynamicOpOverride()
registry.override("softmax", my_softmax)
...
registry.restore("softmax")     # restore a single operator
registry.restore_all()          # or restore everything at once
```

## 2. Loading an implementation from a standalone file

Rather than defining the replacement inline, `override_from_file` loads a
function from any `.py` file on disk — the file does not need to live inside
the *FlagGems* package or be importable as a module:

```python
registry.override_from_file(
    op_name="softmax",
    filepath="./candidates/softmax_v2.py",
    func_name="my_softmax",  # defaults to op_name when omitted
)
```

Multiple operators can be swapped in one call with
`override_batch_from_files`:

```python
registry.override_batch_from_files({
    "softmax": ("./candidates/softmax_v2.py", "my_softmax"),
    "rms_norm": "./candidates/rms_norm_v2.py",  # function name == op name
})
```

## 3. Driving overrides from the command line

`flag_gems.cli_override` exposes `add_override_arguments()` and
`apply_overrides_from_args()`, which add a consistent `--override` /
`--override-config` interface to any `argparse`-based script or `pytest`
`conftest.py`.

### `--override op_name:filepath[:func_name]`

Pass one or more `--override` flags, each pointing at the implementation file
for one operator:

```shell
python my_test.py \
    --override softmax:./candidates/softmax_v2.py:my_softmax \
    --override rms_norm:./candidates/rms_norm_v2.py
```

The `op_name=filepath[:func_name]` form is also accepted, if you prefer `=`
over `:` as the top-level separator.

### `--override-config path/to/overrides.yaml`

For a larger set of overrides, collect them in a YAML (or JSON) file instead:

```yaml
overrides:
  softmax:
    file: ./candidates/softmax_v2.py
    function: my_softmax
  rms_norm: ./candidates/rms_norm_v2.py       # function name == op name
  layer_norm: ./candidates/layer_norm_v2.py:my_layer_norm
```

```shell
python my_test.py --override-config ./overrides.yaml
```

`--override` and `--override-config` can be combined; entries from
`--override` are applied after the config file, so they take precedence for
any operator listed in both places.

## 4. Integrating with `pytest`

Add the CLI options to `conftest.py` and apply them once per session, so any
test file under `tests/` can be pointed at a custom implementation without
code changes:

```python
# tests/conftest.py
import pytest
from flag_gems.dynamic_registry import DynamicOpOverride
from flag_gems.cli_override import add_override_arguments, apply_overrides_from_args

def pytest_addoption(parser):
    add_override_arguments(parser)

def pytest_configure(config):
    config._override_registry = apply_overrides_from_args(config.option)

def pytest_unconfigure(config):
    if hasattr(config, "_override_registry"):
        config._override_registry.restore_all()
```

Running the existing accuracy test for `softmax` against a candidate
implementation then requires no change to `test_softmax.py` itself:

```shell
pytest tests/test_softmax.py \
    --override softmax:./candidates/softmax_v2.py:my_softmax
```

## 5. Concurrent testing of multiple implementations

Because each `DynamicOpOverride` only ever mutates attributes of the
already-imported `flag_gems` module *within its own process*, independent
`pytest` invocations — run in separate OS processes — can each load a
different candidate implementation for the same operator without conflict:

```shell
pytest tests/test_softmax.py --override softmax:./variant_a.py &
pytest tests/test_softmax.py --override softmax:./variant_b.py &
pytest tests/test_softmax.py &   # baseline, no override
wait
```

This is useful for A/B-testing kernel variants, or for running the full test
suite against several candidate implementations in a CI matrix, without
maintaining separate copies of the operator source tree.

> [!WARNING]
> **Warning**
>
> A `DynamicOpOverride` instance overrides attributes on the shared
> `flag_gems` module object. Within a *single* process, overrides from
> different `DynamicOpOverride` instances (or different threads) are not
> isolated from each other — the most recent `override()` call wins, and
> `restore_all()` on one registry only restores the operators it itself
> overrode. Prefer one registry per process/test session, scoped with the
> `with` statement, to keep behavior predictable.
