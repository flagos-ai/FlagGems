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

Operators that do not yet have a public FlagGems implementation can also be
injected. For example, `registry.override("sparse_csr_tensor", candidate)` makes
`flag_gems.sparse_csr_tensor(...)` call the candidate even when that attribute
did not previously exist. On restoration, newly added attributes are removed;
existing attributes are restored to their original values, including `None`.
The same behavior applies to `--override` and `--override-config`.

Adding a public callable does not add an ATen dispatcher registration. Tests
for these operators must call `flag_gems.<op_name>(...)` directly. File-loaded
candidates still fail cleanup if they were never invoked, including candidates
installed under a misspelled operator name.

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

The `--override` and `--override-config` options are already integrated into
both `tests/conftest.py` and `benchmark/conftest.py`, so any test file under
`tests/` or `benchmark/` can be pointed at a custom implementation without
code changes.

Running the existing accuracy test for `softmax` against a candidate
implementation requires no change to `test_softmax.py` itself:

```shell
pytest tests/test_softmax.py \
    --override softmax:./candidates/softmax_v2.py:my_softmax
```

Similarly, benchmark tests can use the same options:

```shell
pytest benchmark/test_reduction_perf.py \
    --override sum:./candidates/sum_v2.py:my_sum \
    --level core -s
```

The override is applied once per test session in `pytest_configure`, and
automatically restored in `pytest_unconfigure` after all tests complete.

## 5. Overrides are picked up when operators are (re-)registered

`GeneralOpRegistrar`, the class that binds each `flag_gems.<op_name>`
implementation to its ATen dispatch key, re-resolves every config entry
against the live `flag_gems` module before registering it. This means that
if you apply an override *before* `flag_gems.enable()` / `only_enable()`
runs (or before an operator is re-registered for any other reason), the
dispatch table picks up your override instead of the reference captured in
the module's internal config tuple at import time:

```python
with DynamicOpOverride() as registry:
    registry.override("softmax", my_softmax)
    flag_gems.enable()  # registers `_softmax` (and its overloads) against my_softmax
```

To keep this resolution unambiguous, the registrar looks each dispatch key
up in `flag_gems._FULL_CONFIG` — the authoritative source of registrable
ops — to find the function the key was *originally* bound to, then resolves
the live attribute by that function's name. This avoids collisions between
overloads that share a dispatch-key prefix but bind to different functions
(e.g. `_softmax` vs. `_softmax.out`).

As a consequence, if a dispatch key passed to the registrar cannot be found
in `flag_gems._FULL_CONFIG` at all, registration raises a `ValueError`
rather than guessing at a name — registering (or overriding) an operator
that was never a real registration is not allowed.

## 6. Unused overrides fail the test run

`override_from_file` (and, transitively, `override_batch_from_files` and
`--override`/`--override-config`) wraps the loaded candidate so that
`DynamicOpOverride` can track whether it was actually called. If
`restore()` or `restore_all()` runs and finds that a tracked override was
never invoked, it raises an `AssertionError` — this turns "I pointed
`--override` at the wrong operator name (or a candidate with a typo that
never gets exercised)" into a hard test failure instead of a silently
useless run:

```shell
pytest tests/test_softmax.py \
    --override softmax:./candidates/softmax_v2.py:my_softmax
# fails the session if `my_softmax` is never actually called
```

Overrides applied directly via the bare `registry.override(...)` call (as
opposed to `override_from_file`) are not tracked this way, since there is
no file-loading step to guard against typos or unreachable candidates.

## 7. Concurrent testing of multiple implementations

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

## Candidate-only profiling and external capture

`--profile-only --case-id <id>` replays exactly one case from `--list-cases`. FlagGems constructs the inputs, runs `--profile-warmup` calls and synchronizes, then runs `--profile-iterations` calls and a final synchronization inside an optional capture context. It does not call the correctness reference or collect benchmark latency.

An embedding evaluator supplies a pytest plugin object explicitly; no module-path environment variable, dynamic import of the evaluator, or special `__main__` convention is required:

```python
from contextlib import contextmanager
import pytest

class CapturePlugin:
    @pytest.hookimpl
    @contextmanager
    def pytest_flaggems_profile_scope(self, backend, case_id):
        # backend is the FlagGems vendor name; validate backend/case_id here.
        start_capture()
        try:
            yield
        finally:
            stop_capture()

pytest.main(pytest_args, plugins=[CapturePlugin()])
```

The benchmark conftest registers this first-result hook. It returns a context manager, not the result of running the candidate. Without a provider (or when providers return None), standalone pytest performs ordinary candidate-only replay without external capture. An evaluator that requires capture must independently verify that its plugin was actually entered and completed; pytest success alone is insufficient. The external evaluator owns compiler/profiler preparation, completion markers and artifacts; FlagGems has no dependency on KGS. Use a separate process for each benchmark session rather than concurrent `pytest.main()` calls in one process.

## Candidate-only preflight

Ordinary correctness and benchmark execution also use `--override`. Each correctness JSON case records `candidate_calls` observed during its test call phase. A benchmark marks `candidate_source: override` only after actually invoking the injected callable; the original `torch_op` is still timed separately as the baseline. A passing pytest session is not a substitute for candidate coverage checks. When all correctness cases are skipped, overrides are restored without an unused-candidate error; this is not correctness success, and KGS preserves `ALL_SKIP` while running applicable benchmarks.

`benchmark/` supports `--preflight-only`: reuse case enumeration and input construction, invoke the selected candidate once per case, then synchronize the device. It does not run the correctness reference, benchmark warmup/timing, or speedup calculations. One invocation means one Python operator call; compilation or autotuning inside the candidate may still launch several kernels. Passing means executable, not numerically correct or faster.

From the FlagGems checkout root, check every core case for `addmm_`:

```bash
python -m pytest -q benchmark/test_addmm_.py --level core \
  --preflight-only --override addmm_:/path/candidate.py:run \
  --record json --output /tmp/preflight.json
```

Without `--case-id`, all enumerated cases are checked. Repeat `--case-id <id>` to select a subset using IDs from `--list-cases` on the same checkout. Without an override, preflight checks the current Gems implementation. Benchmarks without case enumeration fail explicitly instead of falling back to a full benchmark.

The JSON report uses `schema_version: flaggems.preflight/v1`. Each entry in `records` contains `operator`, `nodeid`, `case_id`, whether an `override` was used, invocation `count`, and `status` (`passed`/`failed`), with an optional `error`. No latency or speedup is reported. Each run replaces the report instead of merging old results. Consumers must check the pytest exit code, complete expected-case coverage, and successful candidate injection; an empty report is not success.

This mode cannot be combined with `--profile-only`, `--list-cases`, `--query`, or nonzero `--parallel`. Unknown/unexecuted explicit case IDs, empty case plans, all-skipped runs, and candidate failures cannot pass. Continue to use `tests/` for correctness, ordinary `benchmark/` for timing, and `--profile-only` for profiling.
