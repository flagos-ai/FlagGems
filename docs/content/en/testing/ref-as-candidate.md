---
title: "Reference as candidate"
---

`--ref-as-candidate` checks whether a correctness test or benchmark can execute
using its explicitly declared reference as the candidate. It does not require a
candidate source file. The option is off by default; existing tests and override
commands keep their original behavior.

## Correctness contract

Tests remain ordinary pytest functions. Explicitly choose the candidate:

```python
from flag_gems.testing.reference import get_candidate


def test_example():
    reference = explicitly_defined_reference
    candidate = get_candidate("example", reference=reference)
    expected_inputs, candidate_inputs = build_independent_inputs()
    expected = reference(*expected_inputs)
    actual = candidate(*candidate_inputs)
    original_comparison(actual, expected)
```

Without the option, the helper reads the current public `flag_gems` export,
including external overrides. With the option, it selects the explicit reference
without looking up the real candidate; the public export may be absent.
No reference is guessed from the operator name. Missing/non-callable references
fail explicitly. Adapt differing signatures with an explicit callable.

The helper does not clone inputs, run assertions, adjust dtypes or move devices.
Retain the original input preparation, independent buffers for in-place calls,
CPU/upcast expected-side inputs, tolerances and mutation/alias assertions.
The former `ReferenceCase` / `@reference_test` factory API is removed.

`negative` is the initial migrated correctness test:

```sh
python -m pytest tests/test_negative.py -m negative --ref-as-candidate
python -m pytest tests/test_negative.py -m negative --ref-as-candidate --ref cpu
```

Other tests need an explicit `get_candidate` declaration to opt in; their normal
execution is unchanged. A passing node must actually call the selected candidate
in its test body; calls only in fixtures do not qualify. Missing declarations or
caught contract errors cannot be hidden by skip/xfail. A capability skip after
valid selection remains a skip; an all-skip run is not successful validation.
Setup/teardown contract failures are preserved in correctness JSON.

JSON records include `candidate_source: reference`, `reference_calls` and
`reference_candidate_calls`. Counters observe calls through the helper only;
direct expected-side calls are not counted.

## Benchmark contract

Standard `Benchmark` subclasses already declare their reference as `torch_op`.
With the new option, that callable is also used for candidate timing, outside
Gems dispatch. Input generation, selected cases, timing mode and metrics remain
in use. The candidate latency metric is required; records identify
`candidate_source: reference`.

```sh
python -m pytest benchmark/test_add.py -m add --level core --ref-as-candidate
```

A non-callable `torch_op` or a custom override of `run`, `_measure_input`,
`get_latency` or `_candidate_call` is an explicit contract error in this mode.
The original behavior of those benchmarks is unchanged without the option.

The option cannot be combined with external overrides, `--reference-only`,
`--preflight-only`, `--profile-only`, `--list-cases`, `--query`, `--parallel`, `--collect-marks`, collection-only or xdist parallel execution. Those modes retain their existing meanings. This mode performs
real timing; `--reference-only` still performs its original baseline-only probe.

Ref-vs-ref is a harness executability check, not independent evidence that a
reference is correct or that external candidate injection works. Continue to
use dummy candidate injection to check the actual override route. These timings
are diagnostic and do not replace the H800 ops-baseline collection workflow.
