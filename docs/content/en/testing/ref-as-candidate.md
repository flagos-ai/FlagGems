---
title: "Reference as candidate"
---

`--ref-as-candidate` runs an explicitly declared reference on the candidate side
of correctness tests and standard benchmarks. It defaults to false.

## Correctness

Tests remain ordinary pytest functions. Select the candidate explicitly:

```python
from flag_gems.testing.reference import get_candidate


def test_example():
    candidate = get_candidate("example", reference=explicit_reference)
    expected_inputs, candidate_inputs = build_independent_inputs()
    expected = explicit_reference(*expected_inputs)
    actual = candidate(*candidate_inputs)
    original_comparison(actual, expected)
```

Normal mode resolves the current public `flag_gems` export, including overrides.
Reference mode uses the supplied callable without looking up the real candidate.
The callable must accept the candidate's arguments; no reference is inferred
from the operator name. Input cloning, CPU/upcast reference preparation and all
original assertions remain the test author's responsibility.

`negative` is the initial migrated test:

```sh
pytest tests/test_negative.py -m negative --ref-as-candidate
pytest tests/test_negative.py -m negative --ref-as-candidate --ref cpu
```

Selected tests must declare a callable reference and invoke it on the candidate
side in the test body. Fixture-only calls do not qualify. Missing declarations
or caught contract errors cannot be hidden by skip/xfail; an all-skip run fails.
Existing correctness JSON records the candidate source, helper-observed call
counts and setup/teardown contract failures. Direct expected-side calls are not
included in those counts.

## Benchmark

Standard benchmarks use their declared `torch_op` as the candidate outside Gems
dispatch. The existing inputs, timer and metrics are retained. Each candidate
timing must invoke that callable; records identify `candidate_source: reference`.
Custom execution paths fail explicitly when they cannot satisfy this contract.

```sh
pytest benchmark/test_add.py -m add --level core --ref-as-candidate
```

The mode is incompatible with override, reference-only, preflight, profiling,
case listing/query, marker collection, collection-only and parallel execution.
`--reference-only` remains the separate baseline-only probe. Reference-vs-reference
validates harness execution; it does not independently establish numerical
correctness or verify external candidate injection.
