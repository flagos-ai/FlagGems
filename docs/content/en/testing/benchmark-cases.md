---
title: Listing and replaying benchmark workloads
weight: 30
---

# Benchmark Case API

`pytest --collect-only` lists Python test nodes, not the shape/dtype/parameter combinations generated inside a benchmark. The Case API separates lightweight workload planning from input construction: `get_case_iter(dtype)` describes cases without allocating tensors, and `build_inputs(case)` materializes only the workloads selected for execution.

This feature is independent of candidate injection. It does not add `--override`, a resolver, candidate coverage reports, Preflight, or Profile modes. Existing direct `gems_op` calls, `use_gems()` dispatch, correctness tests and timing methods are unchanged.

## List, then replay

Run from the repository root in an existing FlagGems test environment:

```bash
python -m pytest benchmark/test_softmax.py::test_softmax --level core --dtypes float32 --list-cases --output /tmp/softmax-cases.json
```

The output uses `flaggems.benchmark-case-list/v2`. The top-level `benchmarks` array contains an entry per benchmark, with `op_name`, `phase="timing"`, `level` and `cases`. Each case has `case_id`, `ordinal`, `dtype`, `shape` and `params`. Private builder state is never serialized. Listing overwrites the specified report instead of merging stale cases from a previous invocation.

Copy an exact `case_id` from that file:

```bash
python -m pytest benchmark/test_softmax.py::test_softmax --level core --dtypes float32 --case-id 'benchmark/test_softmax.py::test_softmax::core::float32::0' --record json --output /tmp/softmax-replay.json
```

Repeat `--case-id` to select multiple workloads. Only selected inputs are constructed and measured; normal benchmark metrics include the selected `case_id`. The reference and Gems implementation are both measured using the existing benchmark path. Unknown IDs and requested-but-unexecuted IDs fail the session, including deselected or skipped test nodes. Duplicate CLI IDs and combining `--list-cases` with `--case-id` or `--query` are usage errors.

IDs are `pytest_nodeid::level::dtype::ordinal`, not content hashes. Keep the checkout, pytest root, shape file, dtype filters, level, case ordering and environment-dependent shape filters fixed between listing and replay. Changing the workload recipe can change what an ordinal means. Input tensor values are generated at execution time; the case list records the workload recipe, not frozen random bytes.

The planning API does not allocate input tensors or run operators, reference code or timing. Importing FlagGems and third-party benchmark modules may still require an installed device runtime; this is not a promise that every benchmark module imports on a CPU-only host. Test authors must also avoid materializing tensors before `bench.run()`.

## Supported providers and legacy benchmarks

Common unary, binary, scalar-binary, unary-out, reduction, BLAS and TransformerEngine GLU benchmark families provide case descriptions. Subclasses that replace the input loop must explicitly provide their own planning/building methods; inheriting a family name alone does not make a custom loop enumerable. BLAS case factories must yield exactly one input tuple per planned shape/layout; extra or missing tuples fail instead of silently dropping workloads.

Unmigrated `GenericBenchmark(input_fn=...)` and custom loops retain normal benchmark execution. Listing or selecting cases on those benchmarks raises a clear unsupported-provider error. This PR does not migrate every operator-specific benchmark.

`GenericBenchmark` accepts a two-stage pair instead of `input_fn`:

```python
import pytest
import torch

from benchmark import base, consts

def case_fn(shape, dtype):
    yield consts.BenchmarkCasePlan(
        shape={"input": shape},
        params={"dim": -1},
        builder_args=(shape,),
    )

def build_inputs_fn(plan, dtype, device):
    x = base.generate_tensor_input(plan.builder_args[0], dtype, device)
    return x, plan.params["dim"]

@pytest.mark.softmax
def test_softmax_cases():
    bench = base.GenericBenchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        case_fn=case_fn,
        build_inputs_fn=build_inputs_fn,
    )
    bench.run()
```

The two callbacks must be supplied together and cannot be combined with legacy `input_fn`. A builder returns one original input tuple, including any kwargs dictionary; it is not a generator. Planning must preserve all original shapes, dtypes, scalar parameters, layouts and workload order. Do not enumerate by running the old tensor generator and discarding its outputs.

## Validation

The serialization contract can be tested without Torch/Triton:

```bash
PYTHONPATH=. python -m pytest --confcutdir=tests/core tests/core/test_benchmark_case_contract.py
```

In the normal FlagGems environment, run `python -m pytest benchmark/test_benchmark_case_api.py` for provider tests, then verify real listing and single-case replay on the target device. Candidate-injection integration should be tested separately after the corresponding feature is available.
