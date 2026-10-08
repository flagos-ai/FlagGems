# Operator definitions

Each `<operator>.json` describes a representative ATen signature and the
semantics exercised by the corresponding FlagGems tests:

- `tests/test_<operator>.py`: correctness coverage and the full calling contract.
- `benchmark/test_<operator>.py`: benchmark cases.
- `definitions/<operator>.json`: structured metadata for consumers such as
  KernelGen Server.

Keep the exact operator name in the filename and the JSON `name` field,
including leading underscores. Pytest marker naming is a separate convention.

## Contract

Definitions retain the Protocol v6.0 ABI fields (`api_version: "v6.0"`) and
include repository planning metadata:

| Field | Meaning |
| --- | --- |
| `api_version` | Definition protocol version. |
| `name` | Exact operator name. |
| `requires_triton_kernel` | Required boolean planning metadata: whether the tested operator contract includes tensor computation or materialization requiring kernel work. |
| `description` | Representative signature, supported schema variants, and test-backed semantics. |
| `parameters` | Ordered parameters, calling kinds, required flags, type hints, and optional defaults. |
| `outputs` | Logical return values; a tensor list is one logical return. |
| `effects` | Mutation and alias metadata for the representative signature. |

A definition is not an exclusive ABI. The original pytest tests remain
responsible for overloads, positional and keyword arguments, `out=`, errors,
return identity, mutation, and aliasing. Do not remove or narrow test cases to
fit the representative signature. Required parameters omit `default`; optional
parameters include their default, including an explicit `null` when applicable.
Schema aliasing does not by itself guarantee Python object identity.

Definitions describe the test contract; their presence does not imply that a
FlagGems implementation exists or that every backend supports the operator.

## Kernel requirement

`requires_triton_kernel` describes implementation work, not whether an operator
already has an implementation, is profitable to optimize, or can run on the
selected Triton backend. It is a JSON boolean, never a string, integer or null.

- `true`: the contract includes arithmetic, reductions, index-content checks,
  value generation, dtype conversion, copying or materializing tensor data.
  CPU-only and vendor-specific numerical operations still belong here; device
  support and opaque-library ABI compatibility need separate assessment.
- `false`: the tested operation only handles views, metadata, scalar/schema
  checks, uninitialized allocation, or framework/runtime management. Host
  pinning, device-to-host transfer and autograd-engine invocation belong to the
  runtime; they do not themselves define a Triton compute kernel.

Review every tested overload and path, including `out=`, noncontiguous inputs
and lazy conjugate/negative values. One compute/materialization path makes the
operator `true`, even if other paths return an alias. For example, `view` is
`false`, while `reshape_as` is `true` because noncontiguous inputs can require a
copy. `_efficientzerotensor` is `true` because its tested `out=` path fills an
existing buffer, despite the default lazy-zero result. Quantizer array getters
with tested `out=` copies also differ from scalar quantizer metadata queries.

Generic autograd support for a view does not make its forward a compute
operation. An explicitly tested operator-specific numerical backward does
count, such as the extra gradient addition in
`_test_autograd_multiple_dispatch_view`. Neither a void/scalar return nor an
internal/test/backend name establishes that an operator is metadata-only.

This flag must not be used to skip correctness cases, replace their reference,
or bypass an unsupported ABI. Reassess it when test coverage changes; `false`
is not a claim about every possible future overload or workload.

## Updating definitions

When adding an operator test suite, include its definition in the same PR.
Update the definition when the tested contract changes, and derive descriptions
from the tests and ATen schema rather than inferring guarantees from a single
reference comparison. Keep runtime validation evidence outside the JSON files.
Always include `requires_triton_kernel` and explain its classification in the
PR, citing the relevant test paths. The publication contract test checks that
every definition supplies a boolean, but does not establish semantic accuracy.

## KernelGen Server import

`requires_triton_kernel` is a repository planning extension, not a change to
the Protocol version or the callable ABI. Consumers must explicitly support
this field before loading these files unchanged. In particular, older strict
`Definition` models with `extra="forbid"` reject it. Such importers must either
add support or extract this planning metadata before validating the remaining
Protocol v6.0 definition. Do not silently default a missing flag to `false`.
The `flaggems_definitions` capability alone does not establish support for this
new field; validate the actual consuming version before adopting the catalog.

Recent KernelGen Server versions expose the logical Catalog `flaggems`, which
reads this directory directly from the same selected checkout as pytest and
benchmark. Use `kg run --catalog-name flaggems --definition <operator>` with a
Server advertising the `flaggems_definitions` capability. No copying into the
Server repository is needed. The original 70 KernelGen test operators are now
published here alongside the newer definitions; existing experiment snapshots
remain unchanged.

The following manual layout is only for consumers that still require a
standalone Catalog:

This directory stores portable definition files, not a complete Server Catalog.
To import them into a FlagGems adapter Catalog, copy the JSON files into the
Catalog's `definitions/` directory and provide this `manifest.json` at the
Catalog root:

```json
{
  "api_version": "v6.0",
  "evaluator": "flaggems",
  "benchmark_level": "core"
}
```

Check for duplicate operator names before importing; do not overwrite existing
definitions silently. Validate the resulting Catalog with the consuming Server
version. The FlagGems checkout used for evaluation must provide the corresponding
pytest and benchmark files. Catalog loading validates metadata and does not
replace runtime validation of those tests.
