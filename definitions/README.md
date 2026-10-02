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

Definitions use Protocol v6.0 (`api_version: "v6.0"`) and contain these fields:

| Field | Meaning |
| --- | --- |
| `api_version` | Definition protocol version. |
| `name` | Exact operator name. |
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

## Updating definitions

When adding an operator test suite, include its definition in the same PR.
Update the definition when the tested contract changes, and derive descriptions
from the tests and ATen schema rather than inferring guarantees from a single
reference comparison. Keep runtime validation evidence outside the JSON files.

## KernelGen Server import

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
