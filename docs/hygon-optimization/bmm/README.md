# Hygon `aten::bmm` optimization

The isolated branch targets official master. The independent hardware control is pinned to `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`; tested candidate production commit: `60b01904510660fc692d02104d887837d41abbab`.

## Correctness and performance

On 2026-11-12 all **32 declared reconstructed workloads passed new-FG full-output correctness**, with complete old/new/DAS timing and independent repeats. No new numerical regression was observed. Eleven old/new pairs meet the correctness and 5% A/A gates; their speedups range from approximately **1.00x to 21.42x**. Fifteen new/DAS pairs qualify independently; all fifteen meet DAS 90%. The remaining pairs have explicit A/A failure flags; their point ratios are diagnostic, not accepted speedups or attainment claims.

The full per-case input metadata, assumptions, all samples and trace/source fingerprints are in [performance-20261112](performance-20261112/README.md). Inputs use fixed generated values and disclosed effective geometry/aliases. These measurements do not establish replay acceptance for uncaptured original model operands. Historical internal results are separate from this independent official-master comparison.

```bash
pytest tests/test_hygon_bmm.py -q --ref cpu
python docs/hygon-optimization/bmm/performance-20261112/verify_results.py
```

The official Hygon CI and required rule checks passed for the production source. Raw profiler binaries remain outside this repository.

## Bottom-layer execution path

Public `aten::bmm` uses a process-private PrivateUse1 binding to the selected FG export; worker traces verify the exact executed source entry and schema. Supported cases enter the Hygon provider, while unsupported layouts/dtypes retain the generic provider. `kernel-witness.json` lists the actual producer-linked device kernels per case and role. DAS executes in an independent process with no FG source frames. Full input backing hashes match across all providers. Device samples sum attributed kernel durations per ROI and divide by the number of calls; API, compiler/autotune, and memory-copy durations are tracked separately. Execution is eager, without graph fusion.
