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


## Audit and CI availability

Process-level A/A repeats and offline verification were performed in stages by one agent; no independent-person signoff is claimed. The32-case correctness and all-role raw measurements are separate from CI scheduling.

The Hygon job in CI run43198954642 failed before operator tests because its runner timed out waiting for an available DCU. It is classified as infrastructure failure rather than operator correctness failure. A subsequent evidence update triggers a fresh latest-head check; this PR remains Draft until that check succeeds. The original failed CI logs and fixed-source hardware raw are preserved outside git.

## Supplementary synchronized API controls

`api-controls.json` provides all50 unprofiled samples for old/new/DAS/new-repeat roles across all32 cases, original control-file hashes and profiler-exclusion proofs. `api-case-summary.csv` lists complete synchronized API medians beside the primary device ratios and qualification flags. API time includes dispatch, allocation, launch and synchronization; it is not kernel duration or model throughput. It is supplementary diagnostic evidence and has no separately frozen API acceptance gate.

For example, `model285-56cb9af233fe717d3249` has a qualified device old/new ratio of21.422948x. Its API old/new medians are256.709000/116.177000us, a diagnostic API ratio of2.209637. Do not substitute this API ratio for device acceptance or extrapolate the kernel ratio to model throughput. All prior32-case device correctness, A/A and90% counts remain unchanged.
