# Independent Hygon clone performance verification

Measured2026-11-12 on Hygon gfx936, Torch2.10 + Torch-FL R7, FlagTree HCU. Official master `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`; isolated candidate `cb05d32f9facefd6f3f5c1266ef48ee81f4459a5`.

All32 declared cases pass full numerical and clone layout/alias semantics. Complete timing32/32.10 focused hardware unit tests also pass.18 independent old/new pairs qualify;11 improve over5%, the maximum speedup is1.713688x, and none has a slowdown over5%.22 independent new/DAS pairs qualify;17 meet90%. Other pairs are A/A-inconclusive and remain diagnostic, not accepted performance gains.

The initial candidate regressed large FP64 noncontiguous-to-contiguous conversion. The tested fix retains the generic copy implementation for FP64 noncontiguous-to-contiguous inputs with at least2**20 elements. Final full32 verification includes these cases and found no accepted old/new regression over5%.

The following five qualified cases still miss the DAS90% target. They remain explicit misses; their near-baseline old/new ratios are not presented as optimization gains.

| Case | Old/New | DAS/New |
|---|---:|---:|
| model285-81dbc2ff6a267a82f4fd | 1.068048 | 0.812307 |
| model285-f37d11eae1496a95d09a | 0.974980 | 0.824275 |
| model285-67f6f5045bb688faa487 | 1.016984 | 0.743181 |
| model285-815cba298e552108b019 | 0.995438 | 0.817221 |
| model285-8af7bbd2c78439e5b9a9 | 1.014599 | 0.733863 |

## Measurement and reproducibility

The primary metric is **complete producer-linked device kernel plus copy duration**, divided by calls per ROI. It includes memcpy rather than comparing a copy-only native path with a kernel-only denominator. Each provider and independent repeat has2x25 samples,100 calls/ROI. Both block-mean drifts must be<=5%. All device events and every trace shard are retained; compile/API time/output transfers lie outside the timed region. No pooling or filtering.

Inputs reconstruct frozen metadata and seeds, not original model payloads. Cases.json discloses offsets/aliases/strides and uncaptured metadata assumptions. Output layout/preserve-format, distinct storage, complete exact values and input immutability remain checked. Full sample arrays, source/backing/schema/shard hashes and actual kernel/copy witnesses are published; raw profiler binaries are retained privately.

```bash
pytest tests/test_hygon_clone.py -q --ref cpu
python docs/hygon-optimization/clone/performance-20261112/verify_results.py
pre-commit run --all-files
```

## Bottom-layer execution path

Reconstructed CPU backing storage transfers through Torch-FL to flagos tensors. `aten::clone` enters exact process-private PrivateUse1 registration, then the Hygon clone provider or guarded generic fallback. Copy-only native paths and Triton kernels are both attributed to the originating worker and ROI; asynchronous activity ends before synchronization and full-output checks. Shape/stride/offset/preserve-format and output storage independence are checked against the declared clone contract. No Dynamo/Inductor fusion is active.
