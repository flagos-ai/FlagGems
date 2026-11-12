# Hygon batched matrix multiplication

The current independent verification uses official master `010d8083b0dc89ab795aee973ad62c83c9d2592d` and measured candidate `b4c330effb09e546456d0094fab525edabf64f93`. Measurement date: **2026-11-13**. This supersedes the earlier9a and preguard010 measurements for acceptance.

All32 declared cases passed full-output old/new/DAS correctness and complete six-role device capture. Focused hardware tests:62 passed. Fifteen old/new pairs pass correctness and5% A/A gates;10 improve by more than5%,range0.967090-10.120444x. Sixteen new/DAS pairs qualify;15 meet90%. The other16 pairs remain A/A-inconclusive. There is no qualified old/new slowdown exceeding5%.

The remaining qualified miss `model285-565362c5211ae608a0b1` has DAS/new0.881283 on the upstream fallback. ESM-2 `model285-809a8e8ca10e8c6cf48b` recovered from a qualified0.914467x regression to1.009804x old/new,0.954441 DAS/new. The narrow DiffDock case remains A/A-inconclusive;its point is not accepted as a confirmed fix. All32 whole cases were replaced after the guard regardless outcome,no sample pooling.

[Current complete evidence](performance-master010-20261113/README.md) includes every case,sample,date,exact source/ATen/kernel witness and accounting verifier. [Preguard010 diagnostics](diagnostic-master010-r2-20261113/README.md) preserve the regression motivating the guard. [Older9a evidence](performance-20261112/README.md) is historical and does not measure the optimized010 baseline.

## Reproduction and execution chain

```bash
pytest tests/test_hygon_bmm.py -q --ref cpu
python docs/hygon-optimization/bmm/performance-master010-20261113/verify_results.py
python docs/hygon-optimization/bmm/performance-master010-20261113/test-accounting-boundaries.py
pre-commit run --all-files
```

Generated inputs restore known storage geometry/aliases and disclose uncaptured assumptions;this is not original model payload replay. Public `aten::bmm` -> private PrivateUse1 registration -> actual Hygon source -> `_small_bmm` or optimized upstream fallback is checked per case. DAS runs an independent native BLAS process. Six roles each contain2x25 device ROIs,20calls/ROI;producer-linked kernel sums define device latency. Both block drifts must be<=5%.50-call unprofiled API controls are separate wrapper/allocation/launch/synchronization diagnostics,not model throughput. All original trace shards and source hashes remain audit-linked. Final nonproducing host wait boundary overruns are separately recorded;device/producer containment remains strict,no timestamp shifting or trimming. Hardware: Hygon gfx936,Torch2.10/Torch-FL R7,FlagTree HIP/gfx936. No graph fusion or independent-person approval claim.
