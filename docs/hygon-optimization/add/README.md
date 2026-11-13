# Hygon add optimization

Independent official-master baseline `010d8083b0dc89ab795aee973ad62c83c9d2592d`, measured source `baa3719bd423bec0c179af709dc8404f42ca655b`, date2026-11-13. All200 cases and all6 roles complete;old/new/DAS correctness200each,76 focused hardware tests pass.63 correct/stable old-new pairs,29 gains above5%,range0.963901-5.080014x;all29 gains have actual specialized-source/kernel/count witnesses.91 qualified new/DAS pairs:70 meet90%,21 miss,109 comparisons remain A/A-inconclusive. No qualified latency regression above5%.

The89 original model contracts have30 qualified old/new pairs and8 gains above5%;42 qualified new/DAS pairs include33DAS90 and9misses,47 comparisons remain inconclusive. Everycase remains in [complete evidence](performance-master010-20261113/README.md),including unchanged fallbacks,misses and unstable points. This replaces all200 previous-source cases regardless outcome,without pooling samples or selecting favorable repeats.

All3 preregistered generic-guard domains now meetDAS90. Two old/new ratios qualify(1.000793x and0.963901x);the INT64 point remains old-control A/A-inconclusive,so no confirmed three-case recovery is claimed. [Attribution and guard assessment](performance-master010-20261113/specialization-attribution.json) contains exact source and kernel event counts. Earlier [serial diagnostics](diagnostic-serial-20261113/README.md) and [concurrent diagnostics](diagnostic-concurrent-20261112/README.md) retain all earlier failures and limitations.

```bash
pytest tests/test_hygon_add.py --ref cpu -q
python docs/hygon-optimization/add/performance-master010-20261113/verify_results.py
python docs/hygon-optimization/add/performance-master010-20261113/test-accounting-boundaries.py
pre-commit run --all-files
```

Exact ATen->privatePrivateUse1->realHygon provider->specialized/generic kernel witnesses;full frozen output and backing/geometry/known-alias checks outsideROI. Generated metadata-preserving inputs are not original model payload replay. Six sequential roles,2x25 device samples/role,20calls/ROI,both block drifts<=5%. Producer-linked kernel sums;separate50-call synchronized API controls are diagnostic. Eager,no fusion/model-throughput claim. V5 preserves all events/timestamps and strict producers/device bounds;only final nonproducing waits audited separately.9 portable boundary counterexamples and exact accountant sources included. Hygon gfx936,Torch2.10/Torch-FL R7,FlagTree HIP/gfx936. Latest-head requiredCI and protected Feishu readback gate Ready conversion.
