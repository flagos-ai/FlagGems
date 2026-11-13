# Hygon dtype conversion and `_to_copy`

Route the exact Hygon ATen overload through the process-private PrivateUse1 provider. Preserve full backing/geometry and output-layout semantics, correct nonzero floating-to-bool conversion and use guarded dense conversion kernels. Generic paths remain for measured slower medium BF16/FP32 conversion ranges.

## Official-master branch validation

Pinned baseline `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`; tested source `2862d83d0d8cd7d599f2059131309e6db554744d`,src tree `7d6ebe9c65b06ab3917fdc76142920f042d668f1`. Complete declared46 workbook+13 control cases measured2026-11-12 on Hygon gfx936/Torch2.10/Torch-FL R7/FlagTree HCU.

- **98 focused hardware tests pass.** Noncontiguous device inputs reconstruct the CPU backing and exact stride/offset; direct `.to(flagos)` layout changes are recorded separately rather than changing the output assertion.
-59/59 new correctness passes versus48/59 old and56/59 native.11 old float-to-bool errors are fixed. Three native output-layout failures remain timed diagnostics.
-8 qualified old/new pairs:0.978311–2.869056x,2 gains>5%,no accepted slowdown>5%.
-11 qualified new/DAS pairs,10 DAS90 passes. The sole qualified miss is FP32[15,3802,19]->BF16,at0.764990 despite2.869056x old/new improvement.
-3 identity/empty cases have N/A device ratios. Other pairs that miss A/A are explicitly inconclusive and retain every sample.

[Complete all-case evidence](performance-20261112/README.md) contains full arrays, source/schema/backing/shard fingerprints and actual kernels. No profiler binaries. Fixed generated inputs reconstruct declared metadata; no original model payload replay is claimed. Both independent block-mean drifts<=5%,2x25 samples/role,20calls/ROI. Full values, cast bits and preserve-format output layouts are checked outside timing. No relaxed tolerances or sample filtering.

```bash
pytest tests/test_hygon_to_dtype.py --ref cpu -q
python docs/hygon-optimization/to_dtype/performance-20261112/verify_results.py
pre-commit run --all-files
```

This is a single-agent staged audit with independent provider processes and automated repeats, not independent-person approval. Latest-head CI is required before Ready for Review.
