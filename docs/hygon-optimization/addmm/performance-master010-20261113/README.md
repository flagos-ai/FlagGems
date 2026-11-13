# Independent official-master performance verification

Measured 2026-11-13 on Hygon gfx936 with a private R7 release (Torch 2.10.0 + Torch-FL, FlagTree HIP/gfx936). Baseline: `010d8083b0dc89ab795aee973ad62c83c9d2592d`. Tested candidate: `bee222e93b457a9ea054860de09c8e3256637957`. Production fingerprints are in `source-sha256.json`.

All 126 declared cases are present in `cases.csv` and `cases.json`. New FG passed full-output correctness on 125/126. Complete timing: 126/126. Stable, correct FG-old/FG-new pairs: 87. Stable, correct FG-new/DAS pairs: 98, of which 75 meet `T_DAS/T_new >= 0.9`. Cases that failed A/A remain diagnostic and are explicitly marked. These counts apply to the disclosed reconstructed input contracts; missing original metadata is not silently inferred as model replay evidence.

Six independent provider processes run old FG, new FG, DAS and independent repeats for each. Each role has two blocks of 25 device samples. Each case records its calls per ROI (20 in this evidence set); every producer-linked device duration is retained and the complete sum divided by that call count. Both block means must differ by no more than 5% from the independent repeat. Reference failures keep diagnostic timings; they never count as performance passes. Old/new and new/DAS pairs qualify independently.

`measurements.json` contains all 50 samples per role, full backing hashes, exact ATen schema/source witnesses, and original/all-shard trace hashes. `cases.csv` shows diagnostic ratios separately from their qualification flags. A ratio above 1 means the candidate is faster. Input generation uses frozen seeds and metadata; it does not replay original model tensor values. No samples are dropped or combined across experiments.

Recompute the published results without accelerator dependencies:

```bash
python docs/hygon-optimization/addmm/performance-master010-20261113/verify_results.py
```

Focused correctness tests use the repository's corresponding Hygon/operator test modules. Full raw traces, original worker receipts and isolated source archives are preserved outside the repository; no profiler binaries are added here. Kernel/copy accounting uses strict producer and ROI attribution. CPU/API/compilation costs are separate from the published device metric. Eager execution is used; no Dynamo/Inductor fusion claim is made.

The83 model contracts and43 expanded contracts are separately reviewed in `full-scope-review.json`. There are27 qualified gains above5%,including25 specialized-kernel gains. One qualified fallback case has7.72%higher latency and remains explicitly unresolved;see `fallback-regression-audit.json`. All23 qualified DAS90 misses and28 unqualified comparisons are preserved.108 focused hardware tests pass;9 accounting boundary tests pass.
