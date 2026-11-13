# Independent official-master performance verification

Measured 2026-11-12 on Hygon gfx936 with a private R7 release (Torch 2.10.0 + Torch-FL, FlagTree HCU). Baseline: `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`. Tested candidate: `2862d83d0d8cd7d599f2059131309e6db554744d`. Production fingerprints are in `source-sha256.json`.

All 59 declared cases are present in `cases.csv` and `cases.json`. New FG passed full-output correctness on 59/59. Complete timing: 59/59. Stable, correct FG-old/FG-new pairs: 8. Stable, correct FG-new/DAS pairs: 11, of which 10 meet `T_DAS/T_new >= 0.9`. Cases that failed A/A remain diagnostic and are explicitly marked. These counts apply to the disclosed reconstructed input contracts; missing original metadata is not silently inferred as model replay evidence.

Six independent provider processes run old FG, new FG, DAS and independent repeats for each. Each role has two blocks of 25 device samples. Each case records its calls per ROI (20 in this evidence set); every producer-linked device duration is retained and the complete sum divided by that call count. Both block means must differ by no more than 5% from the independent repeat. Reference failures keep diagnostic timings; they never count as performance passes. Old/new and new/DAS pairs qualify independently.

`measurements.json` contains all 50 samples per role, full backing hashes, exact ATen schema/source witnesses, and original/all-shard trace hashes. `cases.csv` shows diagnostic ratios separately from their qualification flags. A ratio above 1 means the candidate is faster. Input generation uses frozen seeds and metadata; it does not replay original model tensor values. No samples are dropped or combined across experiments.

Recompute the published results without accelerator dependencies:

```bash
python docs/hygon-optimization/to_dtype/performance-20261112/verify_results.py
```

Focused correctness tests use the repository's corresponding Hygon/operator test modules. Full raw traces, original worker receipts and isolated source archives are preserved outside the repository; no profiler binaries are added here. Kernel/copy accounting uses strict producer and ROI attribution. CPU/API/compilation costs are separate from the published device metric. Eager execution is used; no Dynamo/Inductor fusion claim is made.

## Interpretation and explicit misses

All59 cases retain complete samples. Old/new/DAS correctness is48/59,59/59,56/59. Eleven pre-existing floating-to-bool failures are fixed; the3 native preserve-format layout failures remain native-reference diagnostics and were still timed. No new correctness failure. There are8 stable/correct old/new pairs, with speedup0.978311–2.869056x;2 improve>5%, none regress>5%. There are11 stable/correct new/DAS pairs,10 of which meet90%.

The other positive-work pairs remain A/A-inconclusive; this is not a numerical failure or an unmeasured case. Three no-device-work cases (one identity/no-copy and two empty conversions) have API/semantic checks and N/A device ratios, not performance failures. Kernel and copy activities are retained separately; this run has no nonzero device-copy contribution, so kernel-sum equals its complete device-work metric.

| Correct/stable miss | Old / new / DAS device us | Old/new | DAS/new |
|---|---:|---:|---:|
|to-extended-940044:FP32[15,3802,19] to BF16|16.652 /5.804 /4.440|2.869056|0.764990|

This case improves substantially over old FG but remains below the DAS90% threshold. It uses the candidate `cast_dense` kernel. The medium-layout and packed-range guards no longer produce a>5% accepted old/new slowdown in this full repeat. Actual kernels for every case/role are in `kernel-witness.json`. Unprofiled API sample arrays are separate in `measurements.json`; they never replace device timing.
