**Diagnostic archive only: no accepted performance comparisons.**

# Independent official-master performance verification

Measured 2026-11-12 on Hygon gfx936 with a private R7 release (Torch 2.10.0 + Torch-FL, FlagTree HCU). Baseline: `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`. Tested candidate: `d06609e2540e02e23a495f1e28bd6f4ec6539ab5`. Production fingerprints are in `source-sha256.json`.

All 200 declared cases are present in `cases.csv` and `cases.json`. New FG passed full-output correctness on 200/200. Complete timing: 200/200. Stable, correct FG-old/FG-new pairs: 0. Stable, correct FG-new/DAS pairs: 0, of which 0 meet `T_DAS/T_new >= 0.9`. Cases that failed A/A remain diagnostic and are explicitly marked. These counts apply to the disclosed reconstructed input contracts; missing original metadata is not silently inferred as model replay evidence.

Six independent provider processes run old FG, new FG, DAS and independent repeats for each. Each role has two blocks of 25 device samples. Each case records its calls per ROI (20 in this evidence set); every producer-linked device duration is retained and the complete sum divided by that call count. Both block means must differ by no more than 5% from the independent repeat. Reference failures keep diagnostic timings; they never count as performance passes. Old/new and new/DAS pairs qualify independently.

`measurements.json` contains all 50 samples per role, full backing hashes, exact ATen schema/source witnesses, and original/all-shard trace hashes. `cases.csv` shows diagnostic ratios separately from their qualification flags. A ratio above 1 means the candidate is faster. Input generation uses frozen seeds and metadata; it does not replay original model tensor values. No samples are dropped or combined across experiments.

Recompute the published results without accelerator dependencies:

```bash
python docs/hygon-optimization/add/diagnostic-concurrent-20261112/verify_results.py
```

Focused correctness tests use the repository's corresponding Hygon/operator test modules. Full raw traces, original worker receipts and isolated source archives are preserved outside the repository; no profiler binaries are added here. Kernel/copy accounting uses strict producer and ROI attribution. CPU/API/compilation costs are separate from the published device metric. Eager execution is used; no Dynamo/Inductor fusion claim is made.

## Diagnostic status: no accepted performance ratios

All200 original cases and all6role samples are present. Old/new/DAS full-output correctness is200/200 for each. **Zero FG-old/new and zero new/DAS pairs qualify** because the independent process repeats fail the frozen5% block-mean stability gate. These are inconclusive measurements,not missing tests,correctness failures or accepted speedups.

For `model285-f82b55d7fcde304e3a6a`,profiled new kernel median679.607425us contrasts with the same-source independent new-repeat8.019975us;unprofiled synchronized API medians135.192us and136.057us agree much more closely. Old-repeat and DAS-repeat also show the high-duration regime. Producer-linked events name the actual add kernels rather than the excluded gate kernel. This suggests a profiling/runtime/context interaction but does not establish its root cause. The fixed-output route/timing receipts and all raw arrays remain preserved;no samples are removed.

An entire200-case confirmation now runs one provider process at a time(job1103083),with the same production source,input metadata,seeds,reference tolerance,2x25 samples/role and complete device duration boundary. Every case replaces its whole result regardless outcome;the preceding capture remains diagnostic. It changes process scheduling,not kernel arithmetic or the acceptance threshold. Final performance and Ready status remain blocked until valid evidence and latest-head CI are available. No diagnostics from this pack are written to Feishu as accepted ratios.
