# Independent official-master performance verification

Measured 2026-11-12 through2026-11-13 on Hygon gfx936 with a private R7 release (Torch 2.10.0 + Torch-FL, FlagTree HIP/gfx936). Baseline: `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`. Tested candidate: `d06609e2540e02e23a495f1e28bd6f4ec6539ab5`. Production fingerprints are in `source-sha256.json`.

All 200 declared cases are present in `cases.csv` and `cases.json`. New FG passed full-output correctness on 200/200. Complete timing: 200/200. Stable, correct FG-old/FG-new pairs: 79. Stable, correct FG-new/DAS pairs: 99, of which 75 meet `T_DAS/T_new >= 0.9`. Cases that failed A/A remain diagnostic and are explicitly marked. These counts apply to the disclosed reconstructed input contracts; missing original metadata is not silently inferred as model replay evidence.

Six independent provider processes run old FG, new FG, DAS and independent repeats for each. Each role has two blocks of 25 device samples. Each case records its calls per ROI (20 in this evidence set); every producer-linked device duration is retained and the complete sum divided by that call count. Both block means must differ by no more than 5% from the independent repeat. Reference failures keep diagnostic timings; they never count as performance passes. Old/new and new/DAS pairs qualify independently.

`measurements.json` contains all 50 samples per role, full backing hashes, exact ATen schema/source witnesses, and original/all-shard trace hashes. `cases.csv` shows diagnostic ratios separately from their qualification flags. A ratio above 1 means the candidate is faster. Input generation uses frozen seeds and metadata; it does not replay original model tensor values. No samples are dropped or combined across experiments.

Recompute the published results without accelerator dependencies:

```bash
python docs/hygon-optimization/add/diagnostic-serial-20261113/verify_results.py
```

Focused correctness tests use the repository's corresponding Hygon/operator test modules. Full raw traces, original worker receipts and isolated source archives are preserved outside the repository; no profiler binaries are added here. Kernel/copy accounting uses strict producer and ROI attribution. CPU/API/compilation costs are separate from the published device metric. Eager execution is used; no Dynamo/Inductor fusion claim is made.

This is preceding-source diagnostic history,not acceptance for the pending guarded candidate. All200 cases are complete and correct.79 old/new pairs qualify,44 gain>5%,but3 device regressions remain: `model285-ff98957de60effd9753c`0.859160x,`expanded230-ecfb3568671511bd44e4`0.945983x,and`expanded230-90993775a8cc9d5dff7f`0.603094x. Their synchronized API controls improve,which does not erase the device regressions.99 new/DAS pairs qualify,75 meet90%. Ratios on the other101 new/DAS cases are A/A-inconclusive. The fixed final source retains generic scheduling in the three affected metadata domains;all200 complete six-role cases will be replaced regardless outcome. No sample pooling or selected-success repeats. Exact accountant source snapshots use JSON string containers to preserve all original bytes/newlines under formatting hooks;9 boundary tests verify them.
