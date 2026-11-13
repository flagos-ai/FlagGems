# Independent official-master performance verification

Measured 2026-11-13 on Hygon gfx936 with a private R7 release (Torch 2.10.0 + Torch-FL, FlagTree HIP/gfx936). Baseline: `010d8083b0dc89ab795aee973ad62c83c9d2592d`. Tested candidate: `603af7eb112210767dbed397ed0bb99fe3a24dc8`. Production fingerprints are in `source-sha256.json`.

All 109 declared cases are present in `cases.csv` and `cases.json`. New FG passed full-output correctness on 48/109. Complete timing: 109/109. Stable, correct FG-old/FG-new pairs: 33. Stable, correct FG-new/DAS pairs: 35, of which 24 meet `T_DAS/T_new >= 0.9`. Cases that failed A/A remain diagnostic and are explicitly marked. These counts apply to the disclosed reconstructed input contracts; missing original metadata is not silently inferred as model replay evidence.

Six independent provider processes run old FG, new FG, DAS and independent repeats for each. Each role has two blocks of 25 device samples. Each case records its calls per ROI (20 in this evidence set); every producer-linked device duration is retained and the complete sum divided by that call count. Both block means must differ by no more than 5% from the independent repeat. Reference failures keep diagnostic timings; they never count as performance passes. Old/new and new/DAS pairs qualify independently.

`measurements.json` contains all 50 samples per role, full backing hashes, exact ATen schema/source witnesses, and original/all-shard trace hashes. `cases.csv` shows diagnostic ratios separately from their qualification flags. A ratio above 1 means the candidate is faster. Input generation uses frozen seeds and metadata; it does not replay original model tensor values. No samples are dropped or combined across experiments.

Recompute the published results without accelerator dependencies:

```bash
python docs/hygon-optimization/linear/performance-master010-20261113/verify_results.py
```

Focused correctness tests use the repository's corresponding Hygon/operator test modules. Full raw traces, original worker receipts and isolated source archives are preserved outside the repository; no profiler binaries are added here. Kernel/copy accounting uses strict producer and ROI attribution. CPU/API/compilation costs are separate from the published device metric. Eager execution is used; no Dynamo/Inductor fusion claim is made.

## Complete current-source review

All109 cases were freshly remeasured after the wide-vocabulary guard. There are33 qualified old/new pairs and4 gains above5%,all attributed to producer-linked new kernels,including2 of the54 original model contracts. There is no qualified latency regression above5%,and no stable latency regression above5% among reference-failing pairs.35 new/DAS pairs qualify:24 meet90%,11 do not;74 remain unqualified. The two original-model gains still miss DAS90. See `final-review.json`,`specialization-attribution.json`,and `verify-specialization.py`.

The `model54-r55` guard restores upstream `mm_kernel` and eliminates the preceding source's stable7.16x diagnostic slowdown. All three providers still fail reference correctness and DAS remains faster;this is no accepted performance gain. Details are in `wide-vocabulary-guard-review.json`. The inherited old/new output difference on `model54-r50` and the distinction between global and assertion-failing-element error metrics are disclosed in `reference-failure-note.json`;tolerances are unchanged.

Focused Hygon hardware validation passes12 tests;see `focused-unit-proof.json` and `focused-junit.xml`. The production tree still matches measured source603af7eb112210767dbed397ed0bb99fe3a24dc8. A test-only follow-up limits this vendor-specific file to Hygon;all prior test bodies,parameters,assertions and tolerances are unchanged. The updated test sourcebc6c2f7eb674f8bca7a6392544c5d996e1a4973a passes12 actual Hygon tests with zero skips. See `ci-test-scope-proof.json`,`ci-test-scope-junit.xml` and `verify-ci-test-scope.py`. This evidence is a single-agent staged review with automated independent provider processes/repeats;there is no separate human sign-off.
