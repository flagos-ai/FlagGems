> Diagnostic history before the current fallback guards. Measured source
> `8ed056afe466023d1c24e6ede4060324dd73a8f1`, official baseline
> `010d8083b0dc89ab795aee973ad62c83c9d2592d`. All32 cases are complete/correct,
> but one qualified case regresses8.55%; this is not current-source acceptance.
> The current guarded source reruns all32 cases and replaces the whole matrix
> regardless outcome. Both raw versions remain preserved.

# Independent official-master performance verification

Measured 2026-11-13 on Hygon gfx936 with a private R7 release (Torch 2.10.0 + Torch-FL, FlagTree HIP/gfx936). Baseline: `010d8083b0dc89ab795aee973ad62c83c9d2592d`. Tested candidate: `8ed056afe466023d1c24e6ede4060324dd73a8f1`. Production fingerprints are in `source-sha256.json`.

All 32 declared cases are present in `cases.csv` and `cases.json`. New FG passed full-output correctness on 32/32. Complete timing: 32/32. Stable, correct FG-old/FG-new pairs: 18. Stable, correct FG-new/DAS pairs: 21, of which 18 meet `T_DAS/T_new >= 0.9`. Cases that failed A/A remain diagnostic and are explicitly marked. These counts apply to the disclosed reconstructed input contracts; missing original metadata is not silently inferred as model replay evidence.

Six independent provider processes run old FG, new FG, DAS and independent repeats for each. Each role has two blocks of 25 device samples. Each case records its calls per ROI (20 in this evidence set); every producer-linked device duration is retained and the complete sum divided by that call count. Both block means must differ by no more than 5% from the independent repeat. Reference failures keep diagnostic timings; they never count as performance passes. Old/new and new/DAS pairs qualify independently.

`measurements.json` contains all 50 samples per role, full backing hashes, exact ATen schema/source witnesses, and original/all-shard trace hashes. `cases.csv` shows diagnostic ratios separately from their qualification flags. A ratio above 1 means the candidate is faster. Input generation uses frozen seeds and metadata; it does not replay original model tensor values. No samples are dropped or combined across experiments.

Recompute the published results without accelerator dependencies:

```bash
python docs/hygon-optimization/bmm/diagnostic-master010-r2-20261113/verify_results.py
```

Focused correctness tests use the repository's corresponding Hygon/operator test modules. Full raw traces, original worker receipts and isolated source archives are preserved outside the repository; no profiler binaries are added here. Kernel/copy accounting uses strict producer and ROI attribution. CPU/API/compilation costs are separate from the published device metric. Eager execution is used; no Dynamo/Inductor fusion claim is made.
