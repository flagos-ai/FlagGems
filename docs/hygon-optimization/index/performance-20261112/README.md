# Independent official-master performance verification

Measured 2026-11-12 on Hygon gfx936 with a private R7 release (Torch 2.10.0 + Torch-FL, FlagTree HCU). Baseline: `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`. Tested candidate: `e6cec3cde6cf525c5fe6e9ac93754a96d5c50a14`. Production fingerprints are in `source-sha256.json`.

All 125 declared cases are present in `cases.csv` and `cases.json`. New FG passed full-output correctness on 125/125. Complete timing: 125/125. Stable, correct FG-old/FG-new pairs: 47. Stable, correct FG-new/DAS pairs: 55, of which 55 meet `T_DAS/T_new >= 0.9`. Cases that failed A/A remain diagnostic and are explicitly marked. These counts apply to the disclosed reconstructed input contracts; missing original metadata is not silently inferred as model replay evidence.

Six independent provider processes run old FG, new FG, DAS and independent repeats for each. Each role has two blocks of 25 device samples. Each case records its calls per ROI (20 in this evidence set); every producer-linked device duration is retained and the complete sum divided by that call count. Both block means must differ by no more than 5% from the independent repeat. Reference failures keep diagnostic timings; they never count as performance passes. Old/new and new/DAS pairs qualify independently.

`measurements.json` contains all 50 samples per role, full backing hashes, exact ATen schema/source witnesses, and original/all-shard trace hashes. `cases.csv` shows diagnostic ratios separately from their qualification flags. A ratio above 1 means the candidate is faster. Input generation uses frozen seeds and metadata; it does not replay original model tensor values. No samples are dropped or combined across experiments.

Recompute the published results without accelerator dependencies:

```bash
python docs/hygon-optimization/index/performance-20261112/verify_results.py
```

Focused correctness tests use the repository's corresponding Hygon/operator test modules. Full raw traces, original worker receipts and isolated source archives are preserved outside the repository; no profiler binaries are added here. Kernel/copy accounting uses strict producer and ROI attribution. CPU/API/compilation costs are separate from the published device metric. Eager execution is used; no Dynamo/Inductor fusion claim is made.

## Guard confirmation and remaining uncertainty

All125 declared cases(49 workbook+76 expanded) have complete samples and all-provider correctness passes.47 old/new pairs qualify;35 improve>5%,range0.961593–21.101132x.55 new/DAS pairs qualify and all55 meet90%. The70 other new/DAS pairs remain A/A-inconclusive,not unmeasured or declared performance failures. No qualified old/new slowdown exceeds5%,but this does not prove that every unqualified case has no regression.

The previous full125 source showed two accepted slowdowns. The compact/singleton guard affects exactly5 metadata cases. All5 were selected before measurement and all5 whole six-role cases were replaced regardless outcome. Current singleton[1,1]/31360-index case qualifies at0.992440x old/new and6.555659x DAS/new. The[913,3,24]/2101-index case still has a0.879802x old/new point(old2.840us,new3.228us;new repeat2.960us),but both old and new A/A fail(6.07% and6.76% block drift). **That point is retained as unresolved;the regression is not claimed eliminated.** The third inconclusive confirmation case remains retained as well. Of5 confirmation cases,3 qualify and2 fail A/A.

`source-equivalence.json` chains whole-module AST and all125 metadata predicate proofs.120 untouched cases retain their entire preceding capture;5 changed cases retain their entire confirmation capture. Each case preserves its actual executed source and experiment. `kernel-witness.json` selects actual kernels from exactly that experiment without adding counts across runs. No successful-repeat selection or mixed samples. Actual ROI dates and original worker hashes are included per case;all measure2026-11-12.

**59 hardware tests pass,1 existing shared capability skip**,including>2GiB offsets,negative int32/int64 indices and explicit generic-route protection. Latest-head CI remains required before Ready for Review. Metadata-preserving generated inputs are not original payload replay. Single-agent staged audit with independent processes/repeats,not independent-person signoff.
