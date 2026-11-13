# Independent official-master performance verification

Measured 2026-11-12 on Hygon gfx936 with a private R7 release (Torch 2.10.0 + Torch-FL, FlagTree HCU). Baseline: `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`. Tested candidate: `e42573035c2ff5a3946288542c2f0aea649241dc`. Production fingerprints are in `source-sha256.json`.

All 1 declared cases are present in `cases.csv` and `cases.json`. New FG passed full-output correctness on 1/1. Complete timing: 1/1. Stable, correct FG-old/FG-new pairs: 1. Stable, correct FG-new/DAS pairs: 1, of which 1 meet `T_DAS/T_new >= 0.9`. Cases that failed A/A remain diagnostic and are explicitly marked. These counts apply to the disclosed reconstructed input contracts; missing original metadata is not silently inferred as model replay evidence.

Six independent provider processes run old FG, new FG, DAS and independent repeats for each. Each role has two blocks of 25 device samples. Each case records its calls per ROI (20 in this evidence set); every producer-linked device duration is retained and the complete sum divided by that call count. Both block means must differ by no more than 5% from the independent repeat. Reference failures keep diagnostic timings; they never count as performance passes. Old/new and new/DAS pairs qualify independently.

`measurements.json` contains all 50 samples per role, full backing hashes, exact ATen schema/source witnesses, and original/all-shard trace hashes. `cases.csv` shows diagnostic ratios separately from their qualification flags. A ratio above 1 means the candidate is faster. Input generation uses frozen seeds and metadata; it does not replay original model tensor values. No samples are dropped or combined across experiments.

Recompute the published results without accelerator dependencies:

```bash
python docs/hygon-optimization/addmm/vector-correction-20261112/verify_results.py
```

Focused correctness tests use the repository's corresponding Hygon/operator test modules. Full raw traces, original worker receipts and isolated source archives are preserved outside the repository; no profiler binaries are added here. Kernel/copy accounting uses strict producer and ROI attribution. CPU/API/compilation costs are separate from the published device metric. Eager execution is used; no Dynamo/Inductor fusion claim is made.

## Confirmation scope

This folder covers the **only affected case out of the full126-case PR scope**, not a reduction of the declared workload list. The source predicate was frozen before confirmation and evaluated for every input. The other125 whole cases await their separate full-suite offline accounting; see `../source-equivalence.json`. No successful-repeat selection or sample pooling.

The new comparison failure on the preceding source is fixed. Complete old/new/DAS outputs and both repeats pass the unchanged reference. Median producer-linked device time is40,895.990525us(old),40,899.610350us(new),166,800.490125us(DAS):old/new0.999911495,DAS/new4.078290446. Every role has2x25 samples,20calls/ROI;all independent block-mean drifts are below0.32%. Corrected actual kernel:`addmm_kernel`, the existing generic implementation. Separate API medians are41,065.8285us,40,972.8070us,165,529.4160us respectively and are not device measurements.

Hardware job1102530 completed0:0;offline audit1102531 completed0:0. Focused94 tests also passed. This confirmation does not qualify the remaining125 cases or turn the whole PR Ready; complete126-case evidence and latest-head CI are still required.
