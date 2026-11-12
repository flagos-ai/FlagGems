# Independent official-master performance verification

Measured 2026-11-12 on Hygon gfx936 with a private R7 release (Torch 2.10.0 + Torch-FL, FlagTree HCU). Baseline: `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`. Tested candidate: `60b01904510660fc692d02104d887837d41abbab`. Production fingerprints are in `source-sha256.json`.

All 32 declared cases are present in `cases.csv` and `cases.json`. New FG passed full-output correctness on 32/32. Complete timing: 32/32. Stable, correct FG-old/FG-new pairs: 11. Stable, correct FG-new/DAS pairs: 15, of which 15 meet `T_DAS/T_new >= 0.9`. Cases that failed A/A remain diagnostic and are explicitly marked. These counts apply to the disclosed reconstructed input contracts; missing original metadata is not silently inferred as model replay evidence.

Six independent provider processes run old FG, new FG, DAS and independent repeats for each. Each role has two blocks of 25 device samples. A sample contains 20 calls; every producer-linked device duration is retained and the complete sum divided by that call count. Both block means must differ by no more than 5% from the independent repeat. Reference failures keep diagnostic timings; they never count as performance passes. Old/new and new/DAS pairs qualify independently.

`measurements.json` contains all 50 samples per role, full backing hashes, exact ATen schema/source witnesses, and original/all-shard trace hashes. `cases.csv` shows diagnostic ratios separately from their qualification flags. A ratio above 1 means the candidate is faster. Input generation uses frozen seeds and metadata; it does not replay original model tensor values. No samples are dropped or combined across experiments.

Recompute the published results without accelerator dependencies:

```bash
python docs/hygon-optimization/bmm/performance-20261112/verify_results.py
```

Focused correctness tests are in `tests/test_hygon_bmm.py`. Full raw traces, original worker receipts and isolated source archives are preserved outside the repository; no profiler binaries are added here. Kernel/copy accounting uses strict producer and ROI attribution. CPU/API/compilation costs are separate from the published device metric. Eager execution is used; no Dynamo/Inductor fusion claim is made.


## Audit and CI availability

Process-level A/A repeats and offline verification were performed in stages by one agent; no independent-person signoff is claimed. The32-case correctness and all-role raw measurements are separate from CI scheduling.

The Hygon job in CI run43198954642 failed before operator tests because its runner timed out waiting for an available DCU. It is classified as infrastructure failure rather than operator correctness failure. A subsequent evidence update triggers a fresh latest-head check; this PR remains Draft until that check succeeds. The original failed CI logs and fixed-source hardware raw are preserved outside git.
