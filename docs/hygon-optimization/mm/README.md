# Hygon MM optimization and FP64 correctness protection

Guarded native-stride MMAC, split-K, banked and FP32 paths preserve generic fallback. FP64 banked loaders consume repaired canonical lanes sequentially; optional compiler hints are enabled only after capability probing.

Full official-master verification2026-11-12 covers98 main+109 expanded=207 cases. New correctness72/98 and161/207; no old-pass/new-fail, one baseline failure fixed. Every case, including inherited numerical failures, has complete timing.410 focused Hygon unit tests pass separately.

Main98:66 qualified old/new comparisons improve1.5750–24.9932x.65 qualified new/DAS comparisons yield50 DAS90 passes and15 misses. In the performance-only diagnostic view,91 cases have stable device timing:74 ratios>=90%,17 below90%;7 unstable. Historical correctness failures are not counted as full acceptance passes.

[Full per-case evidence](performance-20261112/README.md) contains all207 cases, every50-sample role array, main98 summary, target misses, errors/instability flags, source/schema/backing/shard hashes and actual kernel witnesses. Single-agent staged audit; no independent reviewer signoff claimed. Inputs reconstruct metadata, not original model payloads. The earlier two-case legacy-control receipt is preserved separately and is not official-master old/new evidence.

## Supplementary synchronized API controls

`api-controls.json` publishes all50 unprofiled samples for every old/new/DAS/new-repeat role across207 cases,with original control-file hashes and profiler-exclusion proofs. `api-case-summary.csv` places these API medians alongside the device ratios and qualification flags. The API boundary includes dispatch,allocation,launch and synchronization; it does not measure only the kernel and is not model throughput. These supplementary API points did not have a separately frozen acceptance gate; do not turn device AA qualification into an API gain claim.

Kernel gains can be smaller or absent at the complete API boundary. For main row0085(FP64[2208,64]x[64,64]),device time improves25.600->9.707975us(2.637x),while the API median is144.5755->154.0260us(old/new0.9386435;new-repeat153.8255us). For row0130(FP32[32,128]x[128,512]column-major),kernel11.892->4.172us(2.850x),API132.3855->140.4760us,new-repeat133.5405us. These are explicit API overhead/noise diagnostics,not confirmed API regression counts or a claim of end-to-end speedup. The fixed primary device acceptance counts remain unchanged.
