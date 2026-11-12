# Hygon MM optimization and FP64 correctness protection

Guarded native-stride MMAC, split-K, banked and FP32 paths preserve generic fallback. FP64 banked loaders consume repaired canonical lanes sequentially; optional compiler hints are enabled only after capability probing.

Full official-master verification2026-11-12 covers98 main+109 expanded=207 cases. New correctness72/98 and161/207; no old-pass/new-fail, one baseline failure fixed. Every case, including inherited numerical failures, has complete timing.410 focused Hygon unit tests pass separately.

Main98:66 qualified old/new comparisons improve1.5750–24.9932x.65 qualified new/DAS comparisons yield50 DAS90 passes and15 misses. In the performance-only diagnostic view,91 cases have stable device timing:74 ratios>=90%,17 below90%;7 unstable. Historical correctness failures are not counted as full acceptance passes.

[Full per-case evidence](performance-20261112/README.md) contains all207 cases, every50-sample role array, main98 summary, target misses, errors/instability flags, source/schema/backing/shard hashes and actual kernel witnesses. Single-agent staged audit; no independent reviewer signoff claimed. Inputs reconstruct metadata, not original model payloads. The earlier two-case legacy-control receipt is preserved separately and is not official-master old/new evidence.
