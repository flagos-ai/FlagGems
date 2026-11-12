# Hygon addmm kernels and precision guards

Independent verification uses official master `010d8083b0dc89ab795aee973ad62c83c9d2592d`, measured candidate `bee222e93b457a9ea054860de09c8e3256637957`, Hygon gfx936 and the actual FlagTree HIP/gfx936 target. All126 declared inputs were measured on **2026-11-13**, including83 workbook model contracts and43 expansion contracts. Focused hardware tests: **108 passed**. This complete run supersedes preceding9a-baseline and guarded-subset diagnostics; samples are never pooled.

| Scope | Complete cases | Old/new/DAS correct | Qualified old/new | Gains >5% | Qualified new/DAS | DAS90 passes | Qualified DAS misses | New/DAS unqualified |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Model contracts |83|83/83/83|63|16|70|50|20|13|
| Full declared scope |126|123/125/122|87|27|98|75|23|28|

There are **no new correctness regressions**. The remaining candidate reference failure also fails old FG and DAS; all such cases retain complete diagnostic timings. Two previous old/DAS failures pass the candidate. The frozen tolerance is unchanged. Reference failures and A/A-inconclusive ratios do not count as accepted gains or DAS passes.

Qualified old/new ratios span0.928331-12.946162x (model subset0.928331-3.951807x). Of27 gains above5%,25 execute specialized `_precise_vector`, `_gemv` or `_grouped_addmm`;2 use the existing fallback and are not attributed to the new kernels. Long BF16 HIP/gfx936 inputs retain upstream kernels: several still fall well below DAS90, without a corresponding old/new regression. This PR does not claim all workloads meet DAS.

One qualified DiffDock case, `model285-0d5ec457ea9fb526f898`, remains **7.72% slower**: old8.808us,new9.488us,DAS13.280us. It falls outside specialization and both workers enter the upstream packaged Hygon generic launcher. Five relevant fallback/configuration files are byte-identical;all9 shared compiler keys in its group have identical instruction bodies after removing debug directives. These group-level artifacts do not identify the exact per-launch binary or establish the cause of the slowdown. The measured regression remains accepted and unresolved in [the audit](performance-master010-20261113/fallback-regression-audit.json); it is not removed or replaced by a favorable repeat.

[Complete evidence](performance-master010-20261113/README.md) contains all samples, accepted and diagnostic results, source fingerprints, exact ATen/source/kernel witnesses, dates, and portable verifiers. Historical guard/vector evidence remains for traceability;it does not measure the final source on the current baseline. Large BF16 spills on earlier candidates and the introduced extreme-FP32-vector comparison failure motivated the current architecture/scalar guards. The final full suite confirms their corrected routes.

## Reproduction and execution chain

```bash
pytest tests/test_hygon_addmm.py --ref cpu -q
python docs/hygon-optimization/addmm/performance-master010-20261113/verify_results.py
python docs/hygon-optimization/addmm/performance-master010-20261113/test-accounting-boundaries.py
pre-commit run --all-files
```

Frozen generated backing restores known shape/stride/offset/aliases and discloses uncaptured assumptions;this is not replay of original model tensor values. Exact `aten::addmm` -> process-private PrivateUse1 registration -> Hygon provider -> specialized or packaged upstream kernel is witnessed per case. Native DAS uses an independent process. The public source does not depend on the separate mm PR.

Six serial roles(old/new/DAS and their independent repeats) each have2x25 device ROIs,20calls/ROI. Both independent block drifts must be<=5%. Producer-linked device kernel sums define latency;memory copies,50-call synchronized unprofiled API controls,compilation and wrapper costs are separate. Only final same-thread nonproducing blocking waits crossing the right host boundary are audited separately;device and producing API containment remains strict,no timestamp shifts,trimming or omitted events. Nine boundary/counterexample tests cover this accounting. Eager/no_grad;no graph fusion,model-throughput or independent-person-signoff claim. Hardware: Hygon gfx936,Torch2.10/Torch-FL R7,FlagTree HIP/gfx936.
