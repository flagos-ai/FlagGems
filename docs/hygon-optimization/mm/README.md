# Hygon `aten::mm` optimization

## Scope

This PR adds the Hygon native-stride MM kernels used by the model and scale workbook workloads and restores FP64 operand loading in the native MMAC swizzle/coopfast paths. The FP64 correction is required for the affected MACE offset case and is submitted together with the optimization.

## Validation

- Source baseline: official FlagGems `master` plus the MM change set.
- Hygon R7 retest: Slurm job `1097237`, node `f11r2n10`, one isolated DCU; this retest used the official-master-derived branch and the same frozen two-case contract.
- `mm98-row0068` FP64 `[1344,256] @ [256,128]`, B offset `131072`: new FG, old FG and DAS all pass full output correctness and A/A; new kernel `mmac_tiled`; official-master retest: old/new/DAS kernel medians `82.56/41.60/15.84 us`, old/new `1.9846x`, DAS/new `0.3808x`; the optimization is correct but below DAS 90%.
- `mm98-row0056` FP32 transposed-B case: old/new/DAS share the same historical one-element tolerance miss; performance is diagnostic only; official-master retest medians `172.00/72.96/77.76 us`, old/new `2.3575x`, DAS/new `1.0658x`.
- Full raw evidence remains outside the repository. The compact result is archived at `D:/ZZ/analysis/pr2-two-case-retest-20261111/PR2-RETEST-AND-MERGE-RESULT.json`.

A performance ratio is `T_DAS / T_FG`; values below 1.0 are slower than DAS. Correctness failures are never converted into accepted performance results.

## Bottom-layer execution path

The public `aten::mm(Tensor, Tensor)` call enters the PrivateUse1/Torch-FL route, resolves in Hygon `ops/mm.py` and `mm_config.py`, and launches the selected Triton native MMAC implementation. The FP64 correction changes operand lane/LDS loading in `mm_mmac_coopfast.py` and `mm_mmac_swizzle.py`; dispatcher registration, reference tolerances and DAS boxing are unchanged.


## Official-master retest (2026-11-12)

The branch was revalidated from official `master` (`4d3d9f70c9c92ae9adf004ed0202355fe26fd511`) with the combined MM optimization and FP64 MMAC correction (`397ffccd0`). Slurm job `1097237` completed with exit code `0:0` on `f11r2n10`; raw review results are retained outside the repository. The two requested cases used the full old/new/DAS/A-A contract. The FP64 row0068 correctness repair remains clean, while row0056 is a historical shared failure, not a newly introduced regression.

The compact machine-readable receipt is `docs/hygon-optimization/mm/validation.json` (the full raw trace remains outside the repository).
