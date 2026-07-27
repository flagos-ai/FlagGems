# FlagGems operator repair memory

## 2026-07-24: current Kunlunxin accuracy baseline

- User-authoritative baseline: `/workspace/FlagGems/logs_results_20260723_2307/summary.json` represents the latest tested `master` for subsequent repairs.
- Accuracy inventory: 820 markers total; 501 Passed, 197 Failed, 58 Timeout, 34 Error, 16 Skipped, and 14 NotFound.
- The 14 NotFound markers have no test case in the current harness and are excluded from the repair queue for now: `alias_copy_out`, `and_scalar`, `and_tensor`, `clone`, `dispatch_fused_moe_kernel`, `fused_moe`, `median_dim`, `median_dim_values`, `median_out`, `random_`, `scaled_dot_product_flash_attention_backward`, `special_chebyshev_polynomial_w_out`, `special_erfinv_`, and `stage_deepseek_v4_mega_moe_inputs`.
- Existing submitted/topic repairs, including `sinc/sinc_`, must not be duplicated. Start each new logical repair from this master baseline on a focused topic branch; group only 3-5 closely related operators when implementation and risk are genuinely shared.

## 2026-07-24: special_log1p classified as platform/test mismatch

- Status: infrastructure/vendor-capability classification; no product-code change and no branch retained.
- Baseline: `special_log1p` recorded 23 passed / 1 failed. The failing node is `tests/test_special_log1p.py::test_special_log1p_small_values`.
- The test requests a float64 P800 tensor, but the Kunlunxin descriptor declares no fp64 support and the created device tensor is already `torch.float32`. Both native and CPU-roundtrip outputs therefore remain float32, while the test hard-codes `torch.float64` in `gems_assert_close`.
- Because the operator receives a float32 tensor, it cannot recover the originally requested float64 dtype without breaking normal float32 output semantics. Do not hide this with an operator-specific shape/value hack or tolerance change.
- Evidence: `/workspace/results/special_log1p/20260724-1743-current-master/baseline_small_values.json` and `/workspace/docs/special_log1p_accuracy_failure_analysis.md`.

## 2026-07-24: floor_divide mixed-dtype dispatch fixed

- Status: fixed locally on branch `klx/floor-divide-mixed-dtype-fix`; no commit, push, or PR.
- Baseline commit: `68573e211`. Original batch marker: 8 passed / 1 failed; exact reproducer failed in 1.58s with `TypeError('unexpected type fp32')` while compiling the integer floor-divide path.
- Root cause: all three Kunlunxin floor-divide pointwise entry functions checked `x.type.scalar.is_int()` twice. An integer left operand with a floating right operand was therefore sent to `_int_floordiv`.
- Product file changed: `src/flag_gems/runtime/backend/_kunlunxin/ops/div.py`. Each predicate now checks both `x` and `y`; no tests or tolerances changed.
- Validation on P800: exact node 1/1 passed; original `floor_divide_scalar` marker 9/9 passed; full `tests/test_floor_divide.py` 75/75 passed; deterministic tensor/tensor, tensor/scalar, scalar/tensor diagnostics and registered `torch.floor_divide` matched CPU reference.
- Target-file `pre-commit run --files src/flag_gems/runtime/backend/_kunlunxin/ops/div.py` passed all applicable hooks. `git diff --check` passed.
- An initial post-fix test accidentally loaded `/workspace/FlagGems/src` through `_flag_gems_editable.pth`; it reproduced the old failure and is excluded from fix validation. Later commands removed that editable finder per process and verified the imported source path was the worktree.
- Evidence: `/workspace/results/floor_divide_scalar/20260724-1748-current-master/` and `/workspace/docs/floor_divide_scalar_accuracy_fix_report.md`.

## 2026-07-24: amin/amax multi-dimension reductions fixed

- Status: fixed locally on branch `klx/amin-amax-multidim-fix`; no commit, push, or PR.
- Baseline commit: `68573e211`. Current batch results were `amin` 6/12, duplicate `amin_` marker 6/12, and `amax` 6/12. Every multi-dimension case failed before the reduction kernel in `dim_compress -> permute().contiguous() -> copy_` with `CUDA error: invalid device function`.
- Root cause: the Kunlunxin multi-dimension implementations required a full contiguous permutation, which entered the currently broken composite `copy_` redispatch path. Single-dimension and global reduction kernels were not failing.
- Files changed: `src/flag_gems/runtime/backend/_kunlunxin/ops/amin.py` and `amax.py`. Each module captures the native CUDA reduction kernel before FlagGems registration and uses `call_boxed` with a CUDA keyset only when `len(dim) > 1`; existing Triton kernels remain for zero/single-dimension reductions.
- Native A/B: amin and amax, fp16/fp32/bf16, small and large original shapes, both dimension orders: 12/12 exact matches against CPU.
- Validation: original `tests/test_amin.py` 24/24 passed; `tests/test_amax.py` 12/12 passed; repeated combined run 36/36 passed; negative multi-dim indices passed; target-file pre-commit and `git diff --check` passed.
- The `amin_` marker calls `torch.amin`, and the installed ATen namespace has no `aten::amin_` schema. Its 12 cases are duplicate functional coverage, not a real in-place API validation.
- Evidence: `/workspace/results/amin_amax/20260724-1850-current-master/` and `/workspace/docs/amin_amax_multidim_accuracy_fix_report.md`.
