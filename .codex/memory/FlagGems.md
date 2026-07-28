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

## 2026-07-27: acos/arccos P800 numerical accuracy fixed locally

- Status: fixed locally; no commit, push, or PR. Branch `klx/acos-arccos-accuracy-fix`, baseline/current HEAD `68573e211`.
- Reproduction: under the target worktree with the editable finder bypassed, `python -m pytest tests/test_acos.py tests/test_arccos.py -q --ref=cpu` equivalent ran 54 original cases: 29 passed / 25 failed before the fix. `acos` was 10/18, `arccos` 9/18, `arccos_` 10/18. Representative errors were fp16 max abs `0.00390625` and fp32 max abs about `0.003057` against `atol=1e-4`.
- Root cause: both Kunlunxin vendor kernels directly called `tl_extra_shim.acos(x.to(tl.float32))`; the P800 intrinsic has a repeatable in-domain error. This is a vendor numerical-accuracy issue, not a proven specific compiler lowering defect.
- Product files: `src/flag_gems/runtime/backend/_kunlunxin/ops/acos.py` and `arccos.py` now compute `atan2(sqrt(max(1-x*x,0)),x)` for `abs(x)<=1` and retain intrinsic fallback for out-of-domain/NaN; `op_black_list.yaml` no longer marks `acos` unsupported. Generic paths were not changed.
- Validation: target public API boundary values (fp16/fp32, domain/endpoints/NaN, functional/alias/in-place) matched; original 54 cases passed three times (18.62s including first compile, 2.81s repeat, 2.85s after blacklist metadata update); JSON evidence `/workspace/results/acos_arccos/20260727-1336-postfix-target/accuracy_result.json`. `-m abs` was 18/18 with 49,939 deselected; repetition-penalty fallback was 126/126 with 49,831 deselected. Target-file pre-commit and `git diff --check` passed.
- Performance: manual synchronized `[4096,4096]` median changed from baseline FlagGems intrinsic 2.257ms fp16 / 2.109ms fp32 to 7.175ms / 9.090ms; accuracy fix is 3.18x/4.31x slower and needs later optimization. Official benchmark was infrastructure-invalid because `triton.testing.do_bench` estimated 0ms and raised `ZeroDivisionError`.
- Separate collection issue: vLLM `_C.abi3.so` reports an unresolved torch CUDA symbol, while its Python wrapper remains importable; probing missing `torch.ops._C.apply_repetition_penalties_` raises `AttributeError`. The user-authorized test change catches `(ImportError, RuntimeError, AttributeError)` and uses the existing fallback. This does not repair vLLM ABI compatibility. Pattern recorded in the repair skill.
- Remaining risks: `atan2` has an independent rare pi-branch mismatch in related random tests; fp64 is unsupported by the P800 descriptor; no other backends were changed. Detailed report: `/workspace/docs/acos_accuracy_fix_report.md`.

## 2026-07-27: softmax forward P800 runtime failure fixed locally

- Status: fixed for the user-requested `softmax` forward marker. Baseline `68573e211`; source commit `f42d4e072` on `klx/amin-amax-multidim-fix`; no push or PR.
- Reproduction: `python -m pytest -m "softmax" -svv --ref cpu` selected 48 original forward cases and produced 0 passed / 48 failed / 0 skipped in 12.89s. The first independent error was XPU kernel exception `-714`/CUDA status 719; later failures were consequences of the poisoned process context.
- Root cause: the Kunlunxin K>1 path materialized `[M,N,K] -> [M,K,N]` with `.contiguous()`, which entered a broken `copy_` fallback (`invalid device function`). The inner Triton kernel also failed for small and masked-tail reductions (verified N=1 and N=65), while complete power-of-two tiles from N=64 passed.
- Product file: `src/flag_gems/runtime/backend/_kunlunxin/ops/softmax.py`. It captures the native CUDA `_softmax` kernel before FlagGems registration and redispatches K>1, N<64, or non-power-of-two N through an explicit CUDA keyset. The existing Triton path remains for K==1, N>=64 power-of-two tiles.
- Validation on P800: original forward marker passed 48/48 repeatedly, including the final worktree run in 12.52s; after synchronizing master, the user's exact plain command also passed 48/48 in 11.92s. Deterministic fp16/fp32/bf16 diagnostics, `-inf`, negative dims, noncontiguous inputs, half-to-float, and N=1/33/64/65/96/128 passed. Target-file pre-commit and `git diff --check` passed.
- Environment caveat: the installed `_flag_gems_editable` finder points to `/workspace/FlagGems/src`. Initial fix validation removed that finder per process and verified the repair worktree source. After the same commit reached master, plain pytest legitimately loads the fixed master source and passes without a bypass.
- Delivery: synchronized as a one-file local commit to all 9 named local branches: master `63fb0c8ae`, acos `a264cb954`, amin/amax `f42d4e072`, floor-divide `c9eace5c8`, GLU `e0c3ee4ca`, max-pool3d `04a00993d`, scaled-softmax `6306a0933`, scatter `200e5f0c6`, and sort-stable `7a7897b80`. Older branches retained their branch-specific logger/vectorization configuration. The detached XPU event worktree is not a branch and was not changed.
- Cross-version validation: the adapted old max-pool3d branch passed all 48 original forward cases when `tests/test_softmax.py` was named directly (120 other file cases deselected, 4.28s). Its full-suite marker was separately blocked before target execution by the branch's known optional-vLLM `AttributeError` collection issue.
- Remaining scope: `softmax_out` (24/48), `softmax_backward` (18/36), and `softmax_backward_out` (0 passed / 30 failed / 6 skipped) still expose independent generic/backward paths and were not modified in this forward-marker repair.
- Evidence: `/workspace/results/softmax/` and `/workspace/docs/softmax_accuracy_fix_report.md`.
