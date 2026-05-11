# smooth_l1_loss — Test Coverage Checklist

Operator: `aten::smooth_l1_loss` + `aten::smooth_l1_loss_backward`
Test file: `tests/test_smooth_l1_loss.py` (450 lines, 25+ test functions)
Benchmark file: `benchmark/test_smooth_l1_loss.py` (uses `base.GenericBenchmark2DOnly`)

## Dimension × Scale × Dtype × Parameter Coverage

| Test function | Scale | Shape | Dim | Dtype | Reduction | beta | Branch / Feature |
|---|---|---|---|---|---|---|---|
| `test_accuracy_smooth_l1_loss` | small→large | `POINTWISE_SHAPES` (1×N…1024×N) | 1D-4D | fp32/bf16/fp16 | none/mean/sum | 0.5/1.0/2.0 | forward correctness baseline |
| `test_smooth_l1_loss_zero_difference` | medium | (128, 64) | 2D | fp32/bf16/fp16 | none/mean/sum | 1.0 | zero-diff identity (no 0/beta NaN) |
| `test_smooth_l1_loss_quadratic_branch` | medium | (1024,) | 1D | fp32/bf16/fp16 | none | 1.0 | `|diff| < beta` exact value |
| `test_smooth_l1_loss_linear_branch` | medium | (1024,) | 1D | fp32/bf16/fp16 | none | 1.0 | `|diff| ≥ beta` exact value |
| `test_smooth_l1_loss_empty_tensor` | edge | (0,) | 1D | fp32 | none/mean/sum | default | empty input (NaN for mean, 0 for sum) |
| `test_smooth_l1_loss_backward` | small→medium | (128,)/(32,64)/(4,8,16) | 1D-3D | fp32/bf16/fp16 | none/mean/sum | 0.5/1.0/2.0 | grad parity vs autograd reference |
| `test_smooth_l1_loss_backward_kink` | tiny | (5,) | 1D | fp32/bf16/fp16 | sum | 1.0 | grad sign at kink |diff|=beta |
| `test_smooth_l1_loss_large_shapes` | large | (1024², 32×32×1024, 4×8×16×32×64, 1M flat, 1024×1024) | 1D-5D | fp32/bf16/fp16 | mean/sum | default | bandwidth-bound stress |
| `test_smooth_l1_loss_value_scale_sweep` | medium | (256, 256) | 2D | fp32/bf16/fp16 | mean | 1.0 | inputs at 1e-4 … 1e4 (numerical stability) |
| `test_smooth_l1_loss_negative_values` | medium | (128, 64) | 2D | fp32/bf16/fp16 | none | 1.0 | sign-symmetry under `(x,y) -> (-x,-y)` |
| `test_smooth_l1_loss_beta_sweep` | medium | (64, 128) | 2D | fp32/bf16/fp16 | mean | **0.01 / 0.1 / 0.25 / 1.0 / 2.0 / 10.0** | beta extremes |
| `test_smooth_l1_loss_backward_large` | large | (1024², 8×4×1024, 256K flat) | 1D-3D | fp32/bf16/fp16 | mean | default | backward at production sizes |
| `test_smooth_l1_loss_backward_zero_grad_output` | small | (128,) | 1D | fp32/bf16/fp16 | none | default | `grad_output=0 → grad_input=0` |
| `test_smooth_l1_loss_backward_target_grad` | small | (64,) | 1D | fp32/bf16/fp16 | sum | default | `d/dtarget = -d/dinp` |
| `test_smooth_l1_loss_mean_equals_sum_over_n` | medium | (256, 128) | 2D | fp32/bf16/fp16 | mean+sum | default | reduction invariant |
| `test_smooth_l1_loss_none_sum_matches_sum_reduction` | medium | (64, 64) | 2D | fp32/bf16/fp16 | none+sum | default | `sum(none) ≡ sum` |
| `test_smooth_l1_loss_non_contiguous` | medium | (64, 64) stride-2 view | 2D non-contig | fp32/bf16/fp16 | mean | default | strided dispatch |
| `test_smooth_l1_loss_transposed_input` | medium | (64, 128) `.t()` | 2D non-contig | fp32/bf16/fp16 | mean | default | column-major view |
| `test_smooth_l1_loss_deterministic` | medium | (128, 128) | 2D | fp32/bf16/fp16 | mean | default | bitwise reproducibility |
| `test_smooth_l1_loss_int_reduction` | small | (128,) | 1D | default | int 0/1/2 | default | aten enum dispatch path |

## Functional-branch coverage

| Branch | Test |
|---|---|
| Forward quadratic (`|diff| < beta`) | `test_smooth_l1_loss_quadratic_branch` |
| Forward linear (`|diff| ≥ beta`) | `test_smooth_l1_loss_linear_branch` |
| Forward `reduction="none"` | parametrize on every accuracy test |
| Forward `reduction="mean"` | parametrize on every accuracy test |
| Forward `reduction="sum"` | parametrize on every accuracy test |
| Backward via `aten::smooth_l1_loss_backward` | `test_smooth_l1_loss_backward*` |
| Empty input + mean → NaN | `test_smooth_l1_loss_empty_tensor` |
| Empty input + sum → 0 | `test_smooth_l1_loss_empty_tensor` |

## API alignment

- Public function `smooth_l1_loss(inp, target, reduction=1, beta=1.0)` — accepts both **int enum** (aten dispatch) and **string** (`"none"`/`"mean"`/`"sum"`) for `reduction`.
- Public function `smooth_l1_loss_backward(grad_output, inp, target, reduction, beta)` — registered as `aten::smooth_l1_loss_backward` so torch autograd routes through Triton automatically.
- `assert beta > 0` rejects illegal beta.

## Implementation differentiators vs competing PRs

1. **Triton backward kernel** (2 variants: per-element grad_output for `none`; scalar broadcast for `mean`/`sum`). Other PRs rely on torch autograd through the forward.
2. Reduction parameter accepts both int (`aten`) and string conventions.
3. Empty-input semantics match torch exactly (NaN for mean, 0 for sum).
4. Two-phase parallel reduction via `mid` buffer for the `mean`/`sum` path.
