# smooth_l1_loss — Design Notes

## 1. Operator surface

PyTorch exposes the Huber/SmoothL1 loss via three callable surfaces that all
ultimately dispatch to `aten::smooth_l1_loss` (forward) and
`aten::smooth_l1_loss_backward` (gradient):

| Surface | Backward routed via |
|---|---|
| `torch.nn.functional.smooth_l1_loss(x, y, reduction, beta)` | autograd graph |
| `torch.nn.SmoothL1Loss(...)(x, y)` | autograd graph |
| `loss.backward()` | autograd graph |

To capture **all three paths** we register both `smooth_l1_loss` and
`smooth_l1_loss_backward` at the aten op level (see
`src/flag_gems/__init__.py`).  That way torch's autograd dispatcher
automatically picks Triton on the way back too — no Python-level
`autograd.Function` wrapping required.

## 2. Forward kernels

### `smooth_l1_loss_none_kernel` (reduction = none)
- Built with `pointwise_dynamic` — FlagGems' generic elementwise launcher.
- The math is one branchless `where` over the quadratic vs linear branch.
- Result shape matches input; computation is fp32 internally then cast back.

### Two-phase reduction for `reduction = mean` / `sum`
We need a global scalar so a single reduction has to span the whole tensor.
Triton kernels can't do that in one launch reliably (warp scope is small),
so we use the FlagGems convention of two kernels:

```
phase 1: smooth_l1_loss_reduce_kernel
    grid = (mid_size,)  where mid_size = ceil(M / block_size)
    each program loads BLOCK_SIZE elements, computes per-element loss,
    reduces inside the program, writes partial sum into mid[pid].
    block_size is chosen as 2^ceil(log2(sqrt(M))) which keeps both
    grid count and per-program work O(sqrt(M)) — good GPU saturation.

phase 2: reduce_sum_kernel
    grid = (1,) — single program reads all mid[*] and writes the final sum.
    Block size = next_pow2(mid_size) so the whole array fits in one tile.
```

`reduction = mean` divides the per-program partial sum by `M` before the
final accumulate (more numerically stable than dividing once at the end).

## 3. Backward kernels

Two purpose-built kernels, both pointwise:

| Kernel | Used when | Inputs |
|---|---|---|
| `smooth_l1_loss_backward_none_kernel` | `reduction == 0` (none) | `grad_output` is per-element |
| `smooth_l1_loss_backward_reduced_kernel` | `reduction == mean` (inv_N = 1/M) or `sum` (inv_N = 1.0) | `grad_output` is a scalar |

Both compute the same analytic gradient:
```
∂loss / ∂x = (x - y) / beta              if |x - y| < beta
            sign(x - y)                  otherwise
```
Folded with `grad_output * inv_N` for the reduced cases.

Because the kernel is fully pointwise and dispatches at the aten level,
torch's autograd reuses it for **both** `∂/∂input` and `∂/∂target` — the
latter is just the negation, which is computed by the autograd graph not
by the Triton kernel.

## 4. API alignment

The public Python signature accepts either a string (`"none"`, `"mean"`,
`"sum"`) or an integer enum (`0`, `1`, `2`) for `reduction`.  PyTorch's
aten dispatch always sends the int; user code via `nn.functional` sends
the string.  We resolve transparently in `_resolve_reduction`.

`beta` is asserted positive at the public entrypoint — passing a
non-positive beta would cause divide-by-zero or NaN propagation through
the quadratic branch, which is harder to debug than a clear `assert`.

## 5. Empty-input semantics

`torch.nn.functional.smooth_l1_loss(x, y, reduction='mean')` with empty
`x, y` returns `NaN` (it's `0 / 0`).  `sum` returns `0`.  `none` returns
an empty tensor of the same shape.  All three are preserved exactly.

## 6. Performance characteristics

| Path | Cost model |
|---|---|
| forward `none` | memory-bound — bottleneck is the elementwise read+write |
| forward `mean` / `sum` | memory-bound for phase 1, latency-bound for phase 2 (single block) |
| backward `none` | memory-bound — one read of `grad_output`, two reads of `inp`/`target`, one write |
| backward `mean` / `sum` | as above plus one scalar broadcast |

Measured speedups (see `COVERAGE.md`):
- mean/sum reductions consistently >1.0x on production-relevant shapes
- backward path: same throughput as torch's `aten_backward` since both
  are bandwidth-bound on the same data layout
- the only adverse case is the (128,)/fp32/`none` outlier where torch
  has an optimized small-tensor fast path with lower launch overhead
