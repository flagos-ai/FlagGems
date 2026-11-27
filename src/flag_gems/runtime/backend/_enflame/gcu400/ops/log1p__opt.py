"""log1p candidate v2: launch config tiered by dtype / element count (2026-11-16 perf round).

v1 (the 11-02 TRACKED sweep) hardcoded `BLOCK=32768 / GRID=48 / num_warps=4 / NUM_STAGES=4`
and split by dtype into fp16/bf16 BS65536 nw2 ST1 and fp32 BS8192 nw4 ST4 tiers. This round
(11-16, GCU400/Libra, official bench kernel mode, same do_bench median w1000/r100 protocol) ran
an ablation ladder plus a parameter sweep over 40+ configs x 6 variant bodies. Conclusion:
- **copy floor** (same-form flat gs-cap48 masked kernel with only log removed): fp32 2^24
  48-51us (2.7-2.8TB/s), 2^29 1418us (3.03TB/s), roughly equal to native log1p fp32
  48-51us (2.7-2.8TB/s), 2^29 1418us (3.03TB/s), roughly equal to native log1p fp32
- **The 0.64-0.68x on fp32 medium/large shapes is compiler-level**: `tl.log` on the GCU backend
  becomes an opaque extern call `tops_log_v1_fp32(vector<128xf32>)` (proven in llir) and cannot
  overlap with the 2x-wide fp32 load/store pipeline; BS8192/nw4/ST4 is already the optimum of the
  whole (BS, ns, nw, grid, body) space (75.1us vs a 50.2us floor). Widening the grid
  (96/192/dense), EVEN mask removal and ns1-8 across all tiers gave no gain or regressed.
- **Rejected across the whole body space** (fp32 2^24, versus the extern tl.log 75.1us): the
  `fast_logf` extern matches tl.log config by config (reproduced on the same chain);
  `fast_log2f*ln2` +12%; the accurate `__ef_logf` extern 3.0x; `__ef_log1pf` extern 4.3x; a
  **pure Triton bit decomposition + D9 minimax Horner polynomial (5.96e-8 fp32 emulation error)
  is 10.6x worse** (801.9us) -- the backend materializes intermediates pathologically for
  multi-op element chains with bitcast/where (ablation ladder: pure bit-decomposition body
  82.9us, Horner+select body 119.9us, select-free body 130.9us, and 801.9us once 4 special-value
  selects are added); the lgamma Lanczos pure-body precedent does not transfer to chains with
  bitcast/select. The compiler issue is filed in the P archive format (tops_log_v1_fp32 extern not pipelinable).
- **fp16/bf16 large shapes**: from n >= 2^26, BS131072 overtakes BS65536 (2^29: 785.7 vs
  805.2us, +2.4%; 2^26: 104.5 vs 107.0us; 2^25 ties), so the boundary is 2^26.
- **Small shapes** (n <= 4096): halve BS to `max(512, npo2(n//2))` to spread launch latency over
  more CTAs (4096 elements: fp32 4.0->3.7us, fp16 3.4->3.3us, nw2/ST1); below n <= 512, grid=1
  ties with a smaller BS.

Host-side dtype promotion / non-contiguous / empty-tensor semantics are verbatim-isomorphic to
production `log1p()`. From the 2026-11-16 evening round the `_config` tier table is single-sourced
to production `log1p.py` (imported and reused here), and the `_launch` of `log1p_` / `log1p_out`
applies the same tiers -- inplace/out and functional share the whole (BS, grid, nw, ns) evidence
chain; the inplace large shapes previously stalled at the old TRACKED config BS16384/nw4/ST4
(official bench core 15-cell mean only 0.70x, fp32 2^29 0.46x) and now match functional.

Probe pitfall: `torch.randn()+2.0` still produces x<-1 making log1p return NaN, and
`torch.equal` is always False for NaN, which would misjudge every config as failing;
sweeps and gates must use fully positive inputs.
"""

import torch

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.log1p import _config, log1p_backward
from flag_gems.runtime.backend._enflame.gcu400.ops.log1p import (
    log1p_kernel as _kernel_wrapped,
)

_K = _kernel_wrapped.fn

__all__ = ["log1p", "Log1pFunction"]


class Log1pFunction(torch.autograd.Function):
    """Autograd wrapper that gives log1p its fused first-order backward.

    ``aten::log1p`` is CompositeImplicitAutograd, so serving only the forward
    kernel leaves the backward to ATen's composite ``1 + x`` add followed by a
    division. Saving ``x`` and launching the fused ``log1p_backward_kernel``
    keeps the whole path inside one pass; higher-order gradients are not built.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _log1p_functional(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return log1p_backward(grad_output, x)


def _log1p_functional(x: torch.Tensor):
    if x.is_floating_point():
        result_dtype = x.dtype
    else:
        result_dtype = torch.float32

    x_contig = x.to(result_dtype).contiguous()
    out = torch.empty_like(x_contig)
    n = out.numel()
    if n == 0:
        return out.view_as(x)

    BS, G, nw, ST = _config(n, result_dtype)
    grid = (G, 1, 1)
    with torch_device_fn.device(out.device.index):
        _K[grid](
            x_contig.view(-1),
            out.view(-1),
            n,
            BLOCK_SIZE=BS,
            GRID_SIZE=G,
            NUM_STAGES=ST,
            num_warps=nw,
        )
    return out.view_as(x)


def log1p(x: torch.Tensor):
    if torch.is_grad_enabled() and x.requires_grad:
        return Log1pFunction.apply(x)
    return _log1p_functional(x)
