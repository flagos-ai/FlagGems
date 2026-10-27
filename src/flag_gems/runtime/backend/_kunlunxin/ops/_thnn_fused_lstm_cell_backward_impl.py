import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

# The GENERIC ops/_thnn_fused_lstm_cell_backward_impl.py is a PURE-TORCH composite:
# it slices the workspace into 4 gate views, then runs ~10 elementwise ops
# (tanh / mul / rsub / add), a torch.cat and a sum(dim=0). Under use_gems every one
# of those decomposes into a separate gems Triton launch, and on the tiny
# (batch<=16, hidden<=64) benchmark shapes the per-launch overhead dominates.
#
# Fix: fuse the WHOLE elementwise backward (10 pointwise ops + the torch.cat) into ONE
# @libentry Triton kernel that reads all inputs and the 4 gate slices in a single pass
# and writes grad_input_gates (the cat result, straight into the 4 column bands) +
# grad_cx. The bias gradient used to be a single grad_input_gates.sum(dim=0) (one
# cached gems reduction); it is now a dedicated lean @libentry reduction kernel
# (`_bias_grad_kernel`, one masked BLOCK_M=128 tile, fp32 accumulation, B-row
# serial loop): single launch, no aten `sum` dispatch. @libentry caches the compiled
# kernel and BLOCK/num_warps are passed EXPLICITLY (never via @triton.heuristics) so
# there is no per-launch recompile. Algorithm is byte-identical to the generic chain
# rule.
#
# The op is launch-bound. Two dominant costs inside the kernel were found and
# removed: (1) integer div/mod (`b = offs // H`, `h = offs % H`) is expensive on
# XPU when H is a runtime scalar, so the small-batch kernels and the general bwd
# kernel now take H as a constexpr; (2) the bias reduction cannot be lowered by
# this compiler (a 2D `tl.reduce` after `tl.trans` hits UNREACHABLE, and axis-0
# reduce is rejected). So batch==1 (bias == the grads themselves) and batch==2
# (bias == one elementwise add of the two rows) are specialized kernels with NO
# reduction; batch>=3 keeps the grid-parallel bwd + the serial-fold bias kernel.

_tanh = tl_extra_shim.tanh

# Lane width of the batch==1/2 single-program kernels. They cover one row with
# `tl.arange(0, BLOCK)`, so they are only dispatched for hidden_size <= BLOCK;
# wider rows go to the grid-parallel kernel.
_SMALL_BATCH_BLOCK = 256


@libentry()
@triton.jit
def _lstm_cell_bwd_b1(
    grad_hy_ptr,
    grad_cy_ptr,
    cx_ptr,
    cy_ptr,
    workspace_ptr,
    grad_gates_ptr,
    grad_cx_ptr,
    grad_biases_ptr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    has_bias: tl.constexpr,
):
    # batch==1 fast path: h == offs, so there is no integer div/mod (expensive
    # on XPU when H is a runtime scalar) and grad_biases == the gate grads
    # themselves (a sum over a single row), so the whole backward is one launch
    # with no reduction and no accumulators.
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    i_gate = tl.load(workspace_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    f_gate = tl.load(workspace_ptr + H + offs, mask=mask, other=0.0).to(tl.float32)
    g_gate = tl.load(workspace_ptr + 2 * H + offs, mask=mask, other=0.0).to(tl.float32)
    o_gate = tl.load(workspace_ptr + 3 * H + offs, mask=mask, other=0.0).to(tl.float32)
    cyv = tl.load(cy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ghy = tl.load(grad_hy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gcy = tl.load(grad_cy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    cxv = tl.load(cx_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    tanh_cy = _tanh(cyv)
    d_cy = ghy * o_gate * (1.0 - tanh_cy * tanh_cy) + gcy
    grad_i = d_cy * g_gate * i_gate * (1.0 - i_gate)
    grad_f = d_cy * cxv * f_gate * (1.0 - f_gate)
    grad_g = d_cy * i_gate * (1.0 - g_gate * g_gate)
    grad_o = ghy * tanh_cy * o_gate * (1.0 - o_gate)
    grad_cx = d_cy * f_gate

    ty = grad_gates_ptr.dtype.element_ty
    tl.store(grad_gates_ptr + offs, grad_i.to(ty), mask=mask)
    tl.store(grad_gates_ptr + H + offs, grad_f.to(ty), mask=mask)
    tl.store(grad_gates_ptr + 2 * H + offs, grad_g.to(ty), mask=mask)
    tl.store(grad_gates_ptr + 3 * H + offs, grad_o.to(ty), mask=mask)
    tl.store(grad_cx_ptr + offs, grad_cx.to(grad_cx_ptr.dtype.element_ty), mask=mask)

    if has_bias:
        ty_b = grad_biases_ptr.dtype.element_ty
        tl.store(grad_biases_ptr + offs, grad_i.to(ty_b), mask=mask)
        tl.store(grad_biases_ptr + H + offs, grad_f.to(ty_b), mask=mask)
        tl.store(grad_biases_ptr + 2 * H + offs, grad_g.to(ty_b), mask=mask)
        tl.store(grad_biases_ptr + 3 * H + offs, grad_o.to(ty_b), mask=mask)


@libentry()
@triton.jit
def _lstm_cell_bwd_b2(
    grad_hy_ptr,
    grad_cy_ptr,
    cx_ptr,
    cy_ptr,
    workspace_ptr,
    grad_gates_ptr,
    grad_cx_ptr,
    grad_biases_ptr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    has_bias: tl.constexpr,
):
    # batch==2 fast path: the two rows are unrolled manually so there is no
    # runtime loop, no integer div/mod, and the bias reduction is a single
    # elementwise add of the two rows' grads -- no tl.reduce, which the XPU
    # compiler cannot lower across a tl.trans. BLOCK=128/num_warps=4 fastest.
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    i0 = tl.load(workspace_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    f0 = tl.load(workspace_ptr + H + offs, mask=mask, other=0.0).to(tl.float32)
    g0 = tl.load(workspace_ptr + 2 * H + offs, mask=mask, other=0.0).to(tl.float32)
    o0 = tl.load(workspace_ptr + 3 * H + offs, mask=mask, other=0.0).to(tl.float32)
    cy0 = tl.load(cy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gh0 = tl.load(grad_hy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gc0 = tl.load(grad_cy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    cx0 = tl.load(cx_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    i1 = tl.load(workspace_ptr + 4 * H + offs, mask=mask, other=0.0).to(tl.float32)
    f1 = tl.load(workspace_ptr + 5 * H + offs, mask=mask, other=0.0).to(tl.float32)
    g1 = tl.load(workspace_ptr + 6 * H + offs, mask=mask, other=0.0).to(tl.float32)
    o1 = tl.load(workspace_ptr + 7 * H + offs, mask=mask, other=0.0).to(tl.float32)
    cy1 = tl.load(cy_ptr + H + offs, mask=mask, other=0.0).to(tl.float32)
    gh1 = tl.load(grad_hy_ptr + H + offs, mask=mask, other=0.0).to(tl.float32)
    gc1 = tl.load(grad_cy_ptr + H + offs, mask=mask, other=0.0).to(tl.float32)
    cx1 = tl.load(cx_ptr + H + offs, mask=mask, other=0.0).to(tl.float32)

    t0 = _tanh(cy0)
    t1 = _tanh(cy1)
    d0 = gh0 * o0 * (1.0 - t0 * t0) + gc0
    d1 = gh1 * o1 * (1.0 - t1 * t1) + gc1
    gi0 = d0 * g0 * i0 * (1.0 - i0)
    gi1 = d1 * g1 * i1 * (1.0 - i1)
    gf0 = d0 * cx0 * f0 * (1.0 - f0)
    gf1 = d1 * cx1 * f1 * (1.0 - f1)
    gg0 = d0 * i0 * (1.0 - g0 * g0)
    gg1 = d1 * i1 * (1.0 - g1 * g1)
    go0 = gh0 * t0 * o0 * (1.0 - o0)
    go1 = gh1 * t1 * o1 * (1.0 - o1)
    gcx0 = d0 * f0
    gcx1 = d1 * f1

    ty = grad_gates_ptr.dtype.element_ty
    tl.store(grad_gates_ptr + offs, gi0.to(ty), mask=mask)
    tl.store(grad_gates_ptr + H + offs, gf0.to(ty), mask=mask)
    tl.store(grad_gates_ptr + 2 * H + offs, gg0.to(ty), mask=mask)
    tl.store(grad_gates_ptr + 3 * H + offs, go0.to(ty), mask=mask)
    tl.store(grad_gates_ptr + 4 * H + offs, gi1.to(ty), mask=mask)
    tl.store(grad_gates_ptr + 5 * H + offs, gf1.to(ty), mask=mask)
    tl.store(grad_gates_ptr + 6 * H + offs, gg1.to(ty), mask=mask)
    tl.store(grad_gates_ptr + 7 * H + offs, go1.to(ty), mask=mask)
    tl.store(grad_cx_ptr + offs, gcx0.to(grad_cx_ptr.dtype.element_ty), mask=mask)
    tl.store(grad_cx_ptr + H + offs, gcx1.to(grad_cx_ptr.dtype.element_ty), mask=mask)

    if has_bias:
        ty_b = grad_biases_ptr.dtype.element_ty
        tl.store(grad_biases_ptr + offs, (gi0 + gi1).to(ty_b), mask=mask)
        tl.store(grad_biases_ptr + H + offs, (gf0 + gf1).to(ty_b), mask=mask)
        tl.store(grad_biases_ptr + 2 * H + offs, (gg0 + gg1).to(ty_b), mask=mask)
        tl.store(grad_biases_ptr + 3 * H + offs, (go0 + go1).to(ty_b), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _lstm_cell_bwd_kernel(
    grad_hy_ptr,
    grad_cy_ptr,
    cx_ptr,
    cy_ptr,
    workspace_ptr,
    grad_gates_ptr,
    grad_cx_ptr,
    N,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    b = offs // H
    h = offs % H
    ws_row = b * (4 * H)

    i_gate = tl.load(workspace_ptr + ws_row + h, mask=mask, other=0.0).to(tl.float32)
    f_gate = tl.load(workspace_ptr + ws_row + H + h, mask=mask, other=0.0).to(
        tl.float32
    )
    g_gate = tl.load(workspace_ptr + ws_row + 2 * H + h, mask=mask, other=0.0).to(
        tl.float32
    )
    o_gate = tl.load(workspace_ptr + ws_row + 3 * H + h, mask=mask, other=0.0).to(
        tl.float32
    )
    ghy = tl.load(grad_hy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gcy = tl.load(grad_cy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    cxv = tl.load(cx_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    cyv = tl.load(cy_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    tanh_cy = _tanh(cyv)
    d_cy = ghy * o_gate * (1.0 - tanh_cy * tanh_cy) + gcy

    grad_i = d_cy * g_gate * i_gate * (1.0 - i_gate)
    grad_f = d_cy * cxv * f_gate * (1.0 - f_gate)
    grad_g = d_cy * i_gate * (1.0 - g_gate * g_gate)
    grad_o = ghy * tanh_cy * o_gate * (1.0 - o_gate)
    grad_cx = d_cy * f_gate

    out_row = b * (4 * H)
    ty = grad_gates_ptr.dtype.element_ty
    tl.store(grad_gates_ptr + out_row + h, grad_i.to(ty), mask=mask)
    tl.store(grad_gates_ptr + out_row + H + h, grad_f.to(ty), mask=mask)
    tl.store(grad_gates_ptr + out_row + 2 * H + h, grad_g.to(ty), mask=mask)
    tl.store(grad_gates_ptr + out_row + 3 * H + h, grad_o.to(ty), mask=mask)
    tl.store(grad_cx_ptr + offs, grad_cx.to(grad_cx_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["B", "M"])
def _bias_grad_kernel(
    grad_gates_ptr,
    grad_biases_ptr,
    B,
    M,
    BLOCK_M: tl.constexpr,
):
    # grad_biases[j] = sum_b grad_gates[b, j]  for j in [0, M)
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs < M
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for b in range(B):
        acc += tl.load(grad_gates_ptr + b * M + offs, mask=m_mask, other=0.0)
    tl.store(
        grad_biases_ptr + offs,
        acc.to(grad_biases_ptr.dtype.element_ty),
        mask=m_mask,
    )


def _thnn_fused_lstm_cell_backward_impl(
    grad_hy: torch.Tensor,
    grad_cy: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    workspace: torch.Tensor,
    has_bias: bool,
):
    logger.debug("GEMS_KUNLUNXIN _THNN_FUSED_LSTM_CELL_BACKWARD_IMPL")

    batch_size, hidden_size = cx.shape

    grad_hy = grad_hy.contiguous()
    grad_cy = grad_cy.contiguous()
    cx = cx.contiguous()
    cy = cy.contiguous()
    workspace = workspace.contiguous()

    grad_input_gates = torch.empty(
        (batch_size, 4 * hidden_size), device=cx.device, dtype=cx.dtype
    )
    grad_cx = torch.empty((batch_size, hidden_size), device=cx.device, dtype=cx.dtype)

    N = batch_size * hidden_size
    if has_bias:
        grad_biases = torch.empty((4 * hidden_size,), device=cx.device, dtype=cx.dtype)
    else:
        grad_biases = torch.zeros(0, dtype=cx.dtype, device=cx.device)
    with torch_device_fn.device(cx.device):
        if N > 0 and batch_size == 1 and hidden_size <= _SMALL_BATCH_BLOCK:
            # Single-row fast path: no div/mod, no reduction, no accumulators.
            # Gated on H: the kernel is one program over `tl.arange(0, BLOCK)`,
            # so a wider row would silently drop every lane past BLOCK.
            _lstm_cell_bwd_b1[(1,)](
                grad_hy,
                grad_cy,
                cx,
                cy,
                workspace,
                grad_input_gates,
                grad_cx,
                grad_biases,
                H=hidden_size,
                BLOCK=_SMALL_BATCH_BLOCK,
                has_bias=has_bias,
                num_warps=4,
            )
        elif N > 0 and batch_size == 2 and hidden_size <= _SMALL_BATCH_BLOCK:
            # Two-row fast path: manual unroll, bias is a single elementwise add.
            # Same H gate as the single-row kernel above.
            _lstm_cell_bwd_b2[(1,)](
                grad_hy,
                grad_cy,
                cx,
                cy,
                workspace,
                grad_input_gates,
                grad_cx,
                grad_biases,
                H=hidden_size,
                BLOCK=_SMALL_BATCH_BLOCK,
                has_bias=has_bias,
                num_warps=4,
            )
        else:
            if N > 0:
                # BLOCK is tl.constexpr. Keep it fixed so all shapes reuse one compiled
                # specialization instead of compiling once per shape-dependent block.
                BLOCK = 256
                grid = (triton.cdiv(N, BLOCK),)
                _lstm_cell_bwd_kernel[grid](
                    grad_hy,
                    grad_cy,
                    cx,
                    cy,
                    workspace,
                    grad_input_gates,
                    grad_cx,
                    N,
                    H=hidden_size,
                    BLOCK=BLOCK,
                    num_warps=4,
                )
            if has_bias and batch_size > 0:
                # BLOCK_M=128 OOMs the XPU uni_sram at compile time, so
                # BLOCK_M=64 is used (verified correct for the supported shapes
                # and dtypes).
                BLOCK_M = 64
                grid = (triton.cdiv(4 * hidden_size, BLOCK_M),)
                _bias_grad_kernel[grid](
                    grad_input_gates,
                    grad_biases,
                    batch_size,
                    4 * hidden_size,
                    BLOCK_M,
                    num_warps=4,
                )

    return grad_input_gates, grad_cx, grad_biases
