import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

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

    N = batch_size * hidden_size
    use_guard = torch_device_fn.current_device() != cx.device.index
    with torch_device_fn.device(cx.device) if use_guard else contextlib.nullcontext():
        # For batch_size == 1 the bias is just grad_input_gates row 0, so fuse the
        # per-element backward and the bias copy into a single launch (saves a launch).
        fuse_b1 = has_bias and batch_size == 1 and N < 256
        if N > 0:
            if fuse_b1:
                # A 256-lane masked tile measured cheaper than 512 on this backend.
                BLOCK = 256
                grid = (triton.cdiv(N, BLOCK),)
                _lstm_cell_bwd_b1_kernel[grid](
                    grad_hy,
                    grad_cy,
                    cx,
                    cy,
                    workspace,
                    grad_input_gates,
                    grad_cx,
                    grad_biases,
                    N,
                    hidden_size,
                    BLOCK,
                    num_warps=4,
                )
            elif 256 <= N <= 2048 and (N & (N - 1)) == 0:
                grid = (1,)
                _lstm_cell_bwd_kernel_exact[grid](
                    grad_hy,
                    grad_cy,
                    cx,
                    cy,
                    workspace,
                    grad_input_gates,
                    grad_cx,
                    hidden_size,
                    N,
                    num_warps=4,
                )
            else:
                BLOCK = 512
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
        # batch_size == 0 still needs grad_biases filled (ATen returns zeros);
        # both bias kernels reduce an empty batch to a zero accumulator.
        if has_bias and not fuse_b1:
            if 4 * hidden_size == 256:
                grid = (1,)
                _bias_grad_kernel_exact[grid](
                    grad_input_gates,
                    grad_biases,
                    batch_size,
                    4 * hidden_size,
                    num_warps=4,
                )
            else:
                # One 256-column program covers every bias width this op sees;
                # the wider masked tile only added empty lanes.
                BLOCK_M = 256
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
