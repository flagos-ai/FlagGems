import contextlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)


_tanh = tl_extra_shim.tanh


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
@triton.jit
def _lstm_cell_bwd_kernel_exact(
    grad_hy_ptr,
    grad_cy_ptr,
    cx_ptr,
    cy_ptr,
    workspace_ptr,
    grad_gates_ptr,
    grad_cx_ptr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    b = offs // H
    h = offs % H
    ws_row = b * (4 * H)

    i_gate = tl.load(workspace_ptr + ws_row + h).to(tl.float32)
    f_gate = tl.load(workspace_ptr + ws_row + H + h).to(tl.float32)
    g_gate = tl.load(workspace_ptr + ws_row + 2 * H + h).to(tl.float32)
    o_gate = tl.load(workspace_ptr + ws_row + 3 * H + h).to(tl.float32)
    ghy = tl.load(grad_hy_ptr + offs).to(tl.float32)
    gcy = tl.load(grad_cy_ptr + offs).to(tl.float32)
    cxv = tl.load(cx_ptr + offs).to(tl.float32)
    cyv = tl.load(cy_ptr + offs).to(tl.float32)

    tanh_cy = _tanh(cyv)
    d_cy = ghy * o_gate * (1.0 - tanh_cy * tanh_cy) + gcy

    grad_i = d_cy * g_gate * i_gate * (1.0 - i_gate)
    grad_f = d_cy * cxv * f_gate * (1.0 - f_gate)
    grad_g = d_cy * i_gate * (1.0 - g_gate * g_gate)
    grad_o = ghy * tanh_cy * o_gate * (1.0 - o_gate)
    grad_cx = d_cy * f_gate

    out_row = b * (4 * H)
    ty = grad_gates_ptr.dtype.element_ty
    tl.store(grad_gates_ptr + out_row + h, grad_i.to(ty))
    tl.store(grad_gates_ptr + out_row + H + h, grad_f.to(ty))
    tl.store(grad_gates_ptr + out_row + 2 * H + h, grad_g.to(ty))
    tl.store(grad_gates_ptr + out_row + 3 * H + h, grad_o.to(ty))
    tl.store(grad_cx_ptr + offs, grad_cx.to(grad_cx_ptr.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["M"])
def _bias_grad_kernel(
    grad_gates_ptr,
    grad_biases_ptr,
    B: tl.constexpr,
    M,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs < M
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for b in tl.static_range(B):
        acc += tl.load(grad_gates_ptr + b * M + offs, mask=m_mask, other=0.0)
    tl.store(
        grad_biases_ptr + offs,
        acc.to(grad_biases_ptr.dtype.element_ty),
        mask=m_mask,
    )


@libentry()
@triton.jit
def _bias_grad_kernel_exact(
    grad_gates_ptr,
    grad_biases_ptr,
    B: tl.constexpr,
    M: tl.constexpr,
):
    offs = tl.arange(0, M)
    acc = tl.zeros((M,), dtype=tl.float32)
    for b in tl.static_range(B):
        acc += tl.load(grad_gates_ptr + b * M + offs)
    tl.store(grad_biases_ptr + offs, acc.to(grad_biases_ptr.dtype.element_ty))


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
    use_guard = torch_device_fn.current_device() != cx.device.index
    with torch_device_fn.device(cx.device) if use_guard else contextlib.nullcontext():
        if N > 0:
            if 256 <= N <= 2048 and (N & (N - 1)) == 0:
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
                    hidden_size,
                    BLOCK,
                    num_warps=4,
                )
        if has_bias:
            grad_biases = torch.empty(
                (4 * hidden_size,), device=cx.device, dtype=cx.dtype
            )
            if batch_size > 0:
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
                    BLOCK_M = 1024
                    grid = (triton.cdiv(4 * hidden_size, BLOCK_M),)
                    _bias_grad_kernel[grid](
                        grad_input_gates,
                        grad_biases,
                        batch_size,
                        4 * hidden_size,
                        BLOCK_M,
                        num_warps=4,
                    )
        else:
            grad_biases = torch.zeros(0, dtype=cx.dtype, device=cx.device)

    return grad_input_gates, grad_cx, grad_biases
