import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

_tanh = tl_extra_shim.tanh

# The GENERIC ops/_thnn_fused_lstm_cell_backward_impl.py is a PURE-TORCH composite:
# it slices the workspace into 4 gate views, then runs ~10 elementwise ops
# (tanh / mul / rsub / add), a torch.cat and a sum(dim=0). Under use_gems every one
# of those decomposes into a separate gems Triton launch. On the XPU triton fork the
# IR dump (ir-thnn_fused_lstm_cell_backward_impl-dev5.log) blows up to 739K lines /
# 11077 kernel modules (mul 2256, sum 1225, copy/cat 1175, rsub 1058, add 423,
# tanh 376), and on the benchmark's tiny (batch<=16, hidden<=64) shapes the per-launch
# overhead dominates -> catastrophic latency.
#
# Fix: fuse the WHOLE elementwise backward (10 pointwise ops + the torch.cat) into ONE
# @libentry Triton kernel that reads all inputs and the 4 gate slices in a single pass
# and writes grad_input_gates (the cat result, straight into the 4 column bands) +
# grad_cx. The bias gradient stays a single grad_input_gates.sum(dim=0) (one cached
# gems reduction). @libentry caches the compiled kernel and BLOCK/num_warps are passed
# EXPLICITLY (never via @triton.heuristics) so there is no per-launch recompile.
# Algorithm is byte-identical to the generic chain rule.


@libentry()
@triton.jit(do_not_specialize=["N", "H"])
def _lstm_cell_bwd_kernel(
    grad_hy_ptr,
    grad_cy_ptr,
    cx_ptr,
    cy_ptr,
    workspace_ptr,
    grad_gates_ptr,
    grad_cx_ptr,
    N,
    H,
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
    if N > 0:
        BLOCK = min(triton.next_power_of_2(N), 256)
        grid = (triton.cdiv(N, BLOCK),)
        with torch_device_fn.device(cx.device):
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
        grad_biases = grad_input_gates.sum(dim=0)
    else:
        grad_biases = torch.zeros(0, dtype=cx.dtype, device=cx.device)

    return grad_input_gates, grad_cx, grad_biases
