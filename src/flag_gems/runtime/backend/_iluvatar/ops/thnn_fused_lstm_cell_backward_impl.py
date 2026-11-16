# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl

# tanh: prefer the backend-neutral libdevice entry, with fallbacks for older
# triton layouts.
try:
    from triton.language.extra import libdevice as _libdevice

    _tanh = _libdevice.tanh
except Exception:  # pragma: no cover
    try:
        from triton.language.extra.cuda import libdevice as _libdevice_cuda

        _tanh = _libdevice_cuda.tanh
    except Exception:  # pragma: no cover
        _tanh = tl.math.tanh


@triton.jit
def _ld(ptr, offs, mask, MASKED: tl.constexpr):
    if MASKED:
        return tl.load(ptr + offs, mask=mask, other=0.0)
    return tl.load(ptr + offs)


@triton.jit
def _st(ptr, offs, val, mask, MASKED: tl.constexpr):
    if MASKED:
        tl.store(ptr + offs, val, mask=mask)
    else:
        tl.store(ptr + offs, val)


@triton.jit
def _lstm_bwd_fused_kernel(
    grad_hy_ptr,
    grad_cy_ptr,
    cx_ptr,
    cy_ptr,
    workspace_ptr,
    grad_gates_ptr,
    grad_cx_ptr,
    grad_bias_ptr,
    N,
    H: tl.constexpr,
    N2: tl.constexpr,
    CH: tl.constexpr,
    MASKED: tl.constexpr,
):
    """Fused LSTM-cell backward in one kernel.

    grid = ceil(H / CH).  Program ``pid`` owns hidden columns
    ``[pid*CH, (pid+1)*CH)`` of every batch row.  For each (batch, hidden)
    element it loads the four gate activations from the workspace (layout
    ``[i, f, g, o]`` slabs of ``H`` contiguous columns per row), computes the
    chain-rule gate gradients and ``grad_cx`` in fp32, stores them, and
    accumulates the per-column sums over the batch dimension in registers so
    the bias gradient needs no second kernel, no atomics, and no memset.

    ``H``, ``N2``, ``CH``, ``MASKED`` are compile-time: H is a power of two in
    every benchmark shape, so the row-stride products fold to shifts, and the
    ``MASKED=False`` specialization drops all mask predicates.
    """
    pid = tl.program_id(0)
    h = pid * CH + tl.arange(0, CH)
    r = tl.arange(0, N2)
    rr = r[:, None]
    hh = h[None, :]

    if MASKED:
        m2 = (r < N)[:, None] & (h < H)[None, :]
    else:
        m2 = None

    ws = rr * (4 * H) + hh
    i_gate = _ld(workspace_ptr, ws, m2, MASKED).to(tl.float32)
    f_gate = _ld(workspace_ptr, ws + H, m2, MASKED).to(tl.float32)
    g_gate = _ld(workspace_ptr, ws + 2 * H, m2, MASKED).to(tl.float32)
    o_gate = _ld(workspace_ptr, ws + 3 * H, m2, MASKED).to(tl.float32)

    e = rr * H + hh
    ghy = _ld(grad_hy_ptr, e, m2, MASKED).to(tl.float32)
    gcy = _ld(grad_cy_ptr, e, m2, MASKED).to(tl.float32)
    cxv = _ld(cx_ptr, e, m2, MASKED).to(tl.float32)
    cyv = _ld(cy_ptr, e, m2, MASKED).to(tl.float32)

    tanh_cy = _tanh(cyv)
    d_cy = ghy * o_gate * (1.0 - tanh_cy * tanh_cy) + gcy

    grad_i = d_cy * g_gate * i_gate * (1.0 - i_gate)
    grad_f = d_cy * cxv * f_gate * (1.0 - f_gate)
    grad_g = d_cy * i_gate * (1.0 - g_gate * g_gate)
    grad_o = ghy * tanh_cy * o_gate * (1.0 - o_gate)
    grad_cx = d_cy * f_gate

    out_ty = grad_gates_ptr.dtype.element_ty
    g0 = rr * (4 * H) + hh
    _st(grad_gates_ptr, g0, grad_i.to(out_ty), m2, MASKED)
    _st(grad_gates_ptr, g0 + H, grad_f.to(out_ty), m2, MASKED)
    _st(grad_gates_ptr, g0 + 2 * H, grad_g.to(out_ty), m2, MASKED)
    _st(grad_gates_ptr, g0 + 3 * H, grad_o.to(out_ty), m2, MASKED)
    _st(grad_cx_ptr, e, grad_cx.to(grad_cx_ptr.dtype.element_ty), m2, MASKED)

    b_ty = grad_bias_ptr.dtype.element_ty
    cm = (h < H) if MASKED else None
    _st(grad_bias_ptr, h, tl.sum(grad_i, axis=0).to(b_ty), cm, MASKED)
    _st(grad_bias_ptr, H + h, tl.sum(grad_f, axis=0).to(b_ty), cm, MASKED)
    _st(grad_bias_ptr, 2 * H + h, tl.sum(grad_g, axis=0).to(b_ty), cm, MASKED)
    _st(grad_bias_ptr, 3 * H + h, tl.sum(grad_o, axis=0).to(b_ty), cm, MASKED)


def _thnn_fused_lstm_cell_backward_impl(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    N, H = cx.shape
    dtype = grad_hy.dtype
    device = grad_hy.device

    # grad_input_gates and grad_biases share one allocation; grad_cx is a
    # second.  The kernel writes them at disjoint offsets.
    gates_elems = N * (4 * H)
    buf = torch.empty((gates_elems + max(4 * H, 1)), dtype=dtype, device=device)
    grad_input_gates = buf[:gates_elems].view(N, 4 * H)
    bias_buf = buf[gates_elems:]
    grad_cx = torch.empty_like(cx)

    if N > 0 and H > 0:
        N2 = triton.next_power_of_2(N)
        # Target ~64 elements per program (2/thread with one warp): CH =
        # 64//N2.  For batch==1 this grows the chunk to H (one program per
        # row), which two independent A/Bs measured slightly faster than the
        # capped variant on the (1,64) tile.
        CH = min(H, 64 // N2)
        MASKED = (N != N2) or (H % CH != 0)
        grid = (triton.cdiv(H, CH),)
        _lstm_bwd_fused_kernel[grid](
            grad_hy,
            grad_cy,
            cx,
            cy,
            workspace,
            grad_input_gates,
            grad_cx,
            bias_buf,
            N,
            H=H,
            N2=N2,
            CH=CH,
            MASKED=MASKED,
            num_warps=1,
        )

    if has_bias:
        grad_biases = bias_buf
    else:
        grad_biases = torch.empty(0, dtype=dtype, device=device)
    return grad_input_gates, grad_cx, grad_biases
