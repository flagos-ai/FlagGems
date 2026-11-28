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

"""GCU400-specific ``cudnn_rnn_backward`` (single-layer unidirectional LSTM BPTT).

The common implementation schedules step by step: per step 1 forward Triton
kernel, 1 backward Triton kernel, 6 BLAS calls, 7 tensor allocations. On GCU
the fixed cost of these launches / allocs far exceeds the actual compute for
small shapes. This implementation keeps the two Triton kernels as the core
compute path while compressing the host scheduling around them:

* one GEMM computes the whole input projection with both biases folded in
  (``torch.addmm`` over the flattened (T*B, I) view), so each step needs a
  single ``addmm`` instead of GEMM + bias + add;
* the gate / state caches are allocated once with a leading slot for the
  initial h0/c0, making every per-step view a pure prefix slice (no stack,
  no per-step list bookkeeping, no clone to protect the caller's grad_cy --
  the backward kernel writes dc_prev to a dedicated buffer);
* the recurrent GEMMs write into preallocated ping-pong workspaces
  (``out=``), removing per-step allocator churn;
* the kernels take the cache base pointers plus a scalar step offset, so the
  host passes two integers instead of seven ``select()`` views per step;
* weight / bias / input gradients are reduced once after the loop with flat
  ``mm`` / ``sum`` on the (T*B, 4H) views -- 3x cheaper than the einsum+stack
  chain on this backend.

``BASELINE_MODE=1`` forwards directly to the common implementation for
collecting an equivalent baseline; after optimization the main path is the
fused scheduling version above.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Compile-time config, adjusted per optimization round; semantics unchanged.
BASELINE_MODE = 0
BLOCK_SIZE_SCALE = 1
NUM_WARPS_FWD = 4
NUM_WARPS_BWD = 8
# --- TUNING PARAMS END ---


@libentry()
@triton.jit
def _lstm_cell_forward_kernel(
    pre_gates_ptr,
    i_ptr,
    f_ptr,
    g_ptr,
    o_ptr,
    c_ptr,
    h_ptr,
    t_off,
    step,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-step forward: convert the four pre-activation gates into activations and write c/h.

    ``i/f/g/o`` are (T, B, H) caches written at ``t_off``; ``c/h`` are (T+1, B, H)
    caches reading the incoming state at ``t_off`` and writing the new one at
    ``t_off + step``. One program per batch row."""
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    gate_base = pre_gates_ptr + batch_idx * 4 * hidden_size

    i_pre = tl.load(gate_base + 0 * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    f_pre = tl.load(gate_base + 1 * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    g_pre = tl.load(gate_base + 2 * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    o_pre = tl.load(gate_base + 3 * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    row = t_off + batch_idx * hidden_size + offsets
    c_prev = tl.load(c_ptr + row, mask=mask, other=0.0).to(tl.float32)

    i = tl.sigmoid(i_pre)
    f = tl.sigmoid(f_pre)
    g = tl_extra_shim.tanh(g_pre)
    o = tl.sigmoid(o_pre)
    c = f * c_prev + i * g
    h = o * tl_extra_shim.tanh(c)

    tl.store(i_ptr + row, i, mask=mask)
    tl.store(f_ptr + row, f, mask=mask)
    tl.store(g_ptr + row, g, mask=mask)
    tl.store(o_ptr + row, o, mask=mask)
    tl.store(c_ptr + row + step, c, mask=mask)
    tl.store(h_ptr + row + step, h, mask=mask)


@libentry()
@triton.jit
def _lstm_cell_backward_kernel(
    i_ptr,
    f_ptr,
    g_ptr,
    o_ptr,
    c_ptr,
    dh_ptr,
    dc_in_ptr,
    dc_out_ptr,
    grad_out_ptr,
    grad_gates_ptr,
    has_grad_out: tl.constexpr,
    t_off,
    step,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-step backward: chain-rule gate gradients from the cached activations.

    Reads the gate caches at ``t_off`` and the cell cache at ``t_off`` (c_prev)
    / ``t_off + step`` (c_t); consumes the recurrent dh / dc and emits the
    pre-activation gate gradients into the (T, B, 4H) cache at ``4 * t_off``.
    ``dc_prev`` goes to the dedicated ``dc_out`` buffer so the caller's
    gradient tensors are never written in place. One program per batch row."""
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    row = batch_idx * hidden_size + offsets

    i = tl.load(i_ptr + t_off + row, mask=mask, other=0.0).to(tl.float32)
    f = tl.load(f_ptr + t_off + row, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + t_off + row, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(o_ptr + t_off + row, mask=mask, other=0.0).to(tl.float32)
    c_prev = tl.load(c_ptr + t_off + row, mask=mask, other=0.0).to(tl.float32)
    c_t = tl.load(c_ptr + t_off + step + row, mask=mask, other=0.0).to(tl.float32)
    dh = tl.load(dh_ptr + row, mask=mask, other=0.0).to(tl.float32)
    dc = tl.load(dc_in_ptr + row, mask=mask, other=0.0).to(tl.float32)
    if has_grad_out:
        dh = dh + tl.load(grad_out_ptr + t_off + row, mask=mask, other=0.0).to(
            tl.float32
        )

    tanh_c = tl_extra_shim.tanh(c_t)
    do_pre = dh * tanh_c * o * (1.0 - o)
    dc = dc + dh * o * (1.0 - tanh_c * tanh_c)
    di_pre = dc * g * i * (1.0 - i)
    df_pre = dc * c_prev * f * (1.0 - f)
    dg_pre = dc * i * (1.0 - g * g)
    dc_prev = dc * f

    gate_base = grad_gates_ptr + 4 * t_off + batch_idx * 4 * hidden_size
    tl.store(gate_base + 0 * hidden_size + offsets, di_pre, mask=mask)
    tl.store(gate_base + 1 * hidden_size + offsets, df_pre, mask=mask)
    tl.store(gate_base + 2 * hidden_size + offsets, dg_pre, mask=mask)
    tl.store(gate_base + 3 * hidden_size + offsets, do_pre, mask=mask)
    tl.store(dc_out_ptr + row, dc_prev, mask=mask)


def _lstm_block_size(hidden_size: int) -> int:
    """BLOCK_SIZE covers a whole hidden row; increasing it by scale changes per-warp work."""
    block = triton.next_power_of_2(hidden_size) * BLOCK_SIZE_SCALE
    return max(32, min(triton.next_power_of_2(block), 4096))


def _public_baseline(*args, **kwargs):
    from flag_gems.ops.cudnn_rnn_backward import cudnn_rnn_backward as public_fn

    return public_fn(*args, **kwargs)


def cudnn_rnn_backward(
    input: torch.Tensor,
    weight: list,
    weight_stride0: int,
    weight_buf: torch.Tensor,
    hx: torch.Tensor,
    cx: torch.Tensor,
    output: torch.Tensor,
    grad_output_in: torch.Tensor,
    grad_hy_in: torch.Tensor,
    grad_cy_in: torch.Tensor,
    mode: int,
    hidden_size: int,
    proj_size: int,
    num_layers: int,
    batch_first: bool,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_sizes: list,
    dropout_state: torch.Tensor,
    reserve: torch.Tensor,
    output_mask: list,
):
    """BPTT gradients of a single-layer unidirectional LSTM; signature matches the common implementation."""
    logger.debug("GEMS_ENFLAME CUDNN_RNN_BACKWARD")
    if BASELINE_MODE == 1:
        return _public_baseline(
            input,
            weight,
            weight_stride0,
            weight_buf,
            hx,
            cx,
            output,
            grad_output_in,
            grad_hy_in,
            grad_cy_in,
            mode,
            hidden_size,
            proj_size,
            num_layers,
            batch_first,
            dropout,
            train,
            bidirectional,
            batch_sizes,
            dropout_state,
            reserve,
            output_mask,
        )

    if mode != 2:
        raise NotImplementedError("Only LSTM (mode=2) is currently supported")
    if num_layers != 1:
        raise NotImplementedError("Only single-layer RNN is currently supported")
    if bidirectional:
        raise NotImplementedError("Bidirectional RNN is not currently supported")
    if proj_size != 0:
        raise NotImplementedError("Projection is not currently supported")
    if batch_sizes:
        raise NotImplementedError("Packed sequences are not currently supported")

    w_ih, w_hh, b_ih, b_hh = weight[0], weight[1], weight[2], weight[3]
    inp = input.transpose(0, 1).contiguous() if batch_first else input.contiguous()
    seq_len, batch_size, input_size = inp.shape
    orig_dtype = input.dtype
    compute_dtype = torch.float32
    inp_c = inp.to(compute_dtype)
    w_ih_c = w_ih.to(compute_dtype)
    w_hh_c = w_hh.to(compute_dtype)
    b_ih_c = b_ih.to(compute_dtype)
    b_hh_c = b_hh.to(compute_dtype)
    h0 = hx.squeeze(0).to(compute_dtype).contiguous()
    c0 = cx.squeeze(0).to(compute_dtype).contiguous()
    device = inp_c.device

    block_size = _lstm_block_size(hidden_size)
    grid = (batch_size,)
    step = batch_size * hidden_size
    gates = 4 * hidden_size
    flat_rows = seq_len * batch_size

    # Whole-sequence caches, allocated once: the gate caches hold the per-step
    # activations; the state caches carry a leading slot with the initial
    # h0/c0 so every per-step view is a plain prefix slice of one tensor.
    i_cache = torch.empty(
        (seq_len, batch_size, hidden_size), device=device, dtype=compute_dtype
    )
    f_cache = torch.empty(
        (seq_len, batch_size, hidden_size), device=device, dtype=compute_dtype
    )
    g_cache = torch.empty(
        (seq_len, batch_size, hidden_size), device=device, dtype=compute_dtype
    )
    o_cache = torch.empty(
        (seq_len, batch_size, hidden_size), device=device, dtype=compute_dtype
    )
    c_cache = torch.empty(
        (seq_len + 1, batch_size, hidden_size), device=device, dtype=compute_dtype
    )
    h_cache = torch.empty(
        (seq_len + 1, batch_size, hidden_size), device=device, dtype=compute_dtype
    )
    grad_gates_all = torch.empty(
        (seq_len, batch_size, gates), device=device, dtype=compute_dtype
    )
    c_cache[0].copy_(c0)
    h_cache[0].copy_(h0)

    # One GEMM for the whole input projection with both biases folded in:
    # each step then needs a single addmm against the recurrence only.
    inp2d = inp_c.view(flat_rows, input_size)
    b_sum = b_ih_c + b_hh_c
    input_proj = torch.addmm(b_sum, inp2d, w_ih_c.t()).view(seq_len, batch_size, gates)

    # Forward recompute. The per-step pre-gate GEMM reuses two preallocated
    # ping-pong buffers (the kernel of step t only reads the buffer of step t).
    pre_gates_bufs = [
        torch.empty((batch_size, gates), device=device, dtype=compute_dtype),
        torch.empty((batch_size, gates), device=device, dtype=compute_dtype),
    ]
    w_hh_t = w_hh_c.t()
    for t in range(seq_len):
        pre_gates = pre_gates_bufs[t & 1]
        torch.addmm(input_proj[t], h_cache[t], w_hh_t, out=pre_gates)
        _lstm_cell_forward_kernel[grid](
            pre_gates,
            i_cache,
            f_cache,
            g_cache,
            o_cache,
            c_cache,
            h_cache,
            t * step,
            step,
            hidden_size,
            block_size,
            num_warps=NUM_WARPS_FWD,
        )

    # Backward through time. dh / dc ride two ping-pong pairs; the kernel
    # writes dc_prev to its own buffer, so the caller's gradient tensors are
    # never mutated in place.
    dh = (
        grad_hy_in.squeeze(0).to(compute_dtype).contiguous()
        if grad_hy_in is not None
        else torch.zeros((batch_size, hidden_size), device=device, dtype=compute_dtype)
    )
    dc_cur = (
        grad_cy_in.squeeze(0).to(compute_dtype).contiguous()
        if grad_cy_in is not None
        else torch.zeros((batch_size, hidden_size), device=device, dtype=compute_dtype)
    )
    if grad_output_in is not None:
        go = grad_output_in.transpose(0, 1) if batch_first else grad_output_in
        grad_out_seq = go.to(compute_dtype).contiguous()
    else:
        grad_out_seq = None
    zero_go = torch.empty(0, device=device, dtype=compute_dtype)
    has_grad_out = grad_out_seq is not None
    grad_out_base = grad_out_seq if has_grad_out else zero_go
    dh_bufs = [
        torch.empty((batch_size, hidden_size), device=device, dtype=compute_dtype),
        torch.empty((batch_size, hidden_size), device=device, dtype=compute_dtype),
    ]
    dc_bufs = [
        torch.empty((batch_size, hidden_size), device=device, dtype=compute_dtype),
        torch.empty((batch_size, hidden_size), device=device, dtype=compute_dtype),
    ]
    for rev in range(seq_len):
        t = seq_len - 1 - rev
        dc_out = dc_bufs[rev & 1]
        _lstm_cell_backward_kernel[grid](
            i_cache,
            f_cache,
            g_cache,
            o_cache,
            c_cache,
            dh,
            dc_cur,
            dc_out,
            grad_out_base,
            grad_gates_all,
            has_grad_out,
            t * step,
            step,
            hidden_size,
            block_size,
            num_warps=NUM_WARPS_BWD,
        )
        # dh for the previous step reuses the other ping-pong pair.
        dh = torch.mm(grad_gates_all[t], w_hh_c, out=dh_bufs[rev & 1])
        dc_cur = dc_out

    # Batched tail: weight / bias / input gradients in one pass over the flat
    # (T*B, 4H) views -- measurably cheaper than the einsum+stack chain.
    grad_w_ih_acc = grad_w_hh_acc = grad_bias = None
    if output_mask[3]:
        grad_gates_2d = grad_gates_all.view(flat_rows, gates)
        grad_w_ih_acc = grad_gates_2d.t() @ inp2d
        grad_w_hh_acc = grad_gates_2d.t() @ h_cache[:seq_len].view(
            flat_rows, hidden_size
        )
        grad_bias = grad_gates_2d.sum(dim=0)

    grad_hx_out = dh.unsqueeze(0).to(orig_dtype) if output_mask[1] else None
    grad_cx_out = dc_cur.unsqueeze(0).to(orig_dtype) if output_mask[2] else None
    if output_mask[0]:
        grad_input_seq = (grad_gates_all.view(flat_rows, gates) @ w_ih_c).view(
            seq_len, batch_size, input_size
        )
        grad_input = grad_input_seq.to(orig_dtype)
        if batch_first:
            grad_input = grad_input.transpose(0, 1).contiguous()
    else:
        grad_input = None
    if output_mask[3]:
        grad_weight = [
            grad_w_ih_acc.to(orig_dtype),
            grad_w_hh_acc.to(orig_dtype),
            grad_bias.to(orig_dtype),
            grad_bias.to(orig_dtype),
        ]
    else:
        grad_weight = []
    return grad_input, grad_hx_out, grad_cx_out, grad_weight
