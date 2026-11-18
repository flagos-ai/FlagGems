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

The common implementation schedules step by step: per step 1 forward Triton kernel, 1 backward
Triton kernel, 6 BLAS calls, 7 tensor allocations. On GCU the fixed cost of these launches / allocs
far exceeds the actual compute for small shapes. This implementation keeps the two Triton kernels as the core
compute path while compressing scheduling on the host:

* preallocate the whole gate cache (FUSED_CACHE), passing a view to the kernel each step to remove per-step alloc;
* precompute the input projection (PRECOMPUTE_INPUT_PROJ), merging T small GEMMs into 1;
* merge per-step weight / bias / input gradient accumulation into one batched BLAS after the loop
  (BATCHED_BLAS), keeping only ``grad_gates @ w_hh`` required by stepwise recurrence.

``BASELINE_MODE=1`` forwards directly to the common implementation for collecting an equivalent baseline; after
optimization the main path is the fused scheduling version above.
"""

import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.utils import tl_extra_shim
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Compile-time config, adjusted per optimization round; semantics unchanged.
BASELINE_MODE = 0
BATCHED_BLAS = 1
PRECOMPUTE_INPUT_PROJ = 1
FUSED_CACHE = 1
BLOCK_SIZE_SCALE = 1
NUM_WARPS_FWD = 4
NUM_WARPS_BWD = 8
USE_ADDMM = 1
CLONE_DC = 1
# --- TUNING PARAMS END ---


@libentry()
@triton.jit
def _lstm_cell_forward_kernel(
    pre_gates_ptr,
    c_prev_ptr,
    i_ptr,
    f_ptr,
    g_ptr,
    o_ptr,
    c_ptr,
    h_ptr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-step forward: convert the four pre-activation gates into activations and write c/h; one program per
    row."""
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
    c_prev = tl.load(
        c_prev_ptr + batch_idx * hidden_size + offsets, mask=mask, other=0.0
    ).to(tl.float32)

    i = tl.sigmoid(i_pre)
    f = tl.sigmoid(f_pre)
    g = tl_extra_shim.tanh(g_pre)
    o = tl.sigmoid(o_pre)
    c = f * c_prev + i * g
    h = o * tl_extra_shim.tanh(c)

    row = batch_idx * hidden_size + offsets
    tl.store(i_ptr + row, i, mask=mask)
    tl.store(f_ptr + row, f, mask=mask)
    tl.store(g_ptr + row, g, mask=mask)
    tl.store(o_ptr + row, o, mask=mask)
    tl.store(c_ptr + row, c, mask=mask)
    tl.store(h_ptr + row, h, mask=mask)


@libentry()
@triton.jit
def _lstm_cell_backward_kernel(
    i_ptr,
    f_ptr,
    g_ptr,
    o_ptr,
    c_ptr,
    c_prev_ptr,
    dh_ptr,
    dc_ptr,
    grad_out_ptr,
    grad_gates_ptr,
    has_grad_out: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-step backward: compute the four gate gradients by the chain rule and write back dc for the previous
    time step."""
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    row = batch_idx * hidden_size + offsets

    i = tl.load(i_ptr + row, mask=mask, other=0.0).to(tl.float32)
    f = tl.load(f_ptr + row, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(o_ptr + row, mask=mask, other=0.0).to(tl.float32)
    c_t = tl.load(c_ptr + row, mask=mask, other=0.0).to(tl.float32)
    c_prev = tl.load(c_prev_ptr + row, mask=mask, other=0.0).to(tl.float32)
    dh = tl.load(dh_ptr + row, mask=mask, other=0.0).to(tl.float32)
    dc = tl.load(dc_ptr + row, mask=mask, other=0.0).to(tl.float32)
    if has_grad_out:
        dh = dh + tl.load(grad_out_ptr + row, mask=mask, other=0.0).to(tl.float32)

    tanh_c = tl_extra_shim.tanh(c_t)
    do_pre = dh * tanh_c * o * (1.0 - o)
    dc = dc + dh * o * (1.0 - tanh_c * tanh_c)
    di_pre = dc * g * i * (1.0 - i)
    df_pre = dc * c_prev * f * (1.0 - f)
    dg_pre = dc * i * (1.0 - g * g)
    dc_prev = dc * f

    gate_base = grad_gates_ptr + batch_idx * 4 * hidden_size
    tl.store(gate_base + 0 * hidden_size + offsets, di_pre, mask=mask)
    tl.store(gate_base + 1 * hidden_size + offsets, df_pre, mask=mask)
    tl.store(gate_base + 2 * hidden_size + offsets, dg_pre, mask=mask)
    tl.store(gate_base + 3 * hidden_size + offsets, do_pre, mask=mask)
    tl.store(dc_ptr + row, dc_prev, mask=mask)


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
    logger.debug("GEMS_ENFLAME_GCU400 CUDNN_RNN_BACKWARD")
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
    seq_len, batch_size, _ = inp.shape
    orig_dtype = input.dtype
    compute_dtype = torch.float32
    inp_c = inp.to(compute_dtype)
    w_ih_c = w_ih.to(compute_dtype)
    w_hh_c = w_hh.to(compute_dtype)
    b_ih_c = b_ih.to(compute_dtype)
    b_hh_c = b_hh.to(compute_dtype)
    h = hx.squeeze(0).to(compute_dtype).contiguous()
    c = cx.squeeze(0).to(compute_dtype).contiguous()
    device = inp_c.device

    block_size = _lstm_block_size(hidden_size)
    grid = (batch_size,)
    # Allocate the whole cache once; each step only takes a view, no tensor is created inside the loop.
    # Batched BLAS depends on the whole grad_gates cache, so both modes share the same cache.
    fused_cache = FUSED_CACHE or BATCHED_BLAS
    if fused_cache:
        caches = [
            torch.empty(
                (seq_len, batch_size, hidden_size), device=device, dtype=compute_dtype
            )
            for _ in range(6)
        ]
        i_all, f_all, g_all, o_all, c_all, h_all = caches
        grad_gates_all = torch.empty(
            (seq_len, batch_size, 4 * hidden_size), device=device, dtype=compute_dtype
        )

    # The input projection is time-independent, so T small GEMMs are merged into one 3D matmul.
    input_proj = inp_c @ w_ih_c.t() + b_ih_c if PRECOMPUTE_INPUT_PROJ else None

    h_prev_list = []
    c_prev_list = []
    i_list, f_list, g_list, o_list, c_new_list = [], [], [], [], []
    for t in range(seq_len):
        if USE_ADDMM:
            pre_gates = flag_gems.addmm(b_hh_c, h, w_hh_c.t())
            if PRECOMPUTE_INPUT_PROJ:
                pre_gates.add_(input_proj[t])
            else:
                pre_gates.add_(inp_c[t] @ w_ih_c.t() + b_ih_c)
        else:
            pre_gates = (
                input_proj[t] + h @ w_hh_c.t() + b_hh_c
                if PRECOMPUTE_INPUT_PROJ
                else inp_c[t] @ w_ih_c.t() + b_ih_c + h @ w_hh_c.t() + b_hh_c
            ).contiguous()

        if fused_cache:
            i_t, f_t, g_t, o_t = (
                i_all.select(0, t),
                f_all.select(0, t),
                g_all.select(0, t),
                o_all.select(0, t),
            )
            c_t, h_t = c_all.select(0, t), h_all.select(0, t)
        else:
            i_t, f_t, g_t, o_t, c_t, h_t = [
                torch.empty(
                    (batch_size, hidden_size), device=device, dtype=compute_dtype
                )
                for _ in range(6)
            ]
        _lstm_cell_forward_kernel[grid](
            pre_gates,
            c,
            i_t,
            f_t,
            g_t,
            o_t,
            c_t,
            h_t,
            hidden_size,
            block_size,
            num_warps=NUM_WARPS_FWD,
        )
        i_list.append(i_t)
        f_list.append(f_t)
        g_list.append(g_t)
        o_list.append(o_t)
        c_new_list.append(c_t)
        h_prev_list.append(h)
        c_prev_list.append(c)
        h, c = h_t, c_t

    grad_w_ih_acc = torch.zeros_like(w_ih_c)
    grad_w_hh_acc = torch.zeros_like(w_hh_c)
    grad_b_ih_acc = torch.zeros_like(b_ih_c)
    grad_b_hh_acc = torch.zeros_like(b_hh_c)

    dh = (
        grad_hy_in.squeeze(0).to(compute_dtype).contiguous()
        if grad_hy_in is not None
        else torch.zeros((batch_size, hidden_size), device=device, dtype=compute_dtype)
    )
    # dc is written back in place by the backward kernel; clone avoids polluting the caller's grad_cy input.
    dc = (
        grad_cy_in.squeeze(0).to(compute_dtype).contiguous()
        if grad_cy_in is not None
        else torch.zeros((batch_size, hidden_size), device=device, dtype=compute_dtype)
    )
    if CLONE_DC:
        dc = dc.clone()
    grad_out_seq = None
    if grad_output_in is not None:
        go = grad_output_in.transpose(0, 1) if batch_first else grad_output_in
        grad_out_seq = go.to(compute_dtype).contiguous()
    zero_go = torch.empty(0, device=device, dtype=compute_dtype)
    has_grad_out = grad_out_seq is not None
    # Only when neither batched BLAS nor the whole-segment cache is used must grad_input be written stepwise in the
    # loop.
    grad_input_seq = (
        torch.zeros_like(inp_c)
        if output_mask[0] and not (BATCHED_BLAS or fused_cache)
        else None
    )

    for t in range(seq_len - 1, -1, -1):
        if fused_cache:
            grad_gates = grad_gates_all.select(0, t)
        else:
            grad_gates = torch.empty(
                (batch_size, 4 * hidden_size), device=device, dtype=compute_dtype
            )
        _lstm_cell_backward_kernel[grid](
            i_list[t],
            f_list[t],
            g_list[t],
            o_list[t],
            c_new_list[t],
            c_prev_list[t],
            dh,
            dc,
            grad_out_seq[t] if has_grad_out else zero_go,
            grad_gates,
            has_grad_out,
            hidden_size,
            block_size,
            num_warps=NUM_WARPS_BWD,
        )
        if not BATCHED_BLAS:
            grad_w_ih_acc += grad_gates.t() @ inp_c[t]
            grad_w_hh_acc += grad_gates.t() @ h_prev_list[t]
            grad_b_ih_acc += grad_gates.sum(dim=0)
            grad_b_hh_acc += grad_gates.sum(dim=0)
        if grad_input_seq is not None:
            grad_input_seq[t] = grad_gates @ w_ih_c
        dh = grad_gates @ w_hh_c

    if BATCHED_BLAS:
        # Weight / bias / input gradients of all time steps are computed in one batched pass.
        h_prev_stack = flag_gems.stack(h_prev_list, dim=0)
        grad_w_ih_acc = torch.einsum("tbi,tbj->ij", grad_gates_all, inp_c)
        grad_w_hh_acc = torch.einsum("tbi,tbj->ij", grad_gates_all, h_prev_stack)
        grad_bias = grad_gates_all.sum(dim=(0, 1))
        grad_b_ih_acc = grad_bias
        grad_b_hh_acc = grad_bias
        grad_input_seq = grad_gates_all @ w_ih_c
    elif fused_cache and output_mask[0]:
        grad_input_seq = grad_gates_all @ w_ih_c

    grad_hx_out = dh.unsqueeze(0).to(orig_dtype) if output_mask[1] else None
    grad_cx_out = dc.unsqueeze(0).to(orig_dtype) if output_mask[2] else None
    if output_mask[0]:
        grad_input = grad_input_seq.to(orig_dtype)
        if batch_first:
            grad_input = grad_input.transpose(0, 1).contiguous()
    else:
        grad_input = None
    if output_mask[3]:
        grad_weight = [
            grad_w_ih_acc.to(orig_dtype),
            grad_w_hh_acc.to(orig_dtype),
            grad_b_ih_acc.to(orig_dtype),
            grad_b_hh_acc.to(orig_dtype),
        ]
    else:
        grad_weight = []
    return grad_input, grad_hx_out, grad_cx_out, grad_weight
