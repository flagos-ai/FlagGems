import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.dropout import dropout as _dropout
from flag_gems.ops.gru import (
    _PACK_BLOCK,
    _batch_offsets_kernel,
    _bias_stride,
    _block_size,
    _ceil_power_of_2,
    _copy_hx_slice,
    _empty,
    _gru_gemv_kernel,
    _gru_persistent_kernel,
    _gru_step_kernel,
    _max_persistent_programs,
    _pack_output_kernel,
    _param_group,
    _store_hx_slice,
    _transpose_weight,
    _unpack_padded_kernel,
    _validate_args,
    _validate_weight,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_BLOCK_B = 16
_BLOCK_K_MAX = 64
# Per-step recurrence launch config
_STEP_BLOCK_H = 64
_STEP_BLOCK_K = 32
_STEP_NUM_WARPS = 4
_STEP_NUM_STAGES = 2
# batch==1 is a GEMV (FMA matvec): hoist the weights and use BLOCK_N=2 for occupancy
# when hidden fits one K tile, otherwise fall back to the split grid.
_GEMV_BLOCK_N_MAX = 16
_GEMV_BLOCK_K_MAX = 128
_GEMV_HOIST_BLOCK_N = 2
_GEMV_HOIST_NUM_WARPS = 1
# Persistent (folded time-loop) launch config: the grid-wide barrier needs a co-resident
# grid, so num_stages=1 and BLOCK_H=32 keep shared memory (and thus the grid) small.
_PERSIST_BLOCK_H = 32
_PERSIST_NUM_STAGES = 1


def _max_persistent_blocks(device, block_h, block_k):
    # Co-resident block capacity: shared memory, not threads or registers, is the binding
    # limit, and overshooting it deadlocks the grid-wide barrier.
    props = torch_device_fn.get_device_properties(device)
    # One h tile (BLOCK_B x K) + three gate-weight tiles (K x BLOCK_H), fp32, per stage.
    shared = 4 * (_BLOCK_B * block_k + 3 * block_k * block_h) * _PERSIST_NUM_STAGES
    blocks_per_sm = max(1, props.shared_memory_per_multiprocessor // shared)
    return props.multi_processor_count * blocks_per_sm


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("gru"),
    key=["input_size", "hidden_size", "batch_size"],
)
@triton.heuristics(
    {
        # tl.dot requires K >= 16, but AABS can shrink BLOCK_K below that for small
        # input_size, so floor the K tile at 16 to keep the dot legal.
        "BLOCK_K_PAD": lambda args: max(args["BLOCK_K"], 16),
    }
)
@triton.jit
def _gru_input_gemm_kernel(
    x_ptr,
    w_ih_ptr,
    b_ih_ptr,
    u_ptr,
    batch_sizes_ptr,
    input_size,
    hidden_size,
    batch_size,
    x_stride_s,
    x_stride_b,
    x_stride_f,
    w_ih_stride_r,
    w_ih_stride_c,
    b_ih_stride,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    PACKED: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_K_PAD: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # u[t,b,n] = sum_k x[t,b,k] * W_ih[n,k] + b_ih[n] for all timesteps in one batched
    # GEMM (n indexes the [r|z|n] gates), replacing per-step recomputation.
    pid_b = tl.program_id(0)
    seq_idx = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_mask = offs_b < batch_size
    n_mask = offs_n < 3 * hidden_size

    if PACKED:
        # Packed input: skip the GEMM for fully-inactive batch tiles (rows b >=
        # batch_sizes[seq]) and zero them; the recurrence freezes those rows anyway.
        bs_t = tl.load(batch_sizes_ptr + seq_idx).to(tl.int32)
        if pid_b * BLOCK_B >= bs_t:
            out_offsets = (
                seq_idx * u_stride_s
                + offs_b[:, None] * u_stride_b
                + offs_n[None, :] * u_stride_f
            )
            tl.store(
                u_ptr + out_offsets,
                tl.zeros((BLOCK_B, BLOCK_N), dtype=COMPUTE_DTYPE),
                mask=b_mask[:, None] & n_mask[None, :],
            )
            return

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=COMPUTE_DTYPE)
    for k_block in range(0, tl.cdiv(input_size, BLOCK_K_PAD)):
        offs_k = k_block * BLOCK_K_PAD + tl.arange(0, BLOCK_K_PAD)
        x = tl.load(
            x_ptr
            + seq_idx * x_stride_s
            + offs_b[:, None] * x_stride_b
            + offs_k[None, :] * x_stride_f,
            mask=(offs_b[:, None] < batch_size) & (offs_k[None, :] < input_size),
            other=0.0,
        )
        w = tl.load(
            w_ih_ptr
            + offs_k[:, None] * w_ih_stride_r
            + offs_n[None, :] * w_ih_stride_c,
            mask=(offs_k[:, None] < input_size) & (offs_n[None, :] < 3 * hidden_size),
            other=0.0,
        )
        acc += tl.dot(x, w, out_dtype=COMPUTE_DTYPE, allow_tf32=False)

    if HAS_BIAS:
        b = tl.load(b_ih_ptr + offs_n * b_ih_stride, mask=n_mask, other=0.0)
        acc += b[None, :]

    out_offsets = (
        seq_idx * u_stride_s
        + offs_b[:, None] * u_stride_b
        + offs_n[None, :] * u_stride_f
    )
    tl.store(u_ptr + out_offsets, acc, mask=b_mask[:, None] & n_mask[None, :])


def _run_direction(
    layer_input,
    hx,
    layer_output,
    final_h,
    params,
    state_idx: int,
    out_feature_offset: int,
    input_size: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    has_biases: bool,
    reverse: bool,
    batch_sizes=None,
):
    w_ih, w_hh, b_ih, b_hh = _param_group(params, state_idx, has_biases)
    _validate_weight(w_ih, 3 * hidden_size, input_size)
    _validate_weight(w_hh, 3 * hidden_size, hidden_size)
    if w_ih.dim() == 1:
        w_ih = w_ih.view(3 * hidden_size, input_size)
    if w_hh.dim() == 1:
        w_hh = w_hh.view(3 * hidden_size, hidden_size)
    # Transpose so the GEMM B (gate) dim is contiguous: a strided B costs ~3x on tl.dot.
    w_ih = _transpose_weight(w_ih, 3 * hidden_size, input_size)
    w_hh = _transpose_weight(w_hh, 3 * hidden_size, hidden_size)
    # Post-transpose: stride_r is the K (reduction) stride, stride_c the contiguous gate dim.
    w_ih_stride_r, w_ih_stride_c = w_ih.stride(0), w_ih.stride(1)
    w_hh_stride_r, w_hh_stride_c = w_hh.stride(0), w_hh.stride(1)
    b_ih_stride = _bias_stride(b_ih, 3 * hidden_size) if has_biases else 1
    b_hh_stride = _bias_stride(b_hh, 3 * hidden_size) if has_biases else 1

    if batch_size == 0:
        return

    block_h_step = _block_size(hidden_size, _STEP_BLOCK_H)
    block_k_step = _block_size(hidden_size, _STEP_BLOCK_K)
    block_k_h = _block_size(hidden_size, _BLOCK_K_MAX)
    grid = (
        triton.cdiv(batch_size, _BLOCK_B),
        triton.cdiv(hidden_size, block_h_step),
    )
    block_h_persist = _block_size(hidden_size, _PERSIST_BLOCK_H)
    grid_persist = (
        triton.cdiv(batch_size, _BLOCK_B),
        triton.cdiv(hidden_size, block_h_persist),
    )
    num_programs_persist = grid_persist[0] * grid_persist[1]

    # Iluvatar has no fp64 support: accumulate in fp32 for every input dtype.
    compute_dtype = tl.float32
    gate_dtype = torch.float32

    max_persistent = _max_persistent_programs(layer_input.device)
    # Co-resident capacity of the persistent kernel (shared-memory limited); the GEMV
    # path below keeps using the raw SM count via max_persistent.
    persist_capacity = _max_persistent_blocks(
        layer_input.device, block_h_persist, block_k_h
    )

    # Precompute input-side pre-activations for all timesteps in one batched GEMM
    # (fp16/bf16 accumulate in fp32); the recurrence below only does the hidden GEMM.
    input_gates = _empty(
        (seq_len, batch_size, 3 * hidden_size), gate_dtype, layer_input.device
    )
    input_gemm_grid = lambda META: (
        triton.cdiv(batch_size, META["BLOCK_B"]),
        seq_len,
        triton.cdiv(3 * hidden_size, META["BLOCK_N"]),
    )

    with torch_device_fn.device(layer_input.device):
        _gru_input_gemm_kernel[input_gemm_grid](
            layer_input,
            w_ih,
            b_ih,
            input_gates,
            batch_sizes if batch_sizes is not None else input_gates,
            input_size,
            hidden_size,
            batch_size,
            layer_input.stride(0),
            layer_input.stride(1),
            layer_input.stride(2),
            w_ih_stride_r,
            w_ih_stride_c,
            b_ih_stride,
            input_gates.stride(0),
            input_gates.stride(1),
            input_gates.stride(2),
            PACKED=batch_sizes is not None,
            HAS_BIAS=has_biases,
            COMPUTE_DTYPE=compute_dtype,
        )

        # The barrier kernels need grid <= max_persistent to be co-resident, so any
        # shape that overshoots it falls through to the barrier-free per-step kernel.
        block_k_g = _block_size(hidden_size, _GEMV_BLOCK_K_MAX)
        gemv_hoist = hidden_size <= block_k_g
        gemv_block_n = (
            _GEMV_HOIST_BLOCK_N
            if gemv_hoist
            else _block_size(hidden_size, _GEMV_BLOCK_N_MAX)
        )
        gemv_num_programs = triton.cdiv(hidden_size, gemv_block_n)
        # Grow BLOCK_N (doubling, stays power-of-2) until the GEMV barrier grid fits the
        # SM count: _GEMV_HOIST_BLOCK_N=2 targets 100+ SM parts and overshoots small ones.
        while gemv_num_programs > max_persistent and gemv_block_n < hidden_size:
            gemv_block_n *= 2
            gemv_num_programs = triton.cdiv(hidden_size, gemv_block_n)

        use_gemv = batch_size == 1 and gemv_num_programs <= max_persistent
        use_persistent = batch_size != 1 and num_programs_persist <= persist_capacity

        if use_gemv:
            # batch==1: spread hidden outputs across BLOCK_N per program (see _gru_gemv_kernel).
            h_buf = _empty((2, batch_size, hidden_size), hx.dtype, hx.device)
            _copy_hx_slice(hx, h_buf[0], state_idx, batch_size, hidden_size)
            barrier = torch.zeros(
                (seq_len,), device=layer_input.device, dtype=torch.int32
            )
            _gru_gemv_kernel[(gemv_num_programs,)](
                input_gates,
                h_buf,
                w_hh,
                b_hh,
                layer_output,
                barrier,
                out_feature_offset,
                hidden_size,
                seq_len,
                input_gates.stride(0),
                input_gates.stride(2),
                w_hh_stride_r,
                w_hh_stride_c,
                b_hh_stride,
                layer_output.stride(0),
                layer_output.stride(2),
                HAS_BIAS=has_biases,
                REVERSE=reverse,
                HOIST=gemv_hoist,
                BLOCK_N=gemv_block_n,
                BLOCK_K=block_k_g,
                NUM_PROGRAMS=gemv_num_programs,
                COMPUTE_DTYPE=compute_dtype,
                num_warps=_GEMV_HOIST_NUM_WARPS if gemv_hoist else 4,
            )
            final_h_state = h_buf[seq_len % 2]
        elif use_persistent:
            # Fold the time loop into one launch (the launch-bound regime); see _gru_persistent_kernel.
            h_buf = _empty((2, batch_size, hidden_size), hx.dtype, hx.device)
            # Copy the initial h into the double buffer (Tensor.copy_ would hit
            # FlagGems' copy_ override, which can't handle this view).
            _copy_hx_slice(hx, h_buf[0], state_idx, batch_size, hidden_size)
            barrier = torch.zeros(
                (seq_len,), device=layer_input.device, dtype=torch.int32
            )
            _gru_persistent_kernel[grid_persist](
                input_gates,
                h_buf,
                w_hh,
                b_hh,
                layer_output,
                barrier,
                batch_sizes if batch_sizes is not None else h_buf,
                out_feature_offset,
                hidden_size,
                batch_size,
                seq_len,
                input_gates.stride(0),
                input_gates.stride(1),
                input_gates.stride(2),
                w_hh_stride_r,
                w_hh_stride_c,
                b_hh_stride,
                layer_output.stride(0),
                layer_output.stride(1),
                layer_output.stride(2),
                HAS_BIAS=has_biases,
                REVERSE=reverse,
                BLOCK_B=_BLOCK_B,
                BLOCK_H=block_h_persist,
                BLOCK_K=block_k_h,
                NUM_PROGRAMS=num_programs_persist,
                COMPUTE_DTYPE=compute_dtype,
                PACKED=batch_sizes is not None,
                num_stages=_PERSIST_NUM_STAGES,
            )
            final_h_state = h_buf[seq_len % 2]
        else:
            h_work = _empty((batch_size, hidden_size), hx.dtype, hx.device)
            _copy_hx_slice(hx, h_work, state_idx, batch_size, hidden_size)
            h_next = _empty((batch_size, hidden_size), hx.dtype, hx.device)
            for step in range(seq_len):
                seq_idx = seq_len - 1 - step if reverse else step
                _gru_step_kernel[grid](
                    input_gates,
                    h_work,
                    w_hh,
                    b_hh,
                    h_next,
                    layer_output,
                    batch_sizes if batch_sizes is not None else h_work,
                    seq_idx,
                    out_feature_offset,
                    hidden_size,
                    batch_size,
                    input_gates.stride(0),
                    input_gates.stride(1),
                    input_gates.stride(2),
                    w_hh_stride_r,
                    w_hh_stride_c,
                    b_hh_stride,
                    layer_output.stride(0),
                    layer_output.stride(1),
                    layer_output.stride(2),
                    HAS_BIAS=has_biases,
                    BLOCK_B=_BLOCK_B,
                    BLOCK_H=block_h_step,
                    BLOCK_K=block_k_step,
                    COMPUTE_DTYPE=compute_dtype,
                    PACKED=batch_sizes is not None,
                    num_warps=_STEP_NUM_WARPS,
                    num_stages=_STEP_NUM_STAGES,
                )
                h_work, h_next = h_next, h_work
            final_h_state = h_work

    _store_hx_slice(final_h_state, final_h, state_idx, batch_size, hidden_size)


def _gru_forward_impl(
    input_view,
    hx,
    params,
    output,
    final_h,
    num_layers,
    num_directions,
    hidden_size,
    input_size,
    batch_size,
    seq_len,
    has_biases,
    train,
    dropout,
    batch_sizes=None,
):
    layer_input = input_view
    for layer in range(num_layers):
        layer_input_size = input_size if layer == 0 else hidden_size * num_directions
        if layer == num_layers - 1:
            layer_output = output
        else:
            layer_output = _empty(
                (seq_len, batch_size, hidden_size * num_directions),
                input_view.dtype,
                input_view.device,
            )
        for direction in range(num_directions):
            state_idx = layer * num_directions + direction
            reverse = direction == 1
            _run_direction(
                layer_input,
                hx,
                layer_output,
                final_h,
                params,
                state_idx,
                direction * hidden_size,
                layer_input_size,
                hidden_size,
                batch_size,
                seq_len,
                has_biases,
                reverse,
                batch_sizes,
            )

        layer_input = layer_output
        if train and dropout != 0.0 and layer + 1 < num_layers:
            layer_input, _ = _dropout(layer_input, dropout, True)

    return layer_input


def gru(
    input,
    hx,
    params,
    has_biases=True,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
    batch_first=False,
):
    logger.debug("GEMS_ILUVATAR GRU")
    _validate_args(input, hx, params, has_biases, num_layers, dropout, bidirectional)

    if batch_first:
        batch_size, seq_len, input_size = input.shape
        input_view = input.transpose(0, 1)
    else:
        seq_len, batch_size, input_size = input.shape
        input_view = input
    if seq_len == 0:
        raise RuntimeError("Expected sequence length to be larger than 0 in RNN")

    hidden_size = hx.shape[2]
    num_directions = 2 if bidirectional else 1

    final_h = _empty(
        (num_layers * num_directions, batch_size, hidden_size),
        input.dtype,
        input.device,
    )
    output_tf = _empty(
        (seq_len, batch_size, hidden_size * num_directions),
        input.dtype,
        input.device,
    )
    _gru_forward_impl(
        input_view,
        hx,
        params,
        output_tf,
        final_h,
        num_layers,
        num_directions,
        hidden_size,
        input_size,
        batch_size,
        seq_len,
        has_biases,
        train,
        dropout,
    )

    output = output_tf.transpose(0, 1) if batch_first else output_tf
    return output, final_h


def gru_data(
    data,
    batch_sizes,
    hx,
    params,
    has_biases=True,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
):
    logger.debug("GEMS_ILUVATAR GRU_DATA")
    if data.dim() != 2:
        raise RuntimeError("gru.data: packed data must have 2 dimensions")
    if batch_sizes.dim() != 1:
        raise RuntimeError("gru.data: batch_sizes must be 1-dimensional")
    if num_layers <= 0:
        raise RuntimeError("gru.data: num_layers must be greater than zero")
    if not 0.0 <= dropout <= 1.0:
        raise RuntimeError("gru.data: dropout probability must be between 0 and 1")
    if data.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems gru.data supports float16, bfloat16, float32, and float64"
        )
    num_directions = 2 if bidirectional else 1
    num_states = num_layers * num_directions
    if hx.dim() != 3:
        raise RuntimeError("gru.data: hidden state must have 3 dimensions")
    if hx.shape[0] != num_states:
        raise RuntimeError(
            f"gru.data: expected {num_states} hidden state rows, got {hx.shape[0]}"
        )
    expected_params = num_states * (4 if has_biases else 2)
    if len(params) != expected_params:
        raise RuntimeError(
            f"gru.data: expected {expected_params} parameter tensors, got {len(params)}"
        )
    if hx.device != data.device:
        raise RuntimeError("gru.data: data and hidden state must share a device")
    if hx.dtype != data.dtype:
        raise RuntimeError("gru.data: data and hidden state must share a dtype")

    num_steps = batch_sizes.numel()
    input_size = data.shape[1]
    batch = hx.shape[1]
    hidden_size = hx.shape[2]

    # pack_padded_sequence produces batch_sizes on CPU; the kernels below need it on
    # the data's device. Move it here (no-op when already resident).
    batch_sizes = batch_sizes.to(data.device)

    # Exclusive prefix-sum of batch_sizes (plus an int32 copy for the recurrence mask)
    # via a kernel, avoiding torch.cumsum/sub dispatch (crashes on packed input).
    offsets = _empty((num_steps,), torch.int32, data.device)
    bs32 = _empty((num_steps,), torch.int32, data.device)
    with torch_device_fn.device(data.device):
        _batch_offsets_kernel[(1,)](
            batch_sizes,
            offsets,
            bs32,
            num_steps,
            BLOCK=_ceil_power_of_2(num_steps),
        )

    # Gather the packed input into a zero-padded (num_steps, batch, input) tensor so
    # the existing batched recurrence can be reused unchanged; padding rows are zeros.
    x_padded = _empty((num_steps, batch, input_size), data.dtype, data.device)
    with torch_device_fn.device(data.device):
        _unpack_padded_kernel[(num_steps * batch,)](
            data,
            x_padded,
            offsets,
            bs32,
            input_size,
            batch,
            data.stride(0),
            x_padded.stride(0),
            x_padded.stride(1),
            x_padded.stride(2),
            BLOCK_F=_PACK_BLOCK,
        )

    hidden_total = hidden_size * num_directions
    final_h = _empty((num_states, batch, hidden_size), data.dtype, data.device)
    out_padded = _empty((num_steps, batch, hidden_total), data.dtype, data.device)
    _gru_forward_impl(
        x_padded,
        hx,
        params,
        out_padded,
        final_h,
        num_layers,
        num_directions,
        hidden_size,
        input_size,
        batch,
        num_steps,
        has_biases,
        train,
        dropout,
        batch_sizes=bs32,
    )

    # Pack the padded output back into the (sum(batch_sizes), hidden) layout; for
    # bidirectional rows carry the concatenated [forward | reverse] hidden states.
    out_packed = _empty((data.shape[0], hidden_total), data.dtype, data.device)
    with torch_device_fn.device(data.device):
        _pack_output_kernel[(num_steps * batch,)](
            out_padded,
            out_packed,
            offsets,
            bs32,
            hidden_total,
            batch,
            out_padded.stride(0),
            out_padded.stride(1),
            out_padded.stride(2),
            out_packed.stride(0),
            BLOCK_F=_PACK_BLOCK,
        )

    return out_packed, final_h
