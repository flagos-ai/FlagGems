import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.dropout import dropout as _dropout
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

_BLOCK_B = 16
_BLOCK_H_MAX = 32
_BLOCK_K_MAX = 64
_COPY_BLOCK = 1024
# One-shot weight transposition tile (weights are small).
_TRANSPOSE_BLOCK = 32
# Element width for the packed<->padded scatter/gather kernels.
_PACK_BLOCK = 256
# batch==1 is a GEMV (FMA matvec): hoist the weights and use BLOCK_N=2 for occupancy
# when hidden fits one K tile, otherwise fall back to the split grid.
_GEMV_BLOCK_N_MAX = 16
_GEMV_BLOCK_K_MAX = 128
_GEMV_HOIST_BLOCK_N = 2
_GEMV_HOIST_NUM_WARPS = 1

# Per-step recurrence launch config
_STEP_BLOCK_H = 64
_STEP_BLOCK_K = 32
_STEP_NUM_WARPS = 4
_STEP_NUM_STAGES = 2


@libentry()
@triton.jit
def _copy_hx_slice_kernel(
    src,
    dst,
    state_idx,
    batch_size,
    hidden_size,
    src_stride_0,
    src_stride_1,
    src_stride_2,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < batch_size * hidden_size
    batch_offsets = offsets // hidden_size
    hidden_offsets = offsets - batch_offsets * hidden_size
    src_offsets = (
        state_idx * src_stride_0
        + batch_offsets * src_stride_1
        + hidden_offsets * src_stride_2
    )
    values = tl.load(src + src_offsets, mask=mask, other=0.0)
    tl.store(dst + offsets, values, mask=mask)


@libentry()
@triton.jit
def _store_hx_slice_kernel(
    src,
    dst,
    state_idx,
    batch_size,
    hidden_size,
    dst_stride_0,
    dst_stride_1,
    dst_stride_2,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < batch_size * hidden_size
    batch_offsets = offsets // hidden_size
    hidden_offsets = offsets - batch_offsets * hidden_size
    dst_offsets = (
        state_idx * dst_stride_0
        + batch_offsets * dst_stride_1
        + hidden_offsets * dst_stride_2
    )
    values = tl.load(src + offsets, mask=mask, other=0.0)
    tl.store(dst + dst_offsets, values, mask=mask)


@libentry()
@triton.jit
def _transpose_weight_kernel(
    src,
    dst,
    rows,
    cols,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Transpose to (cols, rows) so the GEMM B operand's gate dim is contiguous;
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_r[:, None] < rows) & (offs_c[None, :] < cols)
    vals = tl.load(src + offs_r[:, None] * cols + offs_c[None, :], mask=mask, other=0.0)
    tl.store(dst + offs_c[None, :] * rows + offs_r[:, None], vals, mask=mask)


@libentry()
@triton.jit
def _batch_offsets_kernel(
    batch_sizes_ptr,
    offsets_ptr,
    bs32_ptr,
    num_steps,
    BLOCK: tl.constexpr,
):
    # Exclusive prefix-sum of batch_sizes (plus an int32 copy for the recurrence mask),
    # in one program via tl.cumsum. Avoids torch.cumsum/sub dispatch under use_gems().
    offs = tl.arange(0, BLOCK)
    mask = offs < num_steps
    vals = tl.load(batch_sizes_ptr + offs, mask=mask, other=0).to(tl.int32)
    exclusive = tl.cumsum(vals, axis=0) - vals
    tl.store(offsets_ptr + offs, exclusive, mask=mask)
    tl.store(bs32_ptr + offs, vals, mask=mask)


@libentry()
@triton.jit
def _unpack_padded_kernel(
    data_ptr,
    x_ptr,
    offsets_ptr,
    batch_sizes_ptr,
    input_size,
    batch_size,
    data_stride_0,
    x_stride_s,
    x_stride_b,
    x_stride_f,
    BLOCK_F: tl.constexpr,
):
    # Scatter packed (sum(batch_sizes), input) data into zero-padded (num_steps, batch,
    # input). One program per (timestep, batch-row); padding rows are left zero.
    pid = tl.program_id(0)
    t = pid // batch_size
    b = pid - t * batch_size
    bs_t = tl.load(batch_sizes_ptr + t).to(tl.int32)
    active = b < bs_t
    row = tl.load(offsets_ptr + t).to(tl.int32) + b
    for f_block in range(0, tl.cdiv(input_size, BLOCK_F)):
        offs_f = f_block * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = offs_f < input_size
        vals = tl.load(
            data_ptr + row * data_stride_0 + offs_f,
            mask=active & f_mask,
            other=0.0,
        )
        tl.store(
            x_ptr + t * x_stride_s + b * x_stride_b + offs_f * x_stride_f,
            vals,
            mask=f_mask,
        )


@libentry()
@triton.jit
def _pack_output_kernel(
    out_padded_ptr,
    out_packed_ptr,
    offsets_ptr,
    batch_sizes_ptr,
    hidden_size,
    batch_size,
    out_stride_s,
    out_stride_b,
    out_stride_f,
    packed_stride_0,
    BLOCK_F: tl.constexpr,
):
    # Gather padded (num_steps, batch, hidden) output back into packed
    # (sum(batch_sizes), hidden). Only active rows are copied.
    pid = tl.program_id(0)
    t = pid // batch_size
    b = pid - t * batch_size
    bs_t = tl.load(batch_sizes_ptr + t).to(tl.int32)
    active = b < bs_t
    row = tl.load(offsets_ptr + t).to(tl.int32) + b
    for f_block in range(0, tl.cdiv(hidden_size, BLOCK_F)):
        offs_f = f_block * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = offs_f < hidden_size
        vals = tl.load(
            out_padded_ptr
            + t * out_stride_s
            + b * out_stride_b
            + offs_f * out_stride_f,
            mask=f_mask,
            other=0.0,
        )
        tl.store(
            out_packed_ptr + row * packed_stride_0 + offs_f,
            vals,
            mask=active & f_mask,
        )


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("gru"),
    key=["input_size", "hidden_size", "batch_size"],
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
    for k_block in range(0, tl.cdiv(input_size, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
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


@libentry()
@triton.jit
def _gru_step_kernel(
    u_ptr,
    h_prev_ptr,
    w_hh_ptr,
    b_hh_ptr,
    h_next_ptr,
    out_ptr,
    batch_sizes_ptr,
    seq_idx,
    out_feature_offset,
    hidden_size,
    batch_size,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    w_hh_stride_r,
    w_hh_stride_c,
    b_hh_stride,
    out_stride_s,
    out_stride_b,
    out_stride_f,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    PACKED: tl.constexpr,
):
    # One recurrence step: ``u`` already holds the input pre-activations, so only the
    # hidden GEMM + gates remain. Gates r/z sigmoid, n = tanh(u_n + r * (W_hn h + b_hn)).
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    bh_mask = (offs_b[:, None] < batch_size) & (offs_h[None, :] < hidden_size)

    if PACKED:
        # Packed input: carry h_prev forward for rows b >= batch_sizes[seq_idx] (skip
        # the hidden GEMM); the pack kernel drops these rows from the output anyway.
        bs_t = tl.load(batch_sizes_ptr + seq_idx).to(tl.int32)
        if pid_b * BLOCK_B >= bs_t:
            state_offsets = offs_b[:, None] * hidden_size + offs_h[None, :]
            out_offsets = (
                seq_idx * out_stride_s
                + offs_b[:, None] * out_stride_b
                + (out_feature_offset + offs_h[None, :]) * out_stride_f
            )
            h_prev_tile = tl.load(h_prev_ptr + state_offsets, mask=bh_mask, other=0.0)
            tl.store(h_next_ptr + state_offsets, h_prev_tile, mask=bh_mask)
            tl.store(out_ptr + out_offsets, h_prev_tile, mask=bh_mask)
            return

    u_base = (
        seq_idx * u_stride_s
        + offs_b[:, None] * u_stride_b
        + offs_h[None, :] * u_stride_f
    )
    r_acc = tl.load(u_ptr + u_base, mask=bh_mask, other=0.0).to(COMPUTE_DTYPE)
    z_acc = tl.load(
        u_ptr + u_base + hidden_size * u_stride_f, mask=bh_mask, other=0.0
    ).to(COMPUTE_DTYPE)
    n_in = tl.load(
        u_ptr + u_base + 2 * hidden_size * u_stride_f, mask=bh_mask, other=0.0
    ).to(COMPUTE_DTYPE)
    n_h_acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=COMPUTE_DTYPE)

    for k_block in range(0, tl.cdiv(hidden_size, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        h = tl.load(
            h_prev_ptr + offs_b[:, None] * hidden_size + offs_k[None, :],
            mask=(offs_b[:, None] < batch_size) & (offs_k[None, :] < hidden_size),
            other=0.0,
        )

        w_r = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + offs_h[None, :] * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        w_z = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + (hidden_size + offs_h[None, :]) * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        w_n = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + (2 * hidden_size + offs_h[None, :]) * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        r_acc += tl.dot(h, w_r, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
        z_acc += tl.dot(h, w_z, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
        n_h_acc += tl.dot(h, w_n, out_dtype=COMPUTE_DTYPE, allow_tf32=False)

    if HAS_BIAS:
        b_hr = tl.load(
            b_hh_ptr + offs_h * b_hh_stride, mask=offs_h < hidden_size, other=0.0
        )
        b_hz = tl.load(
            b_hh_ptr + (hidden_size + offs_h) * b_hh_stride,
            mask=offs_h < hidden_size,
            other=0.0,
        )
        b_hn = tl.load(
            b_hh_ptr + (2 * hidden_size + offs_h) * b_hh_stride,
            mask=offs_h < hidden_size,
            other=0.0,
        )
        r_acc += b_hr[None, :]
        z_acc += b_hz[None, :]
        n_h_acc += b_hn[None, :]

    # libdevice sigmoid/tanh tracks torch CUDA (expf/tanhf) more closely than lstm's
    # exp2-based fast math.
    r_gate = tl.sigmoid(r_acc)
    z_gate = tl.sigmoid(z_acc)
    n_gate = tl_extra_shim.tanh(n_in + r_gate * n_h_acc)

    h_prev = tl.load(
        h_prev_ptr + offs_b[:, None] * hidden_size + offs_h[None, :],
        mask=bh_mask,
        other=0.0,
    ).to(COMPUTE_DTYPE)
    h_next = (1.0 - z_gate) * n_gate + z_gate * h_prev

    if PACKED:
        # For packed input, rows whose sequence has already ended (b >= batch_sizes[t])
        # must not advance: freeze them at their previous hidden value.
        active = offs_b < tl.load(batch_sizes_ptr + seq_idx).to(tl.int32)
        h_next = tl.where(active[:, None], h_next, h_prev)

    state_offsets = offs_b[:, None] * hidden_size + offs_h[None, :]
    out_offsets = (
        seq_idx * out_stride_s
        + offs_b[:, None] * out_stride_b
        + (out_feature_offset + offs_h[None, :]) * out_stride_f
    )
    tl.store(h_next_ptr + state_offsets, h_next, mask=bh_mask)
    tl.store(out_ptr + out_offsets, h_next, mask=bh_mask)


@libentry()
@triton.jit
def _gru_persistent_kernel(
    u_ptr,
    h_ptr,
    w_hh_ptr,
    b_hh_ptr,
    out_ptr,
    barrier_ptr,
    batch_sizes_ptr,
    out_feature_offset,
    hidden_size,
    batch_size,
    seq_len,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    w_hh_stride_r,
    w_hh_stride_c,
    b_hh_stride,
    out_stride_s,
    out_stride_b,
    out_stride_f,
    HAS_BIAS: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    PACKED: tl.constexpr,
):
    # Time loop folded into one launch. h is double-buffered so step reads never race
    # writes; a grid-wide barrier after each step requires a co-resident grid.
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    bh_mask = (offs_b[:, None] < batch_size) & (offs_h[None, :] < hidden_size)
    state_offsets = offs_b[:, None] * hidden_size + offs_h[None, :]
    buf_stride = batch_size * hidden_size

    h_cur = tl.load(h_ptr + state_offsets, mask=bh_mask, other=0.0).to(COMPUTE_DTYPE)

    for step in range(seq_len):
        t = seq_len - 1 - step if REVERSE else step
        read_base = (step % 2) * buf_stride
        write_base = ((step + 1) % 2) * buf_stride

        u_base = (
            t * u_stride_s + offs_b[:, None] * u_stride_b + offs_h[None, :] * u_stride_f
        )
        r_acc = tl.load(u_ptr + u_base, mask=bh_mask, other=0.0).to(COMPUTE_DTYPE)
        z_acc = tl.load(
            u_ptr + u_base + hidden_size * u_stride_f, mask=bh_mask, other=0.0
        ).to(COMPUTE_DTYPE)
        n_in = tl.load(
            u_ptr + u_base + 2 * hidden_size * u_stride_f, mask=bh_mask, other=0.0
        ).to(COMPUTE_DTYPE)
        n_h_acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=COMPUTE_DTYPE)

        for k_block in range(0, tl.cdiv(hidden_size, BLOCK_K)):
            offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            h = tl.load(
                h_ptr + read_base + offs_b[:, None] * hidden_size + offs_k[None, :],
                mask=(offs_b[:, None] < batch_size) & (offs_k[None, :] < hidden_size),
                other=0.0,
            )
            w_r = tl.load(
                w_hh_ptr
                + offs_k[:, None] * w_hh_stride_r
                + offs_h[None, :] * w_hh_stride_c,
                mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
                other=0.0,
            )
            w_z = tl.load(
                w_hh_ptr
                + offs_k[:, None] * w_hh_stride_r
                + (hidden_size + offs_h[None, :]) * w_hh_stride_c,
                mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
                other=0.0,
            )
            w_n = tl.load(
                w_hh_ptr
                + offs_k[:, None] * w_hh_stride_r
                + (2 * hidden_size + offs_h[None, :]) * w_hh_stride_c,
                mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
                other=0.0,
            )
            r_acc += tl.dot(h, w_r, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
            z_acc += tl.dot(h, w_z, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
            n_h_acc += tl.dot(h, w_n, out_dtype=COMPUTE_DTYPE, allow_tf32=False)

        if HAS_BIAS:
            b_hr = tl.load(
                b_hh_ptr + offs_h * b_hh_stride, mask=offs_h < hidden_size, other=0.0
            )
            b_hz = tl.load(
                b_hh_ptr + (hidden_size + offs_h) * b_hh_stride,
                mask=offs_h < hidden_size,
                other=0.0,
            )
            b_hn = tl.load(
                b_hh_ptr + (2 * hidden_size + offs_h) * b_hh_stride,
                mask=offs_h < hidden_size,
                other=0.0,
            )
            r_acc += b_hr[None, :]
            z_acc += b_hz[None, :]
            n_h_acc += b_hn[None, :]

        r_gate = tl.sigmoid(r_acc)
        z_gate = tl.sigmoid(z_acc)
        n_gate = tl_extra_shim.tanh(n_in + r_gate * n_h_acc)
        h_next = (1.0 - z_gate) * n_gate + z_gate * h_cur

        if PACKED:
            active = offs_b < tl.load(batch_sizes_ptr + t).to(tl.int32)
            h_next = tl.where(active[:, None], h_next, h_cur)

        tl.store(h_ptr + write_base + state_offsets, h_next, mask=bh_mask)
        out_offsets = (
            t * out_stride_s
            + offs_b[:, None] * out_stride_b
            + (out_feature_offset + offs_h[None, :]) * out_stride_f
        )
        tl.store(out_ptr + out_offsets, h_next, mask=bh_mask)

        h_cur = h_next

        # Grid-wide barrier: release this program's h write, then acquire everyone
        # else's before the next step reads h.
        tl.atomic_add(barrier_ptr + step, 1, sem="acq_rel")
        while tl.atomic_add(barrier_ptr + step, 0, sem="acquire") < NUM_PROGRAMS:
            pass


@libentry()
@triton.jit
def _gru_gemv_kernel(
    u_ptr,
    h_ptr,
    w_hh_ptr,
    b_hh_ptr,
    out_ptr,
    barrier_ptr,
    out_feature_offset,
    hidden_size,
    seq_len,
    u_stride_s,
    u_stride_f,
    w_hh_stride_r,
    w_hh_stride_c,
    b_hh_stride,
    out_stride_s,
    out_stride_f,
    HAS_BIAS: tl.constexpr,
    REVERSE: tl.constexpr,
    HOIST: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # batch==1: a (batch, hidden) dot grid would idle the GPU (~6x slower than cuDNN);
    # each program owns BLOCK_N outputs, reducing K element-wise so r gates n directly.
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < hidden_size
    buf_stride = hidden_size

    # When hidden <= BLOCK_K the weights fit a single K tile: load them (and, below,
    # the bias) once before the time loop instead of re-reading from L2 every step.
    if HOIST:
        offs_k = tl.arange(0, BLOCK_K)
        mask_k = offs_k < hidden_size
        w_r = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + offs_n[None, :] * w_hh_stride_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(COMPUTE_DTYPE)
        w_z = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + (hidden_size + offs_n)[None, :] * w_hh_stride_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(COMPUTE_DTYPE)
        w_n = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + (2 * hidden_size + offs_n)[None, :] * w_hh_stride_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(COMPUTE_DTYPE)

    if HAS_BIAS:
        b_r = tl.load(b_hh_ptr + offs_n * b_hh_stride, mask=mask_n, other=0.0).to(
            COMPUTE_DTYPE
        )
        b_z = tl.load(
            b_hh_ptr + (hidden_size + offs_n) * b_hh_stride,
            mask=mask_n,
            other=0.0,
        ).to(COMPUTE_DTYPE)
        b_n = tl.load(
            b_hh_ptr + (2 * hidden_size + offs_n) * b_hh_stride,
            mask=mask_n,
            other=0.0,
        ).to(COMPUTE_DTYPE)

    h_cur = tl.load(h_ptr + offs_n, mask=mask_n, other=0.0).to(COMPUTE_DTYPE)

    for step in range(seq_len):
        t = seq_len - 1 - step if REVERSE else step
        read_base = (step % 2) * buf_stride
        write_base = ((step + 1) % 2) * buf_stride

        r_acc = tl.zeros((BLOCK_N,), dtype=COMPUTE_DTYPE)
        z_acc = tl.zeros((BLOCK_N,), dtype=COMPUTE_DTYPE)
        n_h_acc = tl.zeros((BLOCK_N,), dtype=COMPUTE_DTYPE)
        for k_block in range(0, tl.cdiv(hidden_size, BLOCK_K)):
            offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < hidden_size
            h = tl.load(h_ptr + read_base + offs_k, mask=mask_k, other=0.0).to(
                COMPUTE_DTYPE
            )
            if HOIST:
                r_acc += tl.sum(h[:, None] * w_r, axis=0)
                z_acc += tl.sum(h[:, None] * w_z, axis=0)
                n_h_acc += tl.sum(h[:, None] * w_n, axis=0)
            else:
                w_r = tl.load(
                    w_hh_ptr
                    + offs_k[:, None] * w_hh_stride_r
                    + offs_n[None, :] * w_hh_stride_c,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                ).to(COMPUTE_DTYPE)
                w_z = tl.load(
                    w_hh_ptr
                    + offs_k[:, None] * w_hh_stride_r
                    + (hidden_size + offs_n)[None, :] * w_hh_stride_c,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                ).to(COMPUTE_DTYPE)
                w_n = tl.load(
                    w_hh_ptr
                    + offs_k[:, None] * w_hh_stride_r
                    + (2 * hidden_size + offs_n)[None, :] * w_hh_stride_c,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                ).to(COMPUTE_DTYPE)
                r_acc += tl.sum(h[:, None] * w_r, axis=0)
                z_acc += tl.sum(h[:, None] * w_z, axis=0)
                n_h_acc += tl.sum(h[:, None] * w_n, axis=0)

        r_in = tl.load(
            u_ptr + t * u_stride_s + offs_n * u_stride_f, mask=mask_n, other=0.0
        ).to(COMPUTE_DTYPE)
        z_in = tl.load(
            u_ptr + t * u_stride_s + (hidden_size + offs_n) * u_stride_f,
            mask=mask_n,
            other=0.0,
        ).to(COMPUTE_DTYPE)
        n_in = tl.load(
            u_ptr + t * u_stride_s + (2 * hidden_size + offs_n) * u_stride_f,
            mask=mask_n,
            other=0.0,
        ).to(COMPUTE_DTYPE)
        r_acc += r_in
        z_acc += z_in

        if HAS_BIAS:
            r_acc += b_r
            z_acc += b_z
            n_h_acc += b_n

        r_gate = tl.sigmoid(r_acc)
        z_gate = tl.sigmoid(z_acc)
        n_gate = tl_extra_shim.tanh(n_in + r_gate * n_h_acc)
        h_next = (1.0 - z_gate) * n_gate + z_gate * h_cur

        tl.store(h_ptr + write_base + offs_n, h_next, mask=mask_n)
        out_offsets = t * out_stride_s + (out_feature_offset + offs_n) * out_stride_f
        tl.store(out_ptr + out_offsets, h_next, mask=mask_n)

        h_cur = h_next

        tl.atomic_add(barrier_ptr + step, 1, sem="acq_rel")
        while tl.atomic_add(barrier_ptr + step, 0, sem="acquire") < NUM_PROGRAMS:
            pass


def _ceil_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _block_size(value: int, maximum: int) -> int:
    return min(max(_ceil_power_of_2(value), 16), maximum)


def _max_persistent_programs(device) -> int:

    return torch_device_fn.get_device_properties(device).multi_processor_count


def _validate_weight(tensor: torch.Tensor, rows: int, cols: int):
    if tensor.numel() != rows * cols:
        raise RuntimeError(
            f"invalid GRU weight with {tensor.numel()} elements, expected {rows * cols}"
        )
    if tensor.dim() == 2:
        if tensor.shape != (rows, cols):
            raise RuntimeError(
                f"invalid GRU weight shape {tuple(tensor.shape)}, expected {(rows, cols)}"
            )
    elif tensor.dim() != 1:
        raise RuntimeError(f"GRU weights must be 1-D or 2-D, got {tensor.dim()}-D")


def _bias_stride(tensor: torch.Tensor, size: int) -> int:
    if tensor.numel() != size:
        raise RuntimeError(
            f"invalid GRU bias with {tensor.numel()} elements, expected {size}"
        )
    if tensor.dim() == 1:
        return tensor.stride(0)
    raise RuntimeError(f"GRU bias tensors must be 1-D, got {tensor.dim()}-D")


def _param_group(params, index: int, has_biases: bool):
    # Per-direction layout is [w_ih, w_hh, b_ih, b_hh] / [w_ih, w_hh]: 4/2 params per
    # state (same as LSTM) despite GRU's 3 gates.
    group_size = 4 if has_biases else 2
    base = index * group_size
    if has_biases:
        return params[base], params[base + 1], params[base + 2], params[base + 3]
    return params[base], params[base + 1], params[base], params[base]


def _copy_hx_slice(src, dst, state_idx: int, batch_size: int, hidden_size: int):
    if dst.numel() == 0:
        return
    grid = (triton.cdiv(dst.numel(), _COPY_BLOCK),)
    with torch_device_fn.device(dst.device):
        _copy_hx_slice_kernel[grid](
            src,
            dst,
            state_idx,
            batch_size,
            hidden_size,
            src.stride(0),
            src.stride(1),
            src.stride(2),
            BLOCK=_COPY_BLOCK,
        )


def _store_hx_slice(src, dst, state_idx: int, batch_size: int, hidden_size: int):
    if src.numel() == 0:
        return
    grid = (triton.cdiv(src.numel(), _COPY_BLOCK),)
    with torch_device_fn.device(src.device):
        _store_hx_slice_kernel[grid](
            src,
            dst,
            state_idx,
            batch_size,
            hidden_size,
            dst.stride(0),
            dst.stride(1),
            dst.stride(2),
            BLOCK=_COPY_BLOCK,
        )


def _empty(shape, dtype, device):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return torch.empty_strided(shape, strides, dtype=dtype, device=device)


# Transposed (K, 3H) weight cache (saves two transpose launches per direction). Values
# keep a strong ref to the source (data_ptr isn't unique across freed storage).
_transposed_weight_cache: dict = {}


def _transpose_weight(weight: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    key = (
        weight.data_ptr(),
        weight.storage_offset(),
        weight.numel(),
        weight.dtype,
        getattr(weight, "_version", 0),
    )
    cached = _transposed_weight_cache.get(key)
    if cached is not None:
        return cached[1]

    transposed = torch.empty((cols, rows), dtype=weight.dtype, device=weight.device)
    grid = (
        triton.cdiv(rows, _TRANSPOSE_BLOCK),
        triton.cdiv(cols, _TRANSPOSE_BLOCK),
    )
    with torch_device_fn.device(weight.device):
        _transpose_weight_kernel[grid](
            weight,
            transposed,
            rows,
            cols,
            BLOCK_R=_TRANSPOSE_BLOCK,
            BLOCK_C=_TRANSPOSE_BLOCK,
        )
    _transposed_weight_cache[key] = (weight, transposed)
    return transposed


def _run_direction(
    layer_input,
    hx,
    layer_output,
    final_h,
    params,
    state_idx: int,
    param_idx: int,
    out_feature_offset: int,
    input_size: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    has_biases: bool,
    reverse: bool,
    batch_sizes=None,
):
    w_ih, w_hh, b_ih, b_hh = _param_group(params, param_idx, has_biases)
    # Transpose weights so the GEMM B (gate) dim is contiguous: a strided B costs ~3x on tl.dot.
    _validate_weight(w_ih, 3 * hidden_size, input_size)
    _validate_weight(w_hh, 3 * hidden_size, hidden_size)
    if w_ih.dim() == 1:
        w_ih = w_ih.view(3 * hidden_size, input_size)
    if w_hh.dim() == 1:
        w_hh = w_hh.view(3 * hidden_size, hidden_size)
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
    block_b_persist = _BLOCK_B
    block_h_persist = _block_size(hidden_size, _BLOCK_H_MAX // 2)
    grid_persist = (
        triton.cdiv(batch_size, block_b_persist),
        triton.cdiv(hidden_size, block_h_persist),
    )
    num_programs_persist = grid_persist[0] * grid_persist[1]

    if layer_input.dtype == torch.float64:
        compute_dtype = tl.float64
        gate_dtype = torch.float64
    else:
        compute_dtype = tl.float32
        gate_dtype = torch.float32

    max_persistent = _max_persistent_programs(layer_input.device)

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
        use_persistent = batch_size != 1 and num_programs_persist <= max_persistent

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
                BLOCK_B=block_b_persist,
                BLOCK_H=block_h_persist,
                BLOCK_K=block_k_h,
                NUM_PROGRAMS=num_programs_persist,
                COMPUTE_DTYPE=compute_dtype,
                PACKED=batch_sizes is not None,
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


def _validate_args(input, hx, params, has_biases, num_layers, dropout, bidirectional):
    if input.dim() != 3:
        raise RuntimeError("gru: input must have 3 dimensions")
    if num_layers <= 0:
        raise RuntimeError("gru: num_layers must be greater than zero")
    if not 0.0 <= dropout <= 1.0:
        raise RuntimeError("gru: dropout probability must be between 0 and 1")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems gru supports float16, bfloat16, float32, and float64"
        )

    # GRU has a single hidden state: hx is a Tensor, not a tuple like LSTM.
    if hx.dim() != 3:
        raise RuntimeError("gru: hidden state must have 3 dimensions")
    num_directions = 2 if bidirectional else 1
    expected_states = num_layers * num_directions
    if hx.shape[0] != expected_states:
        raise RuntimeError(
            f"gru: expected {expected_states} hidden state rows, got {hx.shape[0]}"
        )
    expected_params = expected_states * (4 if has_biases else 2)
    if len(params) != expected_params:
        raise RuntimeError(
            f"gru: expected {expected_params} parameter tensors, got {len(params)}"
        )
    if hx.device != input.device:
        raise RuntimeError("gru: input and hidden state must share a device")
    if hx.dtype != input.dtype:
        raise RuntimeError("gru: input and hidden state must share a dtype")


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
    logger.debug("GEMS GRU")
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
    logger.debug("GEMS GRU_DATA")
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
