import logging

import torch
import triton
import triton.language as tl

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
    _gru_input_gemm_kernel,
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
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)


_BLOCK_B = 16
_BLOCK_H_MAX = 32
_BLOCK_K_MAX = 64
_GEMV_BLOCK_N_MAX = 16
_GEMV_BLOCK_K_MAX = 128
_GEMV_HOIST_BLOCK_N = 2
_GEMV_HOIST_NUM_WARPS = 1
_SPLIT_BLOCK = 1024


def _fma_block_k(size: int, cap: int) -> int:
    # K chunk for the FMA fallbacks, capped to stay register-resident.
    return min(cap, max(8, _ceil_power_of_2(size)))


@libentry()
@triton.jit
def _split_fp64_kernel(
    x_ptr,
    hi_ptr,
    lo_ptr,
    S,
    B,
    I,
    x_stride_s,
    x_stride_b,
    x_stride_f,
    BLOCK: tl.constexpr,
):
    # fp64 -> (fp32 hi, fp32 lo) with lo = fp32(x - fp64(hi));
    # x = hi + lo + O(2^-48 |x|).
    # hi/lo are written in row-major flat order; x may have any strides.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = S * B * I
    m = offs < n
    s = offs // (B * I)
    rem = offs % (B * I)
    b = rem // I
    i = rem % I
    x = tl.load(
        x_ptr + s * x_stride_s + b * x_stride_b + i * x_stride_f, mask=m, other=0.0
    )
    hi = x.to(tl.float32)
    lo = (x - hi.to(tl.float64)).to(tl.float32)
    tl.store(hi_ptr + offs, hi, mask=m)
    tl.store(lo_ptr + offs, lo, mask=m)


@libentry()
@triton.jit
def _combine3_bias_kernel(
    u1_ptr,
    u2_ptr,
    u3_ptr,
    b_ptr,
    out_ptr,
    n,
    b_size,
    b_stride,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # fp64: out = u1 + u2 + u3 (+ bias broadcast over the last dim);
    # u1..u3 are the three fp32 split-dot products.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    acc = (
        tl.load(u1_ptr + offs, mask=m, other=0.0).to(tl.float64)
        + tl.load(u2_ptr + offs, mask=m, other=0.0).to(tl.float64)
        + tl.load(u3_ptr + offs, mask=m, other=0.0).to(tl.float64)
    )
    if HAS_BIAS:
        acc += tl.load(b_ptr + (offs % b_size) * b_stride, mask=m, other=0.0).to(
            tl.float64
        )
    tl.store(out_ptr + offs, acc, mask=m)


@libentry()
@triton.jit
def _gru_step_split_kernel(
    u_ptr,
    h_prev_ptr,
    w_hh_hi_ptr,
    w_hh_lo_ptr,
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
    PACKED: tl.constexpr,
):
    # fp64 recurrence via 3 fp32 tl.dot per gate: hi*hi + hi*lo + lo*hi
    # (fp64 tl.dot is broken on metax).
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    bh_mask = (offs_b[:, None] < batch_size) & (offs_h[None, :] < hidden_size)

    if PACKED:
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
    r_in = tl.load(u_ptr + u_base, mask=bh_mask, other=0.0)
    z_in = tl.load(u_ptr + u_base + hidden_size * u_stride_f, mask=bh_mask, other=0.0)
    n_in = tl.load(
        u_ptr + u_base + 2 * hidden_size * u_stride_f, mask=bh_mask, other=0.0
    )

    r1 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
    r2 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
    r3 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
    z1 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
    z2 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
    z3 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
    n1 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
    n2 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
    n3 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(hidden_size, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        h64 = tl.load(
            h_prev_ptr + offs_b[:, None] * hidden_size + offs_k[None, :],
            mask=(offs_b[:, None] < batch_size) & (offs_k[None, :] < hidden_size),
            other=0.0,
        )
        h_hi = h64.to(tl.float32)
        h_lo = (h64 - h_hi.to(tl.float64)).to(tl.float32)
        wr_hi = tl.load(
            w_hh_hi_ptr
            + offs_k[:, None] * w_hh_stride_r
            + offs_h[None, :] * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        wr_lo = tl.load(
            w_hh_lo_ptr
            + offs_k[:, None] * w_hh_stride_r
            + offs_h[None, :] * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        wz_hi = tl.load(
            w_hh_hi_ptr
            + offs_k[:, None] * w_hh_stride_r
            + (hidden_size + offs_h[None, :]) * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        wz_lo = tl.load(
            w_hh_lo_ptr
            + offs_k[:, None] * w_hh_stride_r
            + (hidden_size + offs_h[None, :]) * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        wn_hi = tl.load(
            w_hh_hi_ptr
            + offs_k[:, None] * w_hh_stride_r
            + (2 * hidden_size + offs_h[None, :]) * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        wn_lo = tl.load(
            w_hh_lo_ptr
            + offs_k[:, None] * w_hh_stride_r
            + (2 * hidden_size + offs_h[None, :]) * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        r1 += tl.dot(h_hi, wr_hi, out_dtype=tl.float32, allow_tf32=False)
        r2 += tl.dot(h_hi, wr_lo, out_dtype=tl.float32, allow_tf32=False)
        r3 += tl.dot(h_lo, wr_hi, out_dtype=tl.float32, allow_tf32=False)
        z1 += tl.dot(h_hi, wz_hi, out_dtype=tl.float32, allow_tf32=False)
        z2 += tl.dot(h_hi, wz_lo, out_dtype=tl.float32, allow_tf32=False)
        z3 += tl.dot(h_lo, wz_hi, out_dtype=tl.float32, allow_tf32=False)
        n1 += tl.dot(h_hi, wn_hi, out_dtype=tl.float32, allow_tf32=False)
        n2 += tl.dot(h_hi, wn_lo, out_dtype=tl.float32, allow_tf32=False)
        n3 += tl.dot(h_lo, wn_hi, out_dtype=tl.float32, allow_tf32=False)

    r_acc = r1.to(tl.float64) + r2.to(tl.float64) + r3.to(tl.float64) + r_in
    z_acc = z1.to(tl.float64) + z2.to(tl.float64) + z3.to(tl.float64) + z_in
    n_h_acc = n1.to(tl.float64) + n2.to(tl.float64) + n3.to(tl.float64)

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

    h_prev = tl.load(
        h_prev_ptr + offs_b[:, None] * hidden_size + offs_h[None, :],
        mask=bh_mask,
        other=0.0,
    )
    h_next = (1.0 - z_gate) * n_gate + z_gate * h_prev

    if PACKED:
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
def _gru_input_gemm_fma_kernel(
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
    pid_b = tl.program_id(0)
    seq_idx = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_mask = offs_b < batch_size
    n_mask = offs_n < 3 * hidden_size

    if PACKED:
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
        acc += tl.sum(x[:, :, None] * w[None, :, :], axis=1)

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
def _gru_persistent_split_kernel(
    u_ptr,
    h_ptr,
    w_hh_hi_ptr,
    w_hh_lo_ptr,
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
    PACKED: tl.constexpr,
):
    # Persistent fp64 recurrence via 3 fp32 tl.dot per gate.
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    bh_mask = (offs_b[:, None] < batch_size) & (offs_h[None, :] < hidden_size)
    state_offsets = offs_b[:, None] * hidden_size + offs_h[None, :]
    buf_stride = batch_size * hidden_size

    h_cur = tl.load(h_ptr + state_offsets, mask=bh_mask, other=0.0).to(tl.float64)

    for step in range(seq_len):
        t = seq_len - 1 - step if REVERSE else step
        read_base = (step % 2) * buf_stride
        write_base = ((step + 1) % 2) * buf_stride

        u_base = (
            t * u_stride_s + offs_b[:, None] * u_stride_b + offs_h[None, :] * u_stride_f
        )
        r_in = tl.load(u_ptr + u_base, mask=bh_mask, other=0.0)
        z_in = tl.load(
            u_ptr + u_base + hidden_size * u_stride_f, mask=bh_mask, other=0.0
        )
        n_in = tl.load(
            u_ptr + u_base + 2 * hidden_size * u_stride_f, mask=bh_mask, other=0.0
        )

        r1 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        r2 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        r3 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        z1 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        z2 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        z3 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        n1 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        n2 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        n3 = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        for k_block in range(0, tl.cdiv(hidden_size, BLOCK_K)):
            offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            h64 = tl.load(
                h_ptr + read_base + offs_b[:, None] * hidden_size + offs_k[None, :],
                mask=(offs_b[:, None] < batch_size) & (offs_k[None, :] < hidden_size),
                other=0.0,
            )
            h_hi = h64.to(tl.float32)
            h_lo = (h64 - h_hi.to(tl.float64)).to(tl.float32)
            wr_hi = tl.load(
                w_hh_hi_ptr
                + offs_k[:, None] * w_hh_stride_r
                + offs_h[None, :] * w_hh_stride_c,
                mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
                other=0.0,
            )
            wr_lo = tl.load(
                w_hh_lo_ptr
                + offs_k[:, None] * w_hh_stride_r
                + offs_h[None, :] * w_hh_stride_c,
                mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
                other=0.0,
            )
            wz_hi = tl.load(
                w_hh_hi_ptr
                + offs_k[:, None] * w_hh_stride_r
                + (hidden_size + offs_h[None, :]) * w_hh_stride_c,
                mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
                other=0.0,
            )
            wz_lo = tl.load(
                w_hh_lo_ptr
                + offs_k[:, None] * w_hh_stride_r
                + (hidden_size + offs_h[None, :]) * w_hh_stride_c,
                mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
                other=0.0,
            )
            wn_hi = tl.load(
                w_hh_hi_ptr
                + offs_k[:, None] * w_hh_stride_r
                + (2 * hidden_size + offs_h[None, :]) * w_hh_stride_c,
                mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
                other=0.0,
            )
            wn_lo = tl.load(
                w_hh_lo_ptr
                + offs_k[:, None] * w_hh_stride_r
                + (2 * hidden_size + offs_h[None, :]) * w_hh_stride_c,
                mask=(offs_k[:, None] < hidden_size) & (offs_h[None, :] < hidden_size),
                other=0.0,
            )
            r1 += tl.dot(h_hi, wr_hi, out_dtype=tl.float32, allow_tf32=False)
            r2 += tl.dot(h_hi, wr_lo, out_dtype=tl.float32, allow_tf32=False)
            r3 += tl.dot(h_lo, wr_hi, out_dtype=tl.float32, allow_tf32=False)
            z1 += tl.dot(h_hi, wz_hi, out_dtype=tl.float32, allow_tf32=False)
            z2 += tl.dot(h_hi, wz_lo, out_dtype=tl.float32, allow_tf32=False)
            z3 += tl.dot(h_lo, wz_hi, out_dtype=tl.float32, allow_tf32=False)
            n1 += tl.dot(h_hi, wn_hi, out_dtype=tl.float32, allow_tf32=False)
            n2 += tl.dot(h_hi, wn_lo, out_dtype=tl.float32, allow_tf32=False)
            n3 += tl.dot(h_lo, wn_hi, out_dtype=tl.float32, allow_tf32=False)

        r_acc = r1.to(tl.float64) + r2.to(tl.float64) + r3.to(tl.float64) + r_in
        z_acc = z1.to(tl.float64) + z2.to(tl.float64) + z3.to(tl.float64) + z_in
        n_h_acc = n1.to(tl.float64) + n2.to(tl.float64) + n3.to(tl.float64)

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

        # Grid-wide barrier: release this program's h write, then acquire
        # everyone else's before the next step reads h.
        tl.atomic_add(barrier_ptr + step, 1, sem="acq_rel")
        while tl.atomic_add(barrier_ptr + step, 0, sem="acquire") < NUM_PROGRAMS:
            pass


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
    _validate_weight(w_ih, 3 * hidden_size, input_size)
    _validate_weight(w_hh, 3 * hidden_size, hidden_size)
    if w_ih.dim() == 1:
        w_ih = w_ih.view(3 * hidden_size, input_size)
    if w_hh.dim() == 1:
        w_hh = w_hh.view(3 * hidden_size, hidden_size)
    # Transposed (K, 3H) so the B operand's gate dim is contiguous.
    w_ih = _transpose_weight(w_ih, 3 * hidden_size, input_size)
    w_hh = _transpose_weight(w_hh, 3 * hidden_size, hidden_size)
    w_ih_stride_r, w_ih_stride_c = w_ih.stride(0), w_ih.stride(1)
    w_hh_stride_r, w_hh_stride_c = w_hh.stride(0), w_hh.stride(1)
    b_ih_stride = _bias_stride(b_ih, 3 * hidden_size) if has_biases else 1
    b_hh_stride = _bias_stride(b_hh, 3 * hidden_size) if has_biases else 1

    if batch_size == 0:
        return

    if layer_input.dtype == torch.float64:
        compute_dtype = tl.float64
        gate_dtype = torch.float64
    else:
        compute_dtype = tl.float32
        gate_dtype = torch.float32
    # the split path emulates fp64 GEMM
    # with three fp32 tl.dot products (hi*hi + hi*lo + lo*hi), which is
    # exact to ~1e-7 relative per GEMM.
    use_split = compute_dtype == tl.float64

    max_persistent = _max_persistent_programs(layer_input.device)

    block_k_h = _block_size(hidden_size, _BLOCK_K_MAX)
    block_b_persist = min(_BLOCK_B, _ceil_power_of_2(batch_size))
    block_h_persist = _block_size(hidden_size, _BLOCK_H_MAX // 2)
    grid_persist = (
        triton.cdiv(batch_size, block_b_persist),
        triton.cdiv(hidden_size, block_h_persist),
    )
    num_programs_persist = grid_persist[0] * grid_persist[1]

    # Input-side pre-activations for every timestep in one batched GEMM.
    input_gates = _empty(
        (seq_len, batch_size, 3 * hidden_size), gate_dtype, layer_input.device
    )
    input_gemm_grid = lambda META: (
        triton.cdiv(batch_size, META["BLOCK_B"]),
        seq_len,
        triton.cdiv(3 * hidden_size, META["BLOCK_N"]),
    )
    # fp64: emulate the input GEMM with three fp32 dots (the autotuned kernel);
    # keep the FMA kernel only for tiny input sizes (< 16 dot K).
    use_split_input = use_split and _ceil_power_of_2(input_size) >= 16
    use_fma_input = (
        compute_dtype == tl.float64 and not use_split_input
    ) or _ceil_power_of_2(input_size) < 16

    with torch_device_fn.device(layer_input.device):
        if use_split_input:
            x_hi = _empty(layer_input.shape, torch.float32, layer_input.device)
            x_lo = _empty(layer_input.shape, torch.float32, layer_input.device)
            w_ih_hi = _empty(w_ih.shape, torch.float32, layer_input.device)
            w_ih_lo = _empty(w_ih.shape, torch.float32, layer_input.device)
            u1 = _empty(input_gates.shape, torch.float32, layer_input.device)
            u2 = _empty(input_gates.shape, torch.float32, layer_input.device)
            u3 = _empty(input_gates.shape, torch.float32, layer_input.device)
            _split_fp64_kernel[(triton.cdiv(layer_input.numel(), _SPLIT_BLOCK),)](
                layer_input,
                x_hi,
                x_lo,
                seq_len,
                batch_size,
                input_size,
                layer_input.stride(0),
                layer_input.stride(1),
                layer_input.stride(2),
                BLOCK=_SPLIT_BLOCK,
            )
            _split_fp64_kernel[(triton.cdiv(w_ih.numel(), _SPLIT_BLOCK),)](
                w_ih,
                w_ih_hi,
                w_ih_lo,
                input_size,
                3 * hidden_size,
                1,
                w_ih_stride_r,
                w_ih_stride_c,
                1,
                BLOCK=_SPLIT_BLOCK,
            )
            for x_part, w_part, u_part in (
                (x_hi, w_ih_hi, u1),
                (x_hi, w_ih_lo, u2),
                (x_lo, w_ih_hi, u3),
            ):
                _gru_input_gemm_kernel[input_gemm_grid](
                    x_part,
                    w_part,
                    b_ih,
                    u_part,
                    batch_sizes if batch_sizes is not None else x_part,
                    input_size,
                    hidden_size,
                    batch_size,
                    x_part.stride(0),
                    x_part.stride(1),
                    x_part.stride(2),
                    w_ih_stride_r,
                    w_ih_stride_c,
                    b_ih_stride,
                    u_part.stride(0),
                    u_part.stride(1),
                    u_part.stride(2),
                    PACKED=batch_sizes is not None,
                    HAS_BIAS=False,
                    COMPUTE_DTYPE=tl.float32,
                )
            _combine3_bias_kernel[(triton.cdiv(input_gates.numel(), _SPLIT_BLOCK),)](
                u1,
                u2,
                u3,
                b_ih,
                input_gates,
                input_gates.numel(),
                3 * hidden_size,
                b_ih_stride,
                HAS_BIAS=has_biases,
                BLOCK=_SPLIT_BLOCK,
            )
        elif use_fma_input:
            fma_block_b, fma_block_n = 4, 32
            fma_block_k = _fma_block_k(input_size, 128)
            fma_grid = (
                triton.cdiv(batch_size, fma_block_b),
                seq_len,
                triton.cdiv(3 * hidden_size, fma_block_n),
            )
            _gru_input_gemm_fma_kernel[fma_grid](
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
                BLOCK_B=fma_block_b,
                BLOCK_N=fma_block_n,
                BLOCK_K=fma_block_k,
                COMPUTE_DTYPE=compute_dtype,
                num_warps=4,
                num_stages=1,
            )
        else:
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

        block_k_g = _block_size(hidden_size, _GEMV_BLOCK_K_MAX)
        gemv_hoist = hidden_size <= block_k_g
        gemv_block_n = (
            _GEMV_HOIST_BLOCK_N
            if gemv_hoist
            else _block_size(hidden_size, _GEMV_BLOCK_N_MAX)
        )
        gemv_num_programs = triton.cdiv(hidden_size, gemv_block_n)
        # Grow BLOCK_N until the barrier grid is co-resident (<= 1 CTA/SM).
        while gemv_num_programs > max_persistent and gemv_block_n < hidden_size:
            gemv_block_n *= 2
            gemv_num_programs = triton.cdiv(hidden_size, gemv_block_n)

        use_gemv = batch_size == 1 and gemv_num_programs <= max_persistent
        use_persistent = batch_size != 1 and num_programs_persist <= max_persistent
        use_split_recur = use_split and not use_gemv

        # Pre-split the recurrent weight once for the split-dot kernels.
        if use_split_recur:
            w_hh_hi = _empty(w_hh.shape, torch.float32, layer_input.device)
            w_hh_lo = _empty(w_hh.shape, torch.float32, layer_input.device)
            _split_fp64_kernel[(triton.cdiv(w_hh.numel(), _SPLIT_BLOCK),)](
                w_hh,
                w_hh_hi,
                w_hh_lo,
                hidden_size,
                3 * hidden_size,
                1,
                w_hh_stride_r,
                w_hh_stride_c,
                1,
                BLOCK=_SPLIT_BLOCK,
            )

        if use_gemv:
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
            h_buf = _empty((2, batch_size, hidden_size), hx.dtype, hx.device)
            _copy_hx_slice(hx, h_buf[0], state_idx, batch_size, hidden_size)
            barrier = torch.zeros(
                (seq_len,), device=layer_input.device, dtype=torch.int32
            )
            if use_split_recur:
                _gru_persistent_split_kernel[grid_persist](
                    input_gates,
                    h_buf,
                    w_hh_hi,
                    w_hh_lo,
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
                    BLOCK_K=64,
                    NUM_PROGRAMS=num_programs_persist,
                    PACKED=batch_sizes is not None,
                    num_warps=4,
                    num_stages=1,
                )
            else:
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
            if use_split_recur:
                step_block_b, step_block_h, step_block_k = 32, 16, 64
                step_grid = (
                    triton.cdiv(batch_size, step_block_b),
                    triton.cdiv(hidden_size, step_block_h),
                )
                h_work = _empty((batch_size, hidden_size), hx.dtype, hx.device)
                _copy_hx_slice(hx, h_work, state_idx, batch_size, hidden_size)
                h_next = _empty((batch_size, hidden_size), hx.dtype, hx.device)
                for step in range(seq_len):
                    seq_idx = seq_len - 1 - step if reverse else step
                    _gru_step_split_kernel[step_grid](
                        input_gates,
                        h_work,
                        w_hh_hi,
                        w_hh_lo,
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
                        BLOCK_B=step_block_b,
                        BLOCK_H=step_block_h,
                        BLOCK_K=step_block_k,
                        PACKED=batch_sizes is not None,
                        num_warps=4,
                        num_stages=1,
                    )
                    h_work, h_next = h_next, h_work
                final_h_state = h_work
            else:
                step_block_b = _BLOCK_B
                step_block_h = 32 if hidden_size >= 1024 else 16
                step_block_k = block_k_h
                step_grid = (
                    triton.cdiv(batch_size, step_block_b),
                    triton.cdiv(hidden_size, step_block_h),
                )
                h_work = _empty((batch_size, hidden_size), hx.dtype, hx.device)
                _copy_hx_slice(hx, h_work, state_idx, batch_size, hidden_size)
                h_next = _empty((batch_size, hidden_size), hx.dtype, hx.device)
                for step in range(seq_len):
                    seq_idx = seq_len - 1 - step if reverse else step
                    _gru_step_kernel[step_grid](
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
                        BLOCK_B=step_block_b,
                        BLOCK_H=step_block_h,
                        BLOCK_K=step_block_k,
                        COMPUTE_DTYPE=compute_dtype,
                        PACKED=batch_sizes is not None,
                        num_stages=1,
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
    logger.debug("GEMS_METAX GRU")
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
    logger.debug("GEMS_METAX GRU_DATA")
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
