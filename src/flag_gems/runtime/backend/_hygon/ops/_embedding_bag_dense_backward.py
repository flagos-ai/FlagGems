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

"""Triton implementation of aten::_embedding_bag_dense_backward.

Semantics verified against the ATen reference on this target (tiny probes):
  * mode 0 = sum, 1 = mean, 2 = max
  * sum/mean: g[idx_i, :] += grad[bag_i, :] * s_i, bag_i = offset2bag[i]
      s_i = psw_i * (1/bag_size[bag_i] if mean else 1) * (1/global_count[idx_i] if scale_grad_by_freq)
  * padding_idx (only when 0 <= padding_idx < num_weights): contributions with
    idx == padding_idx are dropped; bag_size is *not* adjusted.
  * max: g[maximum_indices[b, d], d] += grad[b, d]  (drop pad / out-of-range)
  * dtypes fp32/fp16/bf16/fp64 supported; index metadata may be int32 or int64.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MODE_MAP = {"sum": 0, "mean": 1, "max": 2}

ZERO_BLOCK = 4096
COUNT_BLOCK = 2048
MAX_BLOCK = 1024


@triton.jit
def _convert_reset_kernel(src_ptr, dst_ptr, n_elem, BLOCK: tl.constexpr):
    """Round the fp32 accumulator once into the 16-bit output and re-zero it.

    The re-zero keeps the invariant "accumulator is all zeros on entry", so the
    scatter of the next call needs no separate memset kernel.
    """
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elem
    v = tl.load(src_ptr + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, v.to(dst_ptr.dtype.element_ty), mask=mask)
    tl.store(src_ptr + offs, tl.zeros([BLOCK], dtype=tl.float32), mask=mask)


@triton.jit
def _count_kernel(idx_ptr, cnt_ptr, N, W, stride_i, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N
    i = tl.load(idx_ptr + offs * stride_i, mask=mask, other=-1).to(tl.int64)
    mask = mask & (i >= 0) & (i < W)
    tl.atomic_add(cnt_ptr + i, 1, mask=mask, sem="relaxed")


@triton.jit
def _scatter_kernel(
    grad_ptr,
    idx_ptr,
    o2b_ptr,
    bsz_ptr,
    psw_ptr,
    cnt_ptr,
    out_ptr,
    N,
    B,
    W,
    D,
    stride_gb,
    stride_gd,
    stride_i,
    stride_b,
    stride_p,
    pad,
    BLOCK_I: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_MEAN: tl.constexpr,
    HAS_PSW: tl.constexpr,
    HAS_FREQ: tl.constexpr,
    HAS_PAD: tl.constexpr,
):
    pid_i = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I).to(tl.int64)
    mask_i = offs_i < N
    idx = tl.load(idx_ptr + offs_i * stride_i, mask=mask_i, other=-1).to(tl.int64)
    b = tl.load(o2b_ptr + offs_i * stride_b, mask=mask_i, other=-1).to(tl.int64)

    live = mask_i & (b >= 0) & (b < B) & (idx >= 0) & (idx < W)
    if HAS_PAD:
        live = live & (idx != pad)

    scale = tl.full([BLOCK_I], 1.0, tl.float32)
    if HAS_PSW:
        p = tl.load(psw_ptr + offs_i * stride_p, mask=live, other=0.0).to(tl.float32)
        scale = scale * p
    if IS_MEAN:
        sz = tl.load(bsz_ptr + b, mask=live, other=0).to(tl.float32)
        scale = scale * tl.where(sz > 0, 1.0 / sz, 0.0)
    if HAS_FREQ:
        c = tl.load(cnt_ptr + idx, mask=live, other=0).to(tl.float32)
        scale = scale * tl.where(c > 0, 1.0 / c, 0.0)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    mask_d = offs_d < D
    m = live[:, None] & mask_d[None, :]

    v = tl.load(
        grad_ptr + b[:, None] * stride_gb + offs_d[None, :] * stride_gd,
        mask=m,
        other=0.0,
    )
    v = v.to(tl.float32) * scale[:, None]
    tl.atomic_add(
        out_ptr + idx[:, None] * D + offs_d[None, :],
        v.to(out_ptr.dtype.element_ty),
        mask=m,
        sem="relaxed",
    )


@triton.jit
def _scatter_plain_kernel(
    grad_ptr,
    idx_ptr,
    o2b_ptr,
    out_ptr,
    N,
    B,
    W,
    D,
    stride_gb,
    BLOCK_I: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Sum mode with contiguous metadata, no weights/freq/padding.

    Same arithmetic as _scatter_kernel with scale == 1.0 and no padding mask,
    but with 9 runtime arguments instead of 17 and no scalar loads, which
    shortens the host-side launch-key computation this target is bound by.
    """
    pid_i = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I).to(tl.int64)
    mask_i = offs_i < N
    idx = tl.load(idx_ptr + offs_i, mask=mask_i, other=-1).to(tl.int64)
    b = tl.load(o2b_ptr + offs_i, mask=mask_i, other=-1).to(tl.int64)
    live = mask_i & (b >= 0) & (b < B) & (idx >= 0) & (idx < W)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    m = live[:, None] & (offs_d < D)[None, :]

    v = tl.load(grad_ptr + b[:, None] * stride_gb + offs_d[None, :], mask=m, other=0.0)
    tl.atomic_add(
        out_ptr + idx[:, None] * D + offs_d[None, :],
        v.to(out_ptr.dtype.element_ty),
        mask=m,
        sem="relaxed",
    )


@triton.jit
def _rowsum_kernel(
    grad_ptr,
    idx_ptr,
    o2b_ptr,
    out_ptr,
    N,
    B,
    W,
    D,
    stride_gb,
    BLOCK_I: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per output row: gather, fp32-accumulate, single store.

    Equivalent to the scatter for mode 0 with scale == 1 and no padding, but
    needs no zero-initialised output, no atomics and no conversion pass. The
    index list is scanned in BLOCK_I chunks and only the matching grad rows are
    fetched (the load mask is row dependent), so memory traffic stays at one
    read of grad per contribution. Measured to win only for tiny gathers
    (W * N * BLOCK_D <= ~5e5), where the two-kernel path costs more.
    """
    r = tl.program_id(0).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    d_ok = offs_d < D

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for i0 in range(0, N, BLOCK_I):
        ii = i0 + tl.arange(0, BLOCK_I).to(tl.int64)
        i_ok = ii < N
        idxv = tl.load(idx_ptr + ii, mask=i_ok, other=-1).to(tl.int64)
        bv = tl.load(o2b_ptr + ii, mask=i_ok, other=-1).to(tl.int64)
        sel = i_ok & (bv >= 0) & (bv < B) & (idxv == r)
        v = tl.load(
            grad_ptr + bv[:, None] * stride_gb + offs_d[None, :],
            mask=sel[:, None] & d_ok[None, :],
            other=0.0,
        )
        acc += tl.sum(v.to(tl.float32), axis=0)

    tl.store(
        out_ptr + r * D + offs_d,
        acc.to(out_ptr.dtype.element_ty),
        mask=d_ok,
    )


@triton.jit
def _max_scatter_kernel(
    grad_ptr,
    mi_ptr,
    out_ptr,
    n_max,
    D,
    W,
    stride_gb,
    stride_gd,
    pad,
    HAS_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_max
    j = tl.load(mi_ptr + offs, mask=mask, other=-1).to(tl.int64)
    d = offs % D
    bag = offs // D
    live = mask & (j >= 0) & (j < W)
    if HAS_PAD:
        live = live & (j != pad)
    v = tl.load(grad_ptr + bag * stride_gb + d * stride_gd, mask=live, other=0.0)
    tl.atomic_add(
        out_ptr + j * D + d,
        v.to(out_ptr.dtype.element_ty),
        mask=live,
        sem="relaxed",
    )


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _flat_stride(t):
    return t.stride(0) if t.dim() == 1 else 1


# fp32 accumulator scratch for 16-bit outputs. It never holds result or
# input-derived data: _convert_reset_kernel leaves it all-zero after every use
# and it is created zeroed, so the invariant "all zeros on entry" always holds
# and the scatter can accumulate into it directly. Not returned, not read for
# any result. Keyed by (device, element count), bounded in size.
_SCRATCH = {}


def _scratch(device, n):
    key = (str(device), n)
    buf = _SCRATCH.get(key)
    if buf is None:
        if len(_SCRATCH) > 16:
            _SCRATCH.clear()
        buf = torch.zeros(n, dtype=torch.float32, device=device)
        _SCRATCH[key] = buf
    return buf


def _embedding_bag_dense_backward(
    grad,
    indices,
    offset2bag,
    bag_size,
    maximum_indices,
    num_weights,
    scale_grad_by_freq,
    mode,
    per_sample_weights,
    padding_idx=-1,
    **_kwargs,
):
    logger.debug("GEMS_HYGON _EMBEDDING_BAG_DENSE_BACKWARD")
    if isinstance(mode, str):
        mode = _MODE_MAP.get(mode.lower(), 0)
    mode = int(mode)
    pad = int(padding_idx)

    W = int(num_weights)
    if W < 0:
        W = 0
    D = int(grad.shape[1]) if grad.dim() > 1 else int(grad.numel())
    B = int(grad.shape[0])
    stride_gb = int(grad.stride(0))
    stride_gd = int(grad.stride(1)) if grad.dim() > 1 else 1

    N = int(indices.numel())
    stride_i = _flat_stride(indices)
    stride_b = _flat_stride(offset2bag)
    has_psw = per_sample_weights is not None and per_sample_weights.numel() > 0
    stride_p = _flat_stride(per_sample_weights) if has_psw else 1
    use_freq = bool(scale_grad_by_freq)
    has_pad = 0 <= pad < W

    block_d = min(256, _next_pow2(D))
    num_warps = max(1, min(8, block_d // 64))
    plain = (
        mode == 0
        and not has_psw
        and not use_freq
        and not has_pad
        and stride_i == 1
        and stride_b == 1
        and stride_gd == 1
    )

    # Single-device-op path for tiny plain-sum gathers (measured 7.8-9.6 us vs
    # 9.9-11.5 us for the two-op path at <=5e5 lane-operations, and 27-41 us at
    # 6.5e6, hence the deliberately tight bound).
    if (
        plain
        and grad.dtype != torch.float64
        and N > 0
        and W > 0
        and 0 < D <= 256
        and W * N * block_d <= 500000
    ):
        out = torch.empty((W, D), dtype=grad.dtype, device=grad.device)
        _rowsum_kernel[(W,)](
            grad,
            indices,
            offset2bag,
            out,
            N,
            B,
            W,
            D,
            stride_gb,
            max(1, min(256, 4096 // block_d)),
            block_d,
            num_warps=4,
        )
        return out

    # The ATen oracle accumulates every contribution in fp32 and rounds once at
    # the end (verified bit-exact on this target for fp16/bf16 sum/mean/max).
    # Mirror that: fp32/fp64 accumulate straight into a zeroed output, while
    # 16-bit outputs accumulate into an fp32 buffer that _convert_reset_kernel
    # rounds once and re-zeroes (so no host memset is needed).
    n_out = W * D
    if grad.dtype == torch.float32 or grad.dtype == torch.float64:
        out = torch.zeros((W, D), dtype=grad.dtype, device=grad.device)
        acc = out
    else:
        out = torch.empty((W, D), dtype=grad.dtype, device=grad.device)
        acc = _scratch(grad.device, n_out)

    if mode == 2:
        n_max = int(maximum_indices.numel())
        if n_max > 0 and D > 0 and W > 0:
            _max_scatter_kernel[(triton.cdiv(n_max, MAX_BLOCK),)](
                grad,
                maximum_indices,
                acc,
                n_max,
                D,
                W,
                stride_gb,
                stride_gd,
                pad,
                has_pad,
                MAX_BLOCK,
                num_warps=4,
            )
        if acc is not out and n_out > 0:
            _convert_reset_kernel[(triton.cdiv(n_out, ZERO_BLOCK),)](
                acc, out, n_out, ZERO_BLOCK, num_warps=4
            )
        return out

    if N == 0 or D == 0 or W == 0:
        if acc is not out and n_out > 0:
            _convert_reset_kernel[(triton.cdiv(n_out, ZERO_BLOCK),)](
                acc, out, n_out, ZERO_BLOCK, num_warps=4
            )
        return out

    if use_freq:
        counts = torch.zeros((W,), dtype=torch.int32, device=grad.device)
        _count_kernel[(triton.cdiv(N, COUNT_BLOCK),)](
            indices,
            counts,
            N,
            W,
            stride_i,
            COUNT_BLOCK,
            num_warps=4,
        )
    else:
        counts = bag_size

    block_i = 1
    grid = (triton.cdiv(N, block_i), triton.cdiv(D, block_d))
    if plain:
        _scatter_plain_kernel[grid](
            grad,
            indices,
            offset2bag,
            acc,
            N,
            B,
            W,
            D,
            stride_gb,
            block_i,
            block_d,
            num_warps=num_warps,
        )
    else:
        _scatter_kernel[grid](
            grad,
            indices,
            offset2bag,
            bag_size,
            per_sample_weights if has_psw else bag_size,
            counts,
            acc,
            N,
            B,
            W,
            D,
            stride_gb,
            stride_gd,
            stride_i,
            stride_b,
            stride_p,
            pad,
            block_i,
            block_d,
            (mode == 1),
            has_psw,
            use_freq,
            has_pad,
            num_warps=num_warps,
        )
    if acc is not out and n_out > 0:
        _convert_reset_kernel[(triton.cdiv(n_out, ZERO_BLOCK),)](
            acc, out, n_out, ZERO_BLOCK, num_warps=4
        )
    return out
