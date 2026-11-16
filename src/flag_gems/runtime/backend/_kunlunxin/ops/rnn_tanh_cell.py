# Copyright 2026, The FlagOS Contributors.
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
#
# Kunlunxin specialization of ``aten::rnn_tanh_cell``.
#
# The generic implementation (``flag_gems/ops/rnn_tanh_cell.py``) fuses both
# projections, the bias adds, the gate sum and ``tanh`` into a single kernel.
# On XPU3 that kernel cannot be *cold*-compiled: ``tl_extra_shim.tanh`` in a
# module that also holds a ``tl.dot`` kernel is classified as a lookup-table op
# but has no table entry (``KeyError: 'tt.tanh'``, see
# ``wiki/patterns/xpu3-cold-compile-sigmoid-lut-module.md``).  It also
# miscomputes the ``input_size == 0`` case (zero-trip projection loop feeding
# the dot accumulation, measured 31/32 wrong elements).
#
# This override keeps the same numerics but splits the work into
#
#   1. a ``tl.dot``-only kernel that writes ``pre = round(g_i + g_h)`` where
#      ``g_i = round(x @ w_ih^T + b_ih)`` and ``g_h = round(hx @ w_hh^T + b_hh)``
#      (each projection rounded to the operator dtype before the sum, matching
#      ATen's separate eager ``linear``/``add`` roundings), and
#   2. the ``tanh`` tail in the sibling ``_rnn_tanh_cell_apply`` module.
#
# Two projection paths exist:
#
#   * fast -- mask-free ``tl.dot`` with the weight fed as the *transposed* B
#     operand and the batch index clamped into range (padding rows are written
#     to a row-padded staging buffer and simply never read).  It requires a
#     contiguous layout and ``input_size % 8 == 0 and hidden_size % 8 == 0``;
#     every reduction/column extent is then a block multiple, so nothing is
#     masked.  ``BM=16`` and ``BN=8`` are the smallest safe dot tiles (probed
#     bit-exact on this backend).
#   * general -- the generic masked two-dot body with ``HAS_INPUT`` as a
#     constexpr gate, so the ``input_size == 0`` case carries no zero-trip loop
#     at all.  It keeps arbitrary strides and odd sizes correct.
#
# Every ternary falls back to a FlagGems Triton implementation (the generic
# module for unsupported dtypes), never to ATen/CPU.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._kunlunxin.ops._rnn_tanh_cell_apply import (
    rnn_tanh_cell_apply_kernel as _apply_kernel,
    rnn_tanh_cell_apply_kernel_full as _apply_kernel_full,
)

logger = logging.getLogger(__name__)

_ACC_DTYPE = {
    torch.float16: tl.float32,
    torch.bfloat16: tl.float32,
    torch.float32: tl.float32,
}
_ROUND_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def _lowpow2(n):
    """Largest power of two dividing ``n`` (``n > 0``)."""
    return n & (-n)


@triton.jit
def _round(value, ROUND_DTYPE: tl.constexpr, ACC_DTYPE: tl.constexpr):
    """Round an accumulator through the operator dtype, as an eager op would."""
    if ROUND_DTYPE == ACC_DTYPE:
        return value
    else:
        return value.to(ROUND_DTYPE).to(ACC_DTYPE)


@triton.jit
def _rnn_tanh_cell_pre_fast_kernel(
    x_ptr,
    h_ptr,
    w_ih_ptr,
    w_hh_ptr,
    b_ih_ptr,
    b_hh_ptr,
    pre_ptr,
    M,
    s_xm,
    s_xk,
    s_hm,
    s_hk,
    s_win,
    s_wik,
    s_whn,
    s_whk,
    s_pm,
    s_pn,
    I: tl.constexpr,
    H: tl.constexpr,
    HAS_BIH: tl.constexpr,
    HAS_BHH: tl.constexpr,
    ACC: tl.constexpr,
    ROUND: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KI: tl.constexpr,
    BLOCK_KH: tl.constexpr,
):
    """Mask-free ``round(round(x@w_ih^T+b_ih) + round(hx@w_hh^T+b_hh))``.

    ``I``/``H`` are ``tl.constexpr`` (same recipe as ``lstm_cell``'s ``K``):
    the dot reduction loops then have a static trip count.  Feeding a runtime
    reduction bound to ``tl.dot`` produced an illegal DMA on XPU3
    (``status=700``/XID) in the first version of this override.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Wrap padded rows onto real rows: every load stays in bounds without a
    # mask, and padded rows are written into the row-padded staging buffer
    # where the activation kernel never reads them.  ``%`` (not
    # ``tl.minimum(rm, M-1)``): an integer clamp inside a ``tl.dot`` kernel
    # miscompiles this backend's SDNN/DMA path and faults on bf16 (status=700,
    # wiki ``xpu3-clamp-in-dot-kernel-bf16``).  ``% 1`` also miscompiles, so
    # ``batch == 1`` never reaches this kernel (see ``_fast_ok``).
    ram = rm % M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC)
    for k0 in range(0, I, BLOCK_KI):
        rk = k0 + tl.arange(0, BLOCK_KI)
        x = tl.load(x_ptr + ram[:, None] * s_xm + rk[None, :] * s_xk)
        w = tl.load(w_ih_ptr + rn[None, :] * s_win + rk[:, None] * s_wik)
        acc += tl.dot(x, w, input_precision="ieee", out_dtype=ACC)
    if HAS_BIH:
        acc += tl.load(b_ih_ptr + rn).to(ACC)[None, :]
    gate_i = _round(acc, ROUND, ACC)

    acc_h = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC)
    for k0 in range(0, H, BLOCK_KH):
        rk = k0 + tl.arange(0, BLOCK_KH)
        h = tl.load(h_ptr + ram[:, None] * s_hm + rk[None, :] * s_hk)
        w = tl.load(w_hh_ptr + rn[None, :] * s_whn + rk[:, None] * s_whk)
        acc_h += tl.dot(h, w, input_precision="ieee", out_dtype=ACC)
    if HAS_BHH:
        acc_h += tl.load(b_hh_ptr + rn).to(ACC)[None, :]
    gate_h = _round(acc_h, ROUND, ACC)

    pre = _round(gate_i + gate_h, ROUND, ACC)
    tl.store(
        pre_ptr + rm[:, None] * s_pm + rn[None, :],
        pre.to(pre_ptr.dtype.element_ty),
    )


@triton.jit
def _rnn_tanh_cell_pre_general_kernel(
    x_ptr,
    h_ptr,
    w_ih_ptr,
    w_hh_ptr,
    b_ih_ptr,
    b_hh_ptr,
    pre_ptr,
    B,
    s_xb,
    s_xk,
    s_hb,
    s_hk,
    s_wi_n,
    s_wi_k,
    s_wh_n,
    s_wh_k,
    s_pb,
    s_pn,
    I: tl.constexpr,
    H: tl.constexpr,
    HAS_INPUT: tl.constexpr,
    HAS_BIH: tl.constexpr,
    HAS_BHH: tl.constexpr,
    ACC: tl.constexpr,
    ROUND: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Generic masked body, minus the activation.

    ``HAS_INPUT`` is constexpr: at ``input_size == 0`` the projection loop is
    not emitted at all (the zero-trip loop + dot combination miscomputes on this
    backend), leaving only the hidden projection.  ``I``/``H`` are constexpr so
    both reduction loops have a static trip count (see the fast kernel note).
    """
    offs_b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    b_mask = offs_b < B
    n_mask = offs_n < H

    acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=ACC)
    if HAS_INPUT:
        for k0 in range(0, I, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I
            x = tl.load(
                x_ptr + offs_b[:, None] * s_xb + offs_k[None, :] * s_xk,
                mask=b_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                w_ih_ptr + offs_n[None, :] * s_wi_n + offs_k[:, None] * s_wi_k,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(x, w, input_precision="ieee", out_dtype=ACC)
    if HAS_BIH:
        acc += tl.load(b_ih_ptr + offs_n, mask=n_mask, other=0.0).to(ACC)[None, :]
    gate_i = _round(acc, ROUND, ACC)

    acc_h = tl.zeros((BLOCK_B, BLOCK_H), dtype=ACC)
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < H
        h = tl.load(
            h_ptr + offs_b[:, None] * s_hb + offs_k[None, :] * s_hk,
            mask=b_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_hh_ptr + offs_n[None, :] * s_wh_n + offs_k[:, None] * s_wh_k,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc_h += tl.dot(h, w, input_precision="ieee", out_dtype=ACC)
    if HAS_BHH:
        acc_h += tl.load(b_hh_ptr + offs_n, mask=n_mask, other=0.0).to(ACC)[None, :]
    gate_h = _round(acc_h, ROUND, ACC)

    pre = _round(gate_i + gate_h, ROUND, ACC)
    tl.store(
        pre_ptr + offs_b[:, None] * s_pb + offs_n[None, :] * s_pn,
        pre.to(pre_ptr.dtype.element_ty),
        mask=b_mask[:, None] & n_mask[None, :],
    )


def _block(value, limit):
    """Smallest power of two at least ``value``, clamped to [16, limit]."""
    return max(16, min(limit, triton.next_power_of_2(max(int(value), 1))))


def _proj_block_k(k):
    """Largest ``{64, 32, 16, 8}`` tile that divides the reduction length."""
    if k % 64 == 0:
        return 64
    if k % 32 == 0:
        return 32
    if k % 16 == 0:
        return 16
    return 8


def _contiguous_2d(t):
    return t.is_contiguous() and t.stride(1) == 1


def _fast_ok(input, hx, w_ih, w_hh, b_ih, b_hh, input_size, hidden_size):
    if input.dtype not in _ACC_DTYPE:
        return False
    # ``rm % M`` (used for the batch wrap) miscompiles at M == 1.
    if input.shape[0] < 2:
        return False
    if input_size == 0 or input_size % 8 != 0 or hidden_size % 8 != 0:
        return False
    for t in (input, hx, w_ih, w_hh):
        if not _contiguous_2d(t):
            return False
    for b in (b_ih, b_hh):
        if b is not None and not b.is_contiguous():
            return False
    return True


def _check_cell_args(name, input, hx, w_ih, w_hh, b_ih, b_hh, gates):
    """Reproduce the ATen shape/dtype contract for a single RNN cell."""
    if input.dim() < 2:
        raise IndexError(
            "Dimension out of range (expected to be in range of [-1, 0], but got 1)"
        )
    if input.dim() > 2 or input.shape[1] != w_ih.shape[1]:
        raise RuntimeError(
            "input has inconsistent input_size: got %d expected %d"
            % (input.shape[1], w_ih.shape[1])
        )
    if hx.shape[0] != input.shape[0]:
        raise RuntimeError(
            "Input batch size %d doesn't match hidden0 batch size %d"
            % (input.shape[0], hx.shape[0])
        )
    if hx.dim() != 2:
        raise RuntimeError("hidden0 is expected to be 2-D, got %d-D instead" % hx.dim())
    if hx.shape[1] != w_hh.shape[1]:
        raise RuntimeError(
            "hidden0 has inconsistent hidden_size: got %d, expected %d"
            % (hx.shape[1], w_hh.shape[1])
        )
    hidden_size = w_hh.shape[1]
    if w_ih.shape[0] != gates * hidden_size or w_hh.shape[0] != gates * hidden_size:
        raise RuntimeError(
            "%s: expected weight matrices with %d rows, got w_ih with %d and "
            "w_hh with %d" % (name, gates * hidden_size, w_ih.shape[0], w_hh.shape[0])
        )
    for label, bias in (("b_ih", b_ih), ("b_hh", b_hh)):
        if bias is None:
            continue
        if bias.dim() != 1 or bias.shape[0] != gates * hidden_size:
            raise RuntimeError(
                "Expected tensor for argument '%s' to have %d elements; but it "
                "actually has %d elements" % (label, gates * hidden_size, bias.numel())
            )
    for label, tensor in (
        ("hx", hx),
        ("w_ih", w_ih),
        ("w_hh", w_hh),
        ("b_ih", b_ih),
        ("b_hh", b_hh),
    ):
        if tensor is not None and tensor.dtype != input.dtype:
            raise RuntimeError(
                "expected %s and %s to have the same dtype, but got: %s != %s"
                % ("input", label, input.dtype, tensor.dtype)
            )
    if input.is_complex():
        raise RuntimeError('"%s" not implemented for complex dtypes' % name)


def _apply(out, pre, n):
    """Launch the ``tanh`` tail over the ``n = batch * hidden`` real elements."""
    flat_block = min(1024, max(64, _lowpow2(n)))
    if n % flat_block == 0:
        _apply_kernel_full[(n // flat_block,)](pre, out, BLOCK_SIZE=flat_block)
    else:
        _apply_kernel[(triton.cdiv(n, flat_block),)](
            pre, out, n, BLOCK_SIZE=flat_block
        )


def rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
    """Single tanh RNN cell step (``aten::rnn_tanh_cell``)."""
    logger.debug("GEMS RNN_TANH_CELL")
    _check_cell_args("rnn_tanh_cell", input, hx, w_ih, w_hh, b_ih, b_hh, 1)

    batch, input_size = input.shape
    hidden_size = w_hh.shape[1]
    out = torch.empty((batch, hidden_size), dtype=input.dtype, device=input.device)
    if batch == 0 or hidden_size == 0:
        return out

    if input.dtype not in _ACC_DTYPE:
        from flag_gems.ops.rnn_tanh_cell import rnn_tanh_cell as _generic

        return _generic(input, hx, w_ih, w_hh, b_ih, b_hh)

    acc = _ACC_DTYPE[input.dtype]
    rnd = _ROUND_DTYPE[input.dtype]
    # Dummy pointers for absent biases; HAS_BIH/HAS_BHH gate the loads.
    bi = b_ih if b_ih is not None else input
    bh = b_hh if b_hh is not None else input

    if _fast_ok(input, hx, w_ih, w_hh, b_ih, b_hh, input_size, hidden_size):
        block_m = min(128, max(16, triton.next_power_of_2(batch)))
        block_n = min(128, _lowpow2(hidden_size))
        block_ki = _proj_block_k(input_size)
        block_kh = _proj_block_k(hidden_size)
        mp = ((batch + block_m - 1) // block_m) * block_m
        pre = torch.empty((mp, hidden_size), dtype=input.dtype, device=input.device)
        _rnn_tanh_cell_pre_fast_kernel[
            (mp // block_m, hidden_size // block_n)
        ](
            input,
            hx,
            w_ih,
            w_hh,
            bi,
            bh,
            pre,
            batch,
            input.stride(0),
            input.stride(1),
            hx.stride(0),
            hx.stride(1),
            w_ih.stride(0),
            w_ih.stride(1),
            w_hh.stride(0),
            w_hh.stride(1),
            pre.stride(0),
            pre.stride(1),
            I=input_size,
            H=hidden_size,
            HAS_BIH=b_ih is not None,
            HAS_BHH=b_hh is not None,
            ACC=acc,
            ROUND=rnd,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_KI=block_ki,
            BLOCK_KH=block_kh,
            num_warps=4,
        )
        _apply(out, pre, batch * hidden_size)
        return out

    block_b = _block(batch, 16)
    block_h = _block(hidden_size, 32)
    block_k = _block(max(input_size, hidden_size), 64)
    pre = torch.empty((batch, hidden_size), dtype=input.dtype, device=input.device)
    _rnn_tanh_cell_pre_general_kernel[
        (triton.cdiv(batch, block_b), triton.cdiv(hidden_size, block_h))
    ](
        input,
        hx,
        w_ih,
        w_hh,
        bi,
        bh,
        pre,
        batch,
        input.stride(0),
        input.stride(1),
        hx.stride(0),
        hx.stride(1),
        w_ih.stride(0),
        w_ih.stride(1),
        w_hh.stride(0),
        w_hh.stride(1),
        pre.stride(0),
        pre.stride(1),
        I=input_size,
        H=hidden_size,
        HAS_INPUT=input_size > 0,
        HAS_BIH=b_ih is not None,
        HAS_BHH=b_hh is not None,
        ACC=acc,
        ROUND=rnd,
        BLOCK_B=block_b,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        num_warps=4,
    )
    _apply(out, pre, batch * hidden_size)
    return out
