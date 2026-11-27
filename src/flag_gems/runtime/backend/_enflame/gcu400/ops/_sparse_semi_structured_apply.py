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
"""Enflame GCU400 implementation of ``aten::_sparse_semi_structured_apply``.

The generic implementation reshapes the input tile into a rank-7 tensor to
split the eight in-tile axes (mg, a, b, c, kg, d, r).  The GCU lowering rejects
any tensor above rank 5 (``memref_ext.slice_start op rank should <=5``), so the
whole operator failed to compile on this backend.

This version keeps the tile in a rank-2 packed layout instead:

* the value tile is ``(BM, BK // 4, 4)`` -- one group of four consecutive
  columns per (row, group) pair, which is exactly the 2:4 group the op keeps;
* the per-element mask bit of that group is rebuilt as ``(BM, BK // 4)`` by
  selecting the mask record byte ``lane = 4 * a + 2 * d + b`` inside a rank-4
  ``(MG, 8, KG, 2)`` view of the compact mask tile, then taking the low or high
  nibble depending on the row parity ``c``;
* ``out1`` reuses the same bit tensor transposed, because ``out1[k, c]`` packs
  the four mask-selected *rows* of column ``k``.

Mask semantics (reverse-engineered from the native operator, identical to the
generic implementation)::

    p = m % 8 , q = k % 8
    lane   = 4 * (p // 4) + 2 * (q // 4) + ((p // 2) % 2)
    bitpos = (p % 2) * 4 + (q % 4)
    bit(m, k) = (thread_masks[m // 8, k // 8, lane] >> bitpos) & 1

and the two precedence-sensitive slots of every 2:4 group are

    slot0 = (b2 & b3) ? v2 : (b0 ? v0 : (b1 ? v1 : +0))
    slot1 = b3 ? v3 : (b2 ? v2 : ((b0 & b1) ? v1 : +0))

Selection is pure 16-bit data movement (bitcast to int16, integer bit tests), so
selected values stay bit-identical and unselected slots are bit-pattern +0.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_SUPPORTED_INPUT_DTYPES = (torch.float16, torch.bfloat16)
_UINT8 = torch.uint8

# Tile shape tiers: (rows, cols) per program.  Every value must keep the eight
# in-tile axes inside the rank-4/rank-3 views used by the kernel.
_BM = 32
_BK = 64
_LARGE_BM = 64
_LARGE_BK = 64


@triton.jit
def _split4(x):
    """Split a trailing axis of length 4 into four (R, G) tensors."""
    x = tl.reshape(x, (x.shape[0], x.shape[1], 2, 2))
    lo, hi = tl.split(x)  # r in {0, 2} / r in {1, 3}
    v0, v2 = tl.split(lo)
    v1, v3 = tl.split(hi)
    return v0, v1, v2, v3


@triton.jit
def _pack4(v0, v1, v2, v3, b0, b1, b2, b3):
    """Pick the two kept slots of a 2:4 group with the aten precedence rules."""
    z = tl.zeros_like(v0)
    s0 = tl.where(
        (b2 & b3) != 0,
        v2,
        tl.where(b0 != 0, v0, tl.where(b1 != 0, v1, z)),
    )
    s1 = tl.where(
        b3 != 0,
        v3,
        tl.where(b2 != 0, v2, tl.where((b0 & b1) != 0, v1, z)),
    )
    return s0, s1


@triton.jit
def _apply_kernel(
    in_ptr,
    mask_ptr,
    out0_ptr,
    out1_ptr,
    M,
    K,
    stride_im,
    stride_ik,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    KG: tl.constexpr = BK // 8
    M8 = M // 8
    K8 = K // 8

    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    off_m = pid_m * BM + tl.arange(0, BM)
    off_k = pid_k * BK + tl.arange(0, BK)
    m_ok = off_m < M
    k_ok = off_k < K

    # ---- input tile as raw 16-bit words ------------------------------------
    ptrs = in_ptr + off_m[:, None] * stride_im + off_k[None, :] * stride_ik
    t = tl.load(ptrs, mask=m_ok[:, None] & k_ok[None, :], other=0)
    t = t.to(tl.int16, bitcast=True)

    # ---- per (row, column-group) mask byte --------------------------------
    # thread_masks rows are packed (K/8) * 8-byte records after the storage
    # offset, so thread_masks.stride(0) is ignored.  A lane byte sits at
    # offset 4 * a + 2 * d + b inside its record, with
    #   a = (m % 8) // 4,  b = (m % 8) // 2 % 2,  d = (q // 4) = (k / 4) % 2
    # and the in-group position q % 4 only selects one bit of that byte.
    g_ax = tl.arange(0, BK // 4)
    p_ax = tl.arange(0, BM) % 8
    kg_of_g = pid_k * KG + (g_ax // 2)  # column record of group g
    maddr = (
        ((off_m // 8)[:, None] * K8 + kg_of_g[None, :]) * 8
        + 4 * (p_ax // 4)[:, None]
        + 2 * (g_ax % 2)[None, :]
        + ((p_ax // 2) % 2)[:, None]
    )
    byte2 = tl.load(
        mask_ptr + maddr,
        mask=m_ok[:, None] & (kg_of_g[None, :] < K8),
        other=0,
    ).to(tl.int32)

    # bitpos = 4 * (p % 2) + (q % 4): the row parity chooses the nibble, the
    # position inside the four-column group is the shift of that nibble.
    # The four bits are kept as plain rank-2 shifts: a rank-3 broadcast that is
    # reshaped/split afterwards is mis-lowered by the GCU backend, the extracted
    # mask bits then come out shuffled.
    nib = (byte2 >> (4 * (tl.arange(0, BM) % 2))[:, None]) & 0xF
    b0 = nib & 1
    b1 = (nib >> 1) & 1
    b2 = (nib >> 2) & 1
    b3 = (nib >> 3) & 1

    # ---- out0: the kept columns of every row ------------------------------
    v0, v1, v2, v3 = _split4(tl.reshape(t, (BM, BK // 4, 4)))
    s0, s1 = _pack4(v0, v1, v2, v3, b0, b1, b2, b3)
    o0 = tl.reshape(tl.join(s0, s1), (BM, BK // 2))

    o0_cols = pid_k * (BK // 2) + tl.arange(0, BK // 2)
    tl.store(
        out0_ptr + off_m[:, None] * (K // 2) + o0_cols[None, :],
        o0.to(out0_ptr.dtype.element_ty, bitcast=True),
        mask=m_ok[:, None] & (o0_cols[None, :] < K // 2),
    )

    # ---- out1: the kept rows of every column ------------------------------
    # out1 groups four *consecutive rows* of one column, so it is produced from
    # four strided rank-2 loads instead of a transposed tile: every derived
    # tensor then stays rank-2 and no reshape of a broadcast is involved (that
    # combination is what produced shuffled data on this backend).
    gg = pid_m * (BM // 4) + tl.arange(0, BM // 4)  # row groups of four
    o1_cols = gg
    kk = off_k[:, None]  # out1 rows
    base = kk * stride_ik
    g_ok = o1_cols[None, :] < M // 4

    o1_ok = k_ok[:, None] & g_ok
    w0 = tl.load(in_ptr + base + (4 * gg)[None, :] * stride_im, mask=o1_ok, other=0)
    w1 = tl.load(in_ptr + base + (4 * gg + 1)[None, :] * stride_im, mask=o1_ok, other=0)
    w2 = tl.load(in_ptr + base + (4 * gg + 2)[None, :] * stride_im, mask=o1_ok, other=0)
    w3 = tl.load(in_ptr + base + (4 * gg + 3)[None, :] * stride_im, mask=o1_ok, other=0)
    w0 = w0.to(tl.int16, bitcast=True)
    w1 = w1.to(tl.int16, bitcast=True)
    w2 = w2.to(tl.int16, bitcast=True)
    w3 = w3.to(tl.int16, bitcast=True)

    # The four rows of a group differ only in (b, c) = (r // 2, r % 2), so they
    # share the record pair (lane 4a + 2d + 0, lane 4a + 2d + 1): one rank-3
    # load of two adjacent bytes per (column, group) is enough, and splitting the
    # leading half keeps every following tensor rank 2.
    m1 = (gg // 2)[None, :, None]
    a1 = (gg % 2)[None, :, None]
    kg1 = (off_k // 8)[:, None, None]
    d1 = ((off_k % 8) // 4)[:, None, None]
    maddr1 = (m1 * K8 + kg1) * 8 + 4 * a1 + 2 * d1 + tl.arange(0, 2)[None, None, :]
    byte3 = tl.load(
        mask_ptr + maddr1,
        mask=(m1 < M8) & (kg1 < K8) & g_ok[:, :, None],
        other=0,
    ).to(tl.int32)
    byte_a, byte_b = tl.split(byte3)  # (BK, BM // 4), lanes b = 0 / 1

    qq = (off_k % 4)[:, None]
    q0 = (byte_a >> qq) & 1
    q1 = (byte_a >> (4 + qq)) & 1
    q2 = (byte_b >> qq) & 1
    q3 = (byte_b >> (4 + qq)) & 1

    t0, t1 = _pack4(w0, w1, w2, w3, q0, q1, q2, q3)

    o1_row = off_k[:, None] * (M // 2)
    tl.store(
        out1_ptr + o1_row + (2 * o1_cols)[None, :],
        t0.to(out1_ptr.dtype.element_ty, bitcast=True),
        mask=k_ok[:, None] & g_ok,
    )
    tl.store(
        out1_ptr + o1_row + (2 * o1_cols + 1)[None, :],
        t1.to(out1_ptr.dtype.element_ty, bitcast=True),
        mask=k_ok[:, None] & g_ok & ((2 * o1_cols + 1)[None, :] < M // 2),
    )


def _validate(input, thread_masks):
    if input.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise RuntimeError(
            "Unsupported dtype - only `float16` and `bfloat16` are supported "
            "currently"
        )
    # native reads size(1) before its own rank check -> IndexError for rank < 2
    if input.dim() < 2:
        input.size(1)
    if input.dim() != 2:
        raise RuntimeError("expected a rank-2 input tile")
    M, K = input.size()
    if thread_masks.dtype is not _UINT8:
        raise RuntimeError("thread_masks must be uint8")
    if thread_masks.shape != (4 * ((M + 31) // 32), 8 * ((K + 63) // 64), 8):
        raise RuntimeError("Wrong threads_masks shape")
    mstride = thread_masks.stride()
    if mstride[1] != 8 or mstride[2] != 1:
        raise RuntimeError("Wrong threads_masks strides")
    if M % 32 or K % 64 or M <= 0 or K <= 0:
        raise RuntimeError("Wrong alignment shape[1]")
    istride = input.stride()
    if istride[0] != 1 and istride[0] % 8:
        raise RuntimeError("Row-major input requires stride(0) % 8 == 0")


def _sparse_semi_structured_apply(input, thread_masks):

    logger.debug("GEMS_ENFLAME SPARSE_SEMI_STRUCTURED_APPLY")
    _validate(input, thread_masks)
    M, K = input.size()

    out0 = torch.empty((M, K // 2), dtype=input.dtype, device=input.device)
    out1 = torch.empty((K, M // 2), dtype=input.dtype, device=input.device)

    istride = input.stride()
    size = M * K
    # Tiles above 8K elements overflow the local buffer of this kernel (the
    # packed outputs then come out wrong), so the two tiers stay well below it.
    if size > 512 * 512:
        bm, bk = _LARGE_BM, _LARGE_BK
    else:
        bm, bk = _BM, _BK
    grid = (triton.cdiv(M, bm), triton.cdiv(K, bk))
    _apply_kernel[grid](
        input,
        thread_masks,
        out0,
        out1,
        M,
        K,
        istride[0],
        istride[1],
        BM=bm,
        BK=bk,
        num_warps=4,
    )
    return out0, out1
