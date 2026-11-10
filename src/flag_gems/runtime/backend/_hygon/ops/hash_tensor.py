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

"""Gems hash_tensor: XOR-reduce of 64-bit widened element bit patterns.

Oracle semantics (verified on target against aten::hash_tensor):
  * float16 / bfloat16 / float32 -> numeric conversion to float64, then bitcast
  * float64 / int64              -> bitcast of the native 64-bit pattern
  * int8 / int16 / int32         -> sign-extend to int64, then bitcast
  * uint8 / bool                 -> zero-extend to uint64
XOR reduction is over the selected dims (empty/None = all dims); output is a
freshly allocated contiguous uint64 tensor.

Structure: each program owns BLOCK_M output elements, streams the reduction axes
into a register-resident XOR accumulator and performs exactly one cross-lane
reduction at the end (no barrier inside the streaming loop). uint8/bool
accumulate at 32-bit width because widen-to-uint64 in the loop crashes the
target's aillvm backend; widening once at the end is bit-identical.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_DTYPE_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
    torch.int64: 4,
    torch.int32: 5,
    torch.int16: 6,
    torch.int8: 7,
    torch.uint8: 8,
    torch.bool: 9,
}
# 8-bit inputs: accumulate in uint32 (zero extension commutes with XOR).
_ACC32 = {torch.uint8, torch.bool}


@triton.jit
def _xor_combine(a, b):
    return a ^ b


@triton.jit
def _widen(v, DTYPE_CODE: tl.constexpr):
    # Widen to the 64-bit pattern the oracle hashes.
    if DTYPE_CODE < 3:
        u = v.to(tl.float64).to(tl.uint64, bitcast=True)
    elif DTYPE_CODE < 5:
        u = v.to(tl.uint64, bitcast=True)
    else:
        u = v.to(tl.int64).to(tl.uint64, bitcast=True)
    return u


@triton.jit
def _row_hash_kernel(
    x_ptr,
    out_ptr,
    M,
    base_stride,
    red_stride,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DTYPE_CODE: tl.constexpr,
    ACC32: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """One affine output stride and one reduction stride: base = o*base_stride,
    reduce offset = k*red_stride."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    base = rows.to(tl.int64) * base_stride

    if ACC32:
        acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.uint32)
    else:
        acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.uint64)

    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        offs = base[:, None] + cols[None, :].to(tl.int64) * red_stride
        if EVEN_K:
            if EVEN_M:
                v = tl.load(x_ptr + offs)
            else:
                v = tl.load(x_ptr + offs, mask=(rows < M)[:, None], other=0)
        else:
            if EVEN_M:
                v = tl.load(x_ptr + offs, mask=(cols < K)[None, :], other=0)
            else:
                v = tl.load(
                    x_ptr + offs,
                    mask=(rows < M)[:, None] & (cols < K)[None, :],
                    other=0,
                )
        if ACC32:
            acc ^= v.to(tl.uint32)
        else:
            acc ^= _widen(v, DTYPE_CODE)

    if ACC32:
        red = tl.reduce(acc, 1, _xor_combine).to(tl.uint64)
    else:
        red = tl.reduce(acc, 1, _xor_combine)
    if EVEN_M:
        tl.store(out_ptr + rows, red)
    else:
        tl.store(out_ptr + rows, red, mask=rows < M)


@triton.jit
def _general_hash_kernel(
    x_ptr,
    out_ptr,
    M,
    oe0,
    os0,
    oe1,
    os1,
    oe2,
    os2,
    oe3,
    os3,
    re0,
    rs0,
    re1,
    rs1,
    re2,
    rs2,
    re3,
    rs3,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DTYPE_CODE: tl.constexpr,
    ACC32: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """Generic path: up to 4 output levels and up to 4 reduction levels.
    Unused levels are passed with extent 1 and stride 0."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)

    t = rows.to(tl.int64)
    base = (t % oe3) * os3
    t = t // oe3
    base += (t % oe2) * os2
    t = t // oe2
    base += (t % oe1) * os1
    t = t // oe1
    base += t * os0

    KR = re0 * re1 * re2 * re3
    if ACC32:
        acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.uint32)
    else:
        acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.uint64)

    for k0 in range(0, KR, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        t2 = kk.to(tl.int64)
        o = (t2 % re3) * rs3
        t2 = t2 // re3
        o += (t2 % re2) * rs2
        t2 = t2 // re2
        o += (t2 % re1) * rs1
        t2 = t2 // re1
        o += t2 * rs0
        offs = base[:, None] + o[None, :]
        if EVEN_K:
            if EVEN_M:
                v = tl.load(x_ptr + offs)
            else:
                v = tl.load(x_ptr + offs, mask=(rows < M)[:, None], other=0)
        else:
            if EVEN_M:
                v = tl.load(x_ptr + offs, mask=(kk < KR)[None, :], other=0)
            else:
                v = tl.load(
                    x_ptr + offs,
                    mask=(rows < M)[:, None] & (kk < KR)[None, :],
                    other=0,
                )
        if ACC32:
            acc ^= v.to(tl.uint32)
        else:
            acc ^= _widen(v, DTYPE_CODE)

    if ACC32:
        red = tl.reduce(acc, 1, _xor_combine).to(tl.uint64)
    else:
        red = tl.reduce(acc, 1, _xor_combine)
    if EVEN_M:
        tl.store(out_ptr + rows, red)
    else:
        tl.store(out_ptr + rows, red, mask=rows < M)


def _analysis(x, dim, keepdim):
    """Structure-only analysis: reduce/kept (extent, stride) levels."""
    nd = x.dim()
    if dim is None:
        red = list(range(nd))
    elif isinstance(dim, int):
        red = [dim]
    else:
        red = list(dim)
        if not red:
            red = list(range(nd))
    if nd == 0:
        red = []
    else:
        red = sorted({d % nd for d in red})
    red_set = set(red)
    shape = x.shape
    stride = x.stride()

    out_shape = []
    kept = []
    reds = []
    K = 1
    for i in range(nd):
        if i in red_set:
            reds.append((shape[i], stride[i]))
            K *= shape[i]
            if keepdim:
                out_shape.append(1)
        else:
            out_shape.append(shape[i])
            kept.append((shape[i], stride[i]))

    M = 1
    for e in out_shape:
        M *= e

    # Merge nested kept levels (original order preserves the output linearization).
    mkept = []
    for e, s in kept:
        if e == 1:
            continue
        if mkept and mkept[-1][1] == s * e:
            mkept[-1] = (mkept[-1][0] * e, s)
        else:
            mkept.append((e, s))

    # Merge nested reduction levels (order is irrelevant for XOR).
    mred = []
    for e, s in sorted(reds, key=lambda p: -p[1]):
        if e == 1:
            continue
        if mred and mred[-1][1] == s * e:
            mred[-1] = (mred[-1][0] * e, s)
        else:
            mred.append((e, s))

    return out_shape, M, K, mkept, mred


def _pick_blocks(M, K):
    bk = min(1024, triton.next_power_of_2(K)) if K > 1 else 1
    bm = max(1, 1024 // bk)
    mp = triton.next_power_of_2(M) if M > 0 else 1
    if bm > mp:
        bm = mp
    tile = bm * bk
    nw = 4 if tile >= 256 else (2 if tile >= 64 else 1)
    return bm, bk, nw


def hash_tensor(self, dim=None, keepdim=False, mode=0):
    logger.debug("GEMS_HYGON HASH_TENSOR")
    x = self
    if mode != 0:
        raise RuntimeError("Unknown hash_tensor mode: %d" % mode)

    dtype = x.dtype
    code = _DTYPE_CODE[dtype]
    acc32 = dtype in _ACC32
    out_shape, M, K, mkept, mred = _analysis(x, dim, keepdim)
    device = x.device

    if M == 0:
        return torch.empty(out_shape, dtype=torch.uint64, device=device)

    out = torch.empty(M, dtype=torch.uint64, device=device)
    xv = x.view(torch.uint8) if code == 9 else x
    bm, bk, nw = _pick_blocks(M, K)
    even_m = (M % bm) == 0
    even_k = (K % bk) == 0

    if len(mkept) <= 1 and len(mred) <= 1:
        base_stride = mkept[0][1] if mkept else 0
        red_stride = mred[0][1] if mred else 0
        _row_hash_kernel[(triton.cdiv(M, bm),)](
            xv,
            out,
            M,
            base_stride,
            red_stride,
            K,
            BLOCK_M=bm,
            BLOCK_K=bk,
            DTYPE_CODE=code,
            ACC32=acc32,
            EVEN_K=even_k,
            EVEN_M=even_m,
            num_warps=nw,
        )
    else:
        kd = mkept + [(1, 0)] * (4 - len(mkept))
        rd = mred + [(1, 0)] * (4 - len(mred))
        (oe0, os0), (oe1, os1), (oe2, os2), (oe3, os3) = kd
        (re0, rs0), (re1, rs1), (re2, rs2), (re3, rs3) = rd
        _general_hash_kernel[(triton.cdiv(M, bm),)](
            xv,
            out,
            M,
            oe0,
            os0,
            oe1,
            os1,
            oe2,
            os2,
            oe3,
            os3,
            re0,
            rs0,
            re1,
            rs1,
            re2,
            rs2,
            re3,
            rs3,
            BLOCK_M=bm,
            BLOCK_K=bk,
            DTYPE_CODE=code,
            ACC32=acc32,
            EVEN_K=even_k,
            EVEN_M=even_m,
            num_warps=nw,
        )

    if len(out_shape) == 0:
        return out.view(())
    return out.view(out_shape)
