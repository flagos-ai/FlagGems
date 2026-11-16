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
"""Kunlunxin (XPU3 / P800) override for aten::cartesian_prod.

Root cause of the generic 2-D kernel being ~5000x off on XPU3
------------------------------------------------------------
``src/flag_gems/ops/cartesian_prod.py`` materialises the ``[M, N]`` output with a
``[BLOCK_ROWS, BLOCK_N]`` 2-D tile whose store row stride is the runtime ``N``,
and gathers each input column with a per-lane ``(row // PR[k]) % s_k`` division.
On XPU3 a 2-D tile store costs one short strided DMA *per row* and per-lane
`//`/`%` gathers scalarise: 1024x1024 / fp16 measures 166 ms vs torch's 33 us
(`harness/solution/cartesian_prod`).  The flat 1-D store is the only store
pattern that approaches copy bandwidth on this backend (see the vander /
split_copy dossiers), so the output must be written as one contiguous run.

Fast path: pack a whole output row into one wide integer lane
-------------------------------------------------------------
Every output row ``r`` is ``[inp_0[d_0(r)], ..., inp_{N-1}[d_{N-1}(r)]]`` with
``d_k(r) = (r // PR[k]) % s_k``.  When the row is exactly 4 or 8 bytes it fits
in a single ``int32`` / ``int64`` lane, so one lane writes one whole row: the
store becomes a plain contiguous block DMA of the word view, and each column is
read as an ``int16`` / ``int32`` value (bit-view, zero-extended) and shifted
into place.  This is the technique that lifts cartesian_prod from ~0.0002x to
~1.1x on the 2-tensor 2-byte case and ~0.5x on the 4-tensor 2-byte case.

Remaining shapes (row not word-aligned, e.g. 3 columns of 2 bytes = 6 bytes, or
4 columns of 4 bytes = 16 bytes) cannot be packed per row, so they use either a
per-row scalar kernel with contiguous read-only loads (2-byte, N==3) or a flat
per-element kernel with an ``int32`` store (4-byte).  Everything else — int /
bool dtypes, non-contiguous inputs, autograd (``requires_grad``), empty outputs,
error paths and N not covered above — is delegated wholesale to the generic
``flag_gems.ops.cartesian_prod.run`` (a real XPU Triton kernel, never CPU/ATen).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAX_TENSORS = 8
_INT32_SAFE = 1 << 30

# Tuned launch geometry (P800 / KL3, measured in harness/perf_ir/exp_*).
_N2_I16_BLOCK, _N2_I16_WARPS = 16384, 32   # 2 tensors, 2-byte -> int32 row word
_N2_I32_BLOCK, _N2_I32_WARPS = 32768, 4    # 2 tensors, 4-byte -> int64 row word
_N4_I16_BLOCK, _N4_I16_WARPS = 16384, 16   # 4 tensors, 2-byte -> int64 row word
_ROW_I16_BLOCK, _ROW_I16_WARPS = 4096, 8   # 3 tensors, 2-byte -> N int16 stores


def _suffix_products(sizes):
    n = len(sizes)
    prs = [1] * n
    acc = 1
    for k in range(n - 2, -1, -1):
        acc *= sizes[k + 1]
        prs[k] = acc
    return prs


# --------------------------------------------------------------------------
# N == 2, 2-byte element (fp16 / bf16): one int32 lane == one output row.
# --------------------------------------------------------------------------
@triton.jit
def _cp_n2_i16(out_ptr, a_ptr, b_ptr, M, S1: tl.constexpr,
               BLOCK: tl.constexpr, NEED_MASK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.load(a_ptr + (r // S1)).to(tl.uint16, bitcast=True).to(tl.uint32)
    hi = tl.load(b_ptr + (r % S1)).to(tl.uint16, bitcast=True).to(tl.uint32)
    w = lo | (hi << 16)
    if NEED_MASK:
        tl.store(out_ptr + r, w, mask=r < M)
    else:
        tl.store(out_ptr + r, w)


# --------------------------------------------------------------------------
# N == 2, 4-byte element (fp32): one int64 lane == one output row.
# --------------------------------------------------------------------------
@triton.jit
def _cp_n2_i32(out_ptr, a_ptr, b_ptr, M, S1: tl.constexpr,
               BLOCK: tl.constexpr, NEED_MASK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.load(a_ptr + (r // S1)).to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = tl.load(b_ptr + (r % S1)).to(tl.uint32, bitcast=True).to(tl.uint64)
    w = lo | (hi << 32)
    if NEED_MASK:
        tl.store(out_ptr + r, w, mask=r < M)
    else:
        tl.store(out_ptr + r, w)


# --------------------------------------------------------------------------
# N == 4, 2-byte element (fp16 / bf16): one int64 lane == one output row.
# --------------------------------------------------------------------------
@triton.jit
def _cp_n4_i16(out_ptr, p0, p1, p2, p3,
               s0: tl.constexpr, s1: tl.constexpr, s2: tl.constexpr, s3: tl.constexpr,
               pr0: tl.constexpr, pr1: tl.constexpr, pr2: tl.constexpr, pr3: tl.constexpr,
               M, BLOCK: tl.constexpr, NEED_MASK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v0 = tl.load(p0 + ((r // pr0) % s0)).to(tl.uint16, bitcast=True).to(tl.uint64)
    v1 = tl.load(p1 + ((r // pr1) % s1)).to(tl.uint16, bitcast=True).to(tl.uint64)
    v2 = tl.load(p2 + ((r // pr2) % s2)).to(tl.uint16, bitcast=True).to(tl.uint64)
    v3 = tl.load(p3 + ((r // pr3) % s3)).to(tl.uint16, bitcast=True).to(tl.uint64)
    w = v0 | (v1 << 16) | (v2 << 32) | (v3 << 48)
    if NEED_MASK:
        tl.store(out_ptr + r, w, mask=r < M)
    else:
        tl.store(out_ptr + r, w)


# --------------------------------------------------------------------------
# 2-byte element, arbitrary N (used for N == 3): lane == row, one int16 store
# per column.  All loads are affine in the row index (cheap broadcast reads);
# the store is per-column strided but N is tiny.
# --------------------------------------------------------------------------
@triton.jit
def _cp_row_i16(out_ptr, p0, p1, p2, p3, p4, p5, p6, p7,
                s0: tl.constexpr, s1: tl.constexpr, s2: tl.constexpr, s3: tl.constexpr,
                s4: tl.constexpr, s5: tl.constexpr, s6: tl.constexpr, s7: tl.constexpr,
                pr0: tl.constexpr, pr1: tl.constexpr, pr2: tl.constexpr, pr3: tl.constexpr,
                pr4: tl.constexpr, pr5: tl.constexpr, pr6: tl.constexpr, pr7: tl.constexpr,
                N: tl.constexpr, M, BLOCK: tl.constexpr, NEED_MASK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ptrs = (p0, p1, p2, p3, p4, p5, p6, p7)
    sizes = (s0, s1, s2, s3, s4, s5, s6, s7)
    prs = (pr0, pr1, pr2, pr3, pr4, pr5, pr6, pr7)
    if NEED_MASK:
        m = r < M
        for k in tl.static_range(N):
            v = tl.load(ptrs[k] + ((r // prs[k]) % sizes[k]))
            tl.store(out_ptr + N * r + k, v, mask=m)
    else:
        for k in tl.static_range(N):
            v = tl.load(ptrs[k] + ((r // prs[k]) % sizes[k]))
            tl.store(out_ptr + N * r + k, v)


# --------------------------------------------------------------------------
# 4-byte element, arbitrary N (used for N >= 3): lane == flat int32 element,
# contiguous int32 store.  Selects one of N candidate int32 values per lane.
# --------------------------------------------------------------------------
@triton.jit
def _cp_flat_i32(out_ptr, p0, p1, p2, p3,
                 s0: tl.constexpr, s1: tl.constexpr, s2: tl.constexpr, s3: tl.constexpr,
                 pr0: tl.constexpr, pr1: tl.constexpr, pr2: tl.constexpr, pr3: tl.constexpr,
                 N: tl.constexpr, NUMEL: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = e // N
    c = e - N * r
    a = tl.zeros([BLOCK], dtype=tl.int32)
    ptrs = (p0, p1, p2, p3)
    sizes = (s0, s1, s2, s3)
    prs = (pr0, pr1, pr2, pr3)
    for k in tl.static_range(N):
        v = tl.load(ptrs[k] + ((r // prs[k]) % sizes[k])).to(tl.uint32, bitcast=True)
        a = tl.where(c == k, v, a)
    tl.store(out_ptr + e, a, mask=e < NUMEL)


def _normalize(args, kwargs):
    """Only resolve the argument forms we can fast-path; anything unusual
    returns None so the generic implementation handles it (and its errors)."""
    if kwargs:
        return None
    if (
        len(args) == 1
        and isinstance(args[0], (list, tuple))
        and all(isinstance(x, torch.Tensor) for x in args[0])
    ):
        return list(args[0])
    if all(isinstance(x, torch.Tensor) for x in args):
        return list(args)
    return None


def _fast_eligible(tensors):
    n = len(tensors)
    if n < 2 or n > _MAX_TENSORS:
        return False
    for t in tensors:
        if t.dim() != 1:
            return False
        if not t.is_contiguous():
            return False
        if t.requires_grad:
            return False
        if t.dtype != tensors[0].dtype:
            return False
    dt = tensors[0].dtype
    if dt not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    m = 1
    for t in tensors:
        m *= t.shape[0]
    if m <= 0 or m * n > _INT32_SAFE:
        return False
    # The packed/row kernels strength-reduce the per-column ``(r // PR) % s``
    # arithmetic to shifts/masks only when every input length is a power of two;
    # non-power-of-two lengths are handled by the generic kernel (correct, and
    # not exercised by the benchmark matrix).  This keeps the fast path free of
    # the N==4 non-pow2 ``tl.where`` miscompile observed on XPU3.
    for t in tensors:
        s = t.shape[0]
        if s & (s - 1):
            return False
    return True


def _launch_fast(tensors):
    n = len(tensors)
    dt = tensors[0].dtype
    sizes = tuple(t.shape[0] for t in tensors)
    prs = tuple(_suffix_products(sizes))
    m = 1
    for s in sizes:
        m *= s
    out = torch.empty((m, n), dtype=dt, device=tensors[0].device)
    pad = _MAX_TENSORS - n
    ptrs = tuple(tensors) + (tensors[0],) * pad
    szs = sizes + (1,) * pad
    prs2 = prs + (1,) * pad
    esize = tensors[0].element_size()

    if esize == 2 and n == 2:
        blk, warps = _N2_I16_BLOCK, _N2_I16_WARPS
        masked = (m % blk) != 0
        _cp_n2_i16[(triton.cdiv(m, blk),)](
            out.view(torch.int32), tensors[0].view(torch.int16),
            tensors[1].view(torch.int16), m, S1=sizes[1], BLOCK=blk,
            NEED_MASK=masked, num_warps=warps,
        )
        return out
    if esize == 4 and n == 2:
        blk, warps = _N2_I32_BLOCK, _N2_I32_WARPS
        masked = (m % blk) != 0
        _cp_n2_i32[(triton.cdiv(m, blk),)](
            out.view(torch.int64), tensors[0].view(torch.int32),
            tensors[1].view(torch.int32), m, S1=sizes[1], BLOCK=blk,
            NEED_MASK=masked, num_warps=warps,
        )
        return out
    if esize == 2 and n == 4:
        blk, warps = _N4_I16_BLOCK, _N4_I16_WARPS
        masked = (m % blk) != 0
        _cp_n4_i16[(triton.cdiv(m, blk),)](
            out.view(torch.int64), *[t.view(torch.int16) for t in tensors],
            *sizes, *prs, m, BLOCK=blk, NEED_MASK=masked, num_warps=warps,
        )
        return out
    if esize == 2 and n == 3:
        blk, warps = _ROW_I16_BLOCK, _ROW_I16_WARPS
        masked = (m % blk) != 0
        _cp_row_i16[(triton.cdiv(m, blk),)](
            out.view(torch.int16), *[t.view(torch.int16) for t in ptrs],
            *szs, *prs2, N=n, M=m, BLOCK=blk, NEED_MASK=masked, num_warps=warps,
        )
        return out
    if esize == 4 and 3 <= n <= 4:
        numel = m * n
        if numel <= (1 << 20):
            blk, warps = 4096, 8
        else:
            blk, warps = 16384, 16
        _cp_flat_i32[(triton.cdiv(numel, blk),)](
            out.view(torch.int32), *[t.view(torch.int32) for t in ptrs[:4]],
            *szs[:4], *prs2[:4], N=n, NUMEL=numel, BLOCK=blk, num_warps=warps,
        )
        return out
    return None


def cartesian_prod(*args, **kwargs):
    """Kunlunxin override for aten::cartesian_prod."""
    logger.debug("GEMS_KUNLUNXIN CARTESIAN_PROD")
    tensors = _normalize(args, kwargs)
    if tensors is not None and _fast_eligible(tensors):
        try:
            out = _launch_fast(tensors)
        except Exception:  # noqa: BLE001 - any codegen issue -> generic path
            out = None
        if out is not None:
            return out
    # Correct/robust fallback for every remaining shape, dtype and autograd /
    # edge-case semantic: the generic XPU Triton implementation.
    from flag_gems.ops.cartesian_prod import run as _generic_run

    return _generic_run(*args, **kwargs)
