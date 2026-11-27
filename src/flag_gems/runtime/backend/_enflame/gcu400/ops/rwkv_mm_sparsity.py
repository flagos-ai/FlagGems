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

"""gcu400 rwkv_mm_sparsity: out[n] = sum_k k[k] * V[k, n], Triton only.

Semantics follow the generic FlagGems implementation: gate rows whose ``k`` is
exactly zero are never read, so a NaN in a zeroed row of ``V`` contributes 0
instead of poisoning the reduction.

Because zero gate rows contribute nothing, the entire win of this operator is
skipping their ``V`` loads, and the pipeline therefore runs in three Triton
stages at every supported dtype:

1. ``_sparse_scan_kernel`` compacts the nonzero gate rows (stable order) into
   ``idx`` / ``kcomp`` with a single-program cumsum scan, zero-filling the tail
   slots so the gather stage never needs a row predicate on a full block.
2. ``_sparse_gemv_kernel`` walks those compacted rows with a persistent
   ``NPROG``-program grid and unmasked 8-row by ``C_POW2``-column tiles (a
   runtime row mask on a 2D tile load costs ~4x on this backend, so only the one
   partial block per grid pays it), writing per-program partial sums.
3. ``_sparse_combine_kernel`` folds the partials sequentially and rounds once.

Summation accuracy: every stage accumulates with a Kahan (compensated) update,
so the accumulated rounding is O(eps * |result|) instead of O(eps * sum|x_i|).
That matters because the suite compares against an fp64 CPU reference with
atol=1e-4 over a 16384-term reduction whose partial sums reach ~5e3; the
uncompensated order drifted to ~1e-4 on the largest outputs.  With
compensation the products' own rounding (~3e-6) is the whole error budget, for
16-bit operands the products are exact in fp32 and only the final store rounds.

Shapes outside the sparse pipeline's guard (very large K or C) take
``rwkv_mm_sparsity_kernel``, a tiled multiply-reduce that keeps the same
zero-gate masking.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SPARSE_NPROG = 48
_SPARSE_BR = 8
_SPARSE_MAX_K = 32768
_SPARSE_MAX_C_POW2 = 8192


@triton.jit
def rwkv_mm_sparsity_kernel(
    k_ptr,
    v_ptr,
    output_ptr,
    v_cols,
    k_size,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """output[n] = sum_k k[k] * v[k, n], one program per N tile.

    Dense fallback for shapes outside the sparse guard.  The ``k != 0`` row
    predicate reproduces the generic zero-gate masking, and the per-chunk
    result is folded with a compensated update.
    """
    pid_n = tl.program_id(axis=0)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < v_cols
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    comp = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for k0 in range(0, k_size, BLOCK_SIZE_K):
        offs_k = k0 + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < k_size
        k_block = tl.load(k_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        keep = mask_k & (k_block != 0.0)
        v_block = tl.load(
            v_ptr + offs_k[:, None] * v_cols + offs_n[None, :],
            mask=keep[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        chunk = tl.sum(k_block[:, None] * v_block, axis=0)
        corrected = chunk - comp
        total = acc + corrected
        comp = (total - acc) - corrected
        acc = total

    tl.store(
        output_ptr + offs_n,
        acc.to(output_ptr.dtype.element_ty),
        mask=mask_n,
    )


@libentry()
@triton.jit
def _sparse_scan_kernel(
    k_ptr,
    idx_ptr,
    kcomp_ptr,
    count_ptr,
    K,
    BLOCK: tl.constexpr,
    BR: tl.constexpr,
):
    """Single-program compaction of the nonzero gate rows.

    Writes idx[0..nnz) = original row indices (stable order), kcomp to match,
    zeros the tail slots of the last partial block plus the BR pad slots after
    K, and stores nnz.  Every slot the GEMV reads is written here or
    zero-initialized, and the scatter targets [0, nnz) never overlap the
    zero-fill range [nnz, padded).
    """
    offs = tl.arange(0, BLOCK)
    m = offs < K
    kv = tl.load(k_ptr + offs, mask=m, other=0.0)
    keep = (kv.to(tl.float32) != 0.0) & m
    keepi = keep.to(tl.int32)
    pos = tl.cumsum(keepi, axis=0) - keepi
    nnz = tl.sum(keepi, axis=0)
    padded = ((nnz + BR - 1) // BR) * BR
    tl.store(idx_ptr + pos, offs, mask=keep)
    tl.store(kcomp_ptr + pos, kv, mask=keep)
    zero = (offs >= nnz) & (offs < padded)
    tl.store(idx_ptr + offs, 0, mask=zero)
    tl.store(kcomp_ptr + offs, 0.0, mask=zero)
    pad = tl.arange(0, BR)
    tl.store(idx_ptr + K + pad, 0)
    tl.store(kcomp_ptr + K + pad, 0.0)
    tl.store(count_ptr, nnz)


@libentry()
@triton.jit(do_not_specialize=["C_", "NPROG"])
def _sparse_gemv_kernel(
    idx_ptr,
    kcomp_ptr,
    v_ptr,
    part_ptr,
    count_ptr,
    C_,
    NPROG,
    C_POW2: tl.constexpr,
    BR: tl.constexpr,
):
    """Grid-stride GEMV over the compacted nonzero rows.

    Full blocks take the unmasked row path; the single partial block of each
    program masks its tail lanes (those products are exact zeros through the
    zeroed kcomp, so no masked-load slow path is ever hot).  Per-block sums are
    folded into the running total with a compensated update.
    """
    pid = tl.program_id(0)
    offs_c = tl.arange(0, C_POW2)
    mask_c = offs_c < C_
    nnz = tl.load(count_ptr)
    nact = (nnz + BR - 1) // BR
    acc = tl.zeros((C_POW2,), dtype=tl.float32)
    comp = tl.zeros((C_POW2,), dtype=tl.float32)
    for b in tl.range(pid, nact, NPROG):
        lanes = b * BR + tl.arange(0, BR)
        rows = tl.load(idx_ptr + lanes)
        kv = tl.load(kcomp_ptr + lanes).to(tl.float32)
        is_full = (b + 1) * BR <= nnz
        if is_full:
            tile = tl.load(
                v_ptr + rows[:, None] * C_ + offs_c[None, :],
                mask=mask_c[None, :],
                other=0.0,
            )
        else:
            rvalid = lanes < nnz
            tile = tl.load(
                v_ptr + rows[:, None] * C_ + offs_c[None, :],
                mask=rvalid[:, None] & mask_c[None, :],
                other=0.0,
            )
        chunk = tl.sum(kv[:, None] * tile.to(tl.float32), axis=0)
        corrected = chunk - comp
        total = acc + corrected
        comp = (total - acc) - corrected
        acc = total
    tl.store(part_ptr + pid * C_ + offs_c, acc, mask=mask_c)


@libentry()
@triton.jit(do_not_specialize=["C_", "NPROG"])
def _sparse_combine_kernel(part_ptr, out_ptr, C_, NPROG, BLOCK: tl.constexpr):
    """Compensated sequential combine of the per-program partials."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C_
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    comp = tl.zeros((BLOCK,), dtype=tl.float32)
    for p in tl.range(0, NPROG):
        value = tl.load(part_ptr + p * C_ + offs, mask=mask, other=0.0)
        corrected = value - comp
        total = acc + corrected
        comp = (total - acc) - corrected
        acc = total
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def _sparse_gather_admissible(k, v):
    """Shape guard for the sparse pipeline (never reads device values)."""
    n = k.numel()
    c = v.size(1)
    return (
        n <= _SPARSE_MAX_K
        and triton.next_power_of_2(c) <= _SPARSE_MAX_C_POW2
        and n * c < (1 << 31)
    )


def _sparse_gather(k, v):
    """Compact the nonzero gate rows, GEMV them, combine, round once."""
    n = k.numel()
    c = v.size(1)
    idx = torch.empty(n + _SPARSE_BR, device=k.device, dtype=torch.int32)
    kcomp = torch.empty(n + _SPARSE_BR, device=k.device, dtype=k.dtype)
    count = torch.empty(1, device=k.device, dtype=torch.int32)
    _sparse_scan_kernel[(1,)](
        k,
        idx,
        kcomp,
        count,
        n,
        BLOCK=triton.next_power_of_2(n),
        BR=_SPARSE_BR,
        num_warps=8,
    )
    partial = torch.empty((_SPARSE_NPROG, c), device=k.device, dtype=torch.float32)
    _sparse_gemv_kernel[(_SPARSE_NPROG,)](
        idx,
        kcomp,
        v,
        partial,
        count,
        c,
        _SPARSE_NPROG,
        C_POW2=triton.next_power_of_2(c),
        BR=_SPARSE_BR,
        num_warps=4,
        num_stages=1,
    )
    out = torch.empty(c, device=k.device, dtype=k.dtype)
    _sparse_combine_kernel[(triton.cdiv(c, 1024),)](
        partial,
        out,
        c,
        _SPARSE_NPROG,
        BLOCK=1024,
        num_warps=4,
    )
    return out


def _dense_gemv(k, v):
    v_cols = v.size(1)
    k_size = k.size(0)
    output = torch.empty(v_cols, device=k.device, dtype=k.dtype)
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    grid = (triton.cdiv(v_cols, BLOCK_SIZE_N),)
    rwkv_mm_sparsity_kernel[grid](
        k,
        v,
        output,
        v_cols,
        k_size,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=4,
    )
    return output


def rwkv_mm_sparsity(k: torch.Tensor, v: torch.Tensor):
    logger.debug("GEMS_ENFLAME RWKV_MM_SPARSITY")
    assert k.dim() == 1 and v.dim() == 2
    assert k.size(0) == v.size(0)

    n = k.size(0)
    c = v.size(1)
    if n == 0:
        return torch.zeros(c, device=k.device, dtype=k.dtype)
    if c == 0:
        return torch.empty(0, device=k.device, dtype=k.dtype)

    k_c = k if k.is_contiguous() else k.contiguous()
    v_c = v if v.is_contiguous() else v.contiguous()

    if _sparse_gather_admissible(k_c, v_c):
        return _sparse_gather(k_c, v_c)
    return _dense_gemv(k_c, v_c)
