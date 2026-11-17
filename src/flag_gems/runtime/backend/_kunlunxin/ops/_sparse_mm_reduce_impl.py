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
"""Kunlunxin XPU3 override for ``_sparse_mm_reduce_impl``.

Same root cause as the sspaddmm override: the row kernel carries a 1-D
accumulator through a dynamic loop over the CSR row's stored entries, and
``TritonXPUUnrollControl`` only accepts a column tile of **<=64 lanes or exactly
1024 lanes**; 128/256/512 exhaust ``uni_sram`` at compile time.  The generic
implementation uses ``BLOCK_N = min(next_pow2(N), 256)``, which fails to
compile for N > 64 (e.g. the (256, 32, 512) shape in the test matrix).  We
therefore pick ``next_pow2(N)`` for N<=64 and a masked 1024-wide tile otherwise
(measured ~16x faster than a 64-lane tile on N=1024).

When the chosen tile divides ``N`` exactly (``N % BLOCK_N == 0``) every lane of
every program is in range, so the ``rmask`` on the ``mat2`` block load is
provably redundant; ``NEED_MASK=False`` drops it and lets the load take the
plain contiguous block-DMA path (-7 % on 1024^2/4096^2, bit-identical result).
Shapes where the tile does not divide ``N`` keep the fully masked path.  The
output store keeps its (then statically all-true) mask on both paths: removing
both predicates makes the XPU3 compiler miscompile the in-loop block DMA.

The host pipeline (CSR conversion, layout/validation) mirrors the generic
implementation; the reduction itself is the Triton kernel.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._sparse_mm_reduce_impl import (
    _REDUCE_AMAX,
    _REDUCE_AMIN,
    _REDUCE_CODES,
    _REDUCE_MEAN,
    _SPARSE_MM_REDUCE_IMPL_DTYPES,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# XPU3-safe loop-carried 1-D accumulator widths: <=64 lanes, or exactly 1024.
_MAX_BLOCK_N = 64
_WIDE_BLOCK_N = 1024


@libentry()
@triton.jit
def _sparse_mm_reduce_impl_kernel(
    crow_ptr,
    col_ptr,
    val_ptr,
    mat2_ptr,
    out_ptr,
    M,
    N,
    stride_mat2_k,
    stride_mat2_n,
    stride_out_m,
    stride_out_n,
    IS_FP64: tl.constexpr,
    REDUCE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Backend copy of the generic row kernel plus the ``NEED_MASK`` switch.

    ``NEED_MASK=False`` is only legal when ``N % BLOCK_N == 0`` (the host
    guarantees it): then ``rn < N`` is statically all-true and the masked
    memory path is pure overhead.
    """
    row = ext.program_id(0)
    rn = ext.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rn < N
    row64 = row.to(tl.int64)
    col_off = (rn * stride_mat2_n).to(tl.int64)

    acc_dtype = tl.float64 if IS_FP64 else tl.float32
    acc = tl.zeros((BLOCK_N,), dtype=acc_dtype)

    start = tl.load(crow_ptr + row64).to(tl.int64)
    end = tl.load(crow_ptr + row64 + 1).to(tl.int64)

    if REDUCE == _REDUCE_AMAX or REDUCE == _REDUCE_AMIN:
        # Fold with tl.where so the first stored product seeds acc without a
        # dynamic branch; max/min over an empty row collapses to zero below.
        for e in range(start, end):
            c = tl.load(col_ptr + e).to(tl.int64)
            v = tl.load(val_ptr + e).to(acc_dtype)
            if NEED_MASK:
                b = tl.load(
                    mat2_ptr + c * stride_mat2_k + col_off, mask=rmask, other=0.0
                ).to(acc_dtype)
            else:
                b = tl.load(mat2_ptr + c * stride_mat2_k + col_off).to(acc_dtype)
            p = b * v
            if REDUCE == _REDUCE_AMAX:
                acc = tl.where(e == start, p, tl.maximum(acc, p))
            else:
                acc = tl.where(e == start, p, tl.minimum(acc, p))
        acc = tl.where(start < end, acc, 0.0)
    else:
        for e in range(start, end):
            c = tl.load(col_ptr + e).to(tl.int64)
            v = tl.load(val_ptr + e).to(acc_dtype)
            if NEED_MASK:
                b = tl.load(
                    mat2_ptr + c * stride_mat2_k + col_off, mask=rmask, other=0.0
                ).to(acc_dtype)
            else:
                b = tl.load(mat2_ptr + c * stride_mat2_k + col_off).to(acc_dtype)
            acc += b * v
        if REDUCE == _REDUCE_MEAN:
            nnz = (end - start).to(acc_dtype)
            acc = tl.where(nnz > 0, acc / nnz, 0.0)

    cos = (row64 * stride_out_m + (rn * stride_out_n).to(tl.int64)).to(tl.int64)
    # The store keeps its mask unconditionally: on XPU3 dropping *both* the
    # load and the store predicate removes `rmask` entirely and miscompiles the
    # in-loop block DMA (measured: 50 % wrong lanes / non-deterministic at
    # M=256,K=512,N=32).  Dropping only the load predicate is exact.
    tl.store(out_ptr + cos, acc.to(out_ptr.dtype.element_ty), mask=rmask)


def _pick_block_n(N, dtype):
    npow2 = triton.next_power_of_2(N) if N > 0 else 1
    # fp16 cannot lower a 1024-lane tile for this kernel (TritonXPUUnrollControl
    # / uni_sram); it stays on the <=64-lane path.
    if npow2 <= _MAX_BLOCK_N:
        return npow2
    if dtype == torch.float16:
        return _MAX_BLOCK_N  # 64-lane grid-tiled fallback
    return _WIDE_BLOCK_N


def _to_csr(sparse):
    """Backend-local CSR conversion.

    The native XPU ``csc.to_sparse_csr()`` keeps the correct ``crow``/
    ``col_indices`` but permutes the *values* incorrectly (measured: value
    diff ~2.4 on a 5x5 fp32 matrix), while ``csc.to_sparse()`` (CSC -> COO) and
    ``coo.to_sparse_csr()`` are both exact.  Route CSC through COO.
    """
    if sparse.layout == torch.sparse_csr:
        return sparse
    if sparse.layout == torch.sparse_csc:
        return sparse.to_sparse().to_sparse_csr()
    if sparse.layout in (
        torch.sparse_coo,
        torch.sparse_bsr,
        torch.sparse_bsc,
    ):
        return sparse.to_sparse_csr()
    raise RuntimeError(
        f"_sparse_mm_reduce_impl: Expected self to have a sparse layout, but "
        f"got {sparse.layout}"
    )


def _sparse_mm_reduce_impl(self, other, reduce="sum"):
    """Backend of ``aten::_sparse_mm.reduce``: SpMM plus a per-row reduction."""
    logger.debug("GEMS_SPARSE_MM_REDUCE_IMPL")

    if self.layout == torch.strided:
        raise NotImplementedError(
            f"Could not run '_sparse_mm_reduce_impl' with arguments from the "
            f"'{self.layout}' backend. This could be because the operator "
            f"doesn't exist for this backend, or was omitted during the "
            f"selective/custom build process"
        )
    if other.layout != torch.strided:
        raise RuntimeError(
            f"_sparse_mm_reduce_impl: Expected other to have strided layout, "
            f"but got {other.layout}"
        )
    if self.dim() != 2:
        raise RuntimeError(
            f"sparse_mm_reduce: expected self to be a 2-D tensor, got "
            f"{self.dim()}-D tensor."
        )
    if self.dense_dim() != 0:
        raise RuntimeError("sparse_mm_reduce: expected non-hybrid self tensor.")
    if other.dim() != 2 or other.shape[0] != self.shape[1]:
        raise RuntimeError(
            f"Expected a tensor of dimension 2 and tensor.size[0] == "
            f"{self.shape[1]} but got: dimension {other.dim()} and "
            f"tensor.size[0] = {other.shape[0]}"
        )

    if reduce not in _REDUCE_CODES:
        raise RuntimeError(
            f"reduce argument must be either sum, prod, mean, amax or amin, "
            f"got {reduce}"
        )
    if reduce == "prod":
        raise RuntimeError(
            "sparse_mm_reduce: reduce type of prod has not been enabled."
        )

    csr = _to_csr(self)
    if csr.layout != torch.sparse_csr:
        raise RuntimeError(
            f"sparse_mm_reduce: expect self to be SparseCsr, got {csr.layout}"
        )

    if self.dtype not in _SPARSE_MM_REDUCE_IMPL_DTYPES:
        raise RuntimeError(f'"spmm_reduce_kernel" not implemented for {self.dtype!r}')
    if self.dtype != other.dtype:
        raise RuntimeError(
            f"Expected tensor for argument #1 'other' to have scalar type "
            f"{self.dtype} but got {other.dtype}"
        )

    M, K = csr.shape
    N = other.shape[1]

    out = torch.empty((M, N), dtype=other.dtype, device=other.device)
    arg_out = torch.empty((0,), dtype=torch.int64, device=other.device)
    if M == 0 or N == 0:
        return out, arg_out

    crow = csr.crow_indices()
    col = csr.col_indices()
    val = csr.values()

    # Backend-local tile choice: see module docstring.
    block_n = _pick_block_n(N, other.dtype)
    # All lanes are in range exactly when the tile divides N; then the mask on
    # the mat2 block load / output store is provably redundant.
    need_mask = (N % block_n) != 0
    grid = (M, triton.cdiv(N, block_n))
    with torch_device_fn.device(other.device):
        _sparse_mm_reduce_impl_kernel[grid](
            crow,
            col,
            val,
            other,
            out,
            M,
            N,
            other.stride(0),
            other.stride(1),
            out.stride(0),
            out.stride(1),
            IS_FP64=other.dtype == torch.float64,
            REDUCE=_REDUCE_CODES[reduce],
            BLOCK_N=block_n,
            NEED_MASK=need_mask,
            num_warps=4,
        )
    return out, arg_out
