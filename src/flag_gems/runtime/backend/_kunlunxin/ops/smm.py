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
"""Kunlunxin XPU3 override for ``smm`` (sparse COO x dense matmul).

The generic ``src/flag_gems/ops/smm.py`` kernel is **not XPU3-legal** and does
not merely fail to compile -- the lowering error escalates to a device XID
(measured: ``status=299`` / ``KL_XID_KERNEL_EXCEPTION``, axi wresp error):

* ``B + k[:, None] * stride_bk + n_off[None, :]`` is a 2-D data-dependent
  per-row gather -> ``cannot lower per-line gather``;
* ``tl.load(col_indices + e)`` loads an **int64** index buffer into registers
  -> ``Unsupported dma dst data type`` -> ``LLVM ERROR`` -> XID.

This backend implementation avoids both patterns by reusing the scalar-``k``
streaming row kernel from ``sspaddmm`` (``_sspaddmm_row_kernel``): the row's
COO entries are walked one at a time with a scalar index ``k``, so
``mat2[k, :]`` is a plain contiguous row load (no per-line gather, no int64
DMA load, no atomics -- deterministic).  Row segmentation reuses
``unique_consecutive`` + ``cumsum`` on the coalesced row index, exactly like
the sspaddmm host path (that path is verified working on this backend), and
the COO output indices are assembled on the host with ``repeat_interleave`` /
``arange``.  The column tile follows the same XPU3 rule as the other sparse row
kernels: ``next_pow2(N)`` when <=64 lanes, else a masked 1024-wide tile (128 /
256 / 512 lanes trip ``TritonXPUUnrollControl`` -> ``uni_sram`` exhaustion).
"""

import logging

import torch
import triton

from flag_gems.ops.sspaddmm import _sspaddmm_row_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# XPU3-safe loop-carried 1-D accumulator widths: <=64 lanes, or exactly 1024.
_MAX_BLOCK_N = 64
_WIDE_BLOCK_N = 1024


def _pick_block_n(N, dtype):
    npow2 = triton.next_power_of_2(N) if N > 0 else 1
    # fp16 cannot lower a 1024-lane tile for this kernel (TritonXPUUnrollControl
    # / uni_sram); it stays on the <=64-lane path.
    if npow2 <= _MAX_BLOCK_N:
        return npow2
    if dtype == torch.float16:
        return _MAX_BLOCK_N  # 64-lane grid-tiled fallback
    return _WIDE_BLOCK_N


def smm(self, mat):
    """Sparse (COO) matrix ``self`` multiplied by dense matrix ``mat``."""
    logger.debug("GEMS_SMM")

    if not self.is_sparse:
        raise RuntimeError("tensor.sspaddmm(...) can only be called on sparse tensors")
    if self.ndim != 2:
        raise RuntimeError(
            f"sspaddmm: Argument #2: matrices expected, got {self.ndim}D tensor"
        )
    if mat.is_sparse:
        raise RuntimeError(
            "Cannot access data pointer of Tensor that doesn't have storage"
        )
    if mat.ndim != 2:
        raise RuntimeError(
            f"sspaddmm: Argument #3: matrices expected, got {mat.ndim}D tensor"
        )

    M, K = self.shape
    K2, N = mat.shape
    if K != K2:
        raise RuntimeError(f"sspaddmm: Argument #3: Expected dim 0 size {K}, got {K2}")

    if mat.device != self.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but got mat2 is on "
            f"{mat.device}, different from other tensors on {self.device}"
        )

    # torch.smm is fp32-only on the CUDA backend; mirror that restriction.
    if self.dtype != torch.float32 or mat.dtype != torch.float32:
        raise RuntimeError(
            "GEMS smm is only implemented for float32 inputs, matching torch.smm"
        )

    if not self.is_coalesced():
        self = self.coalesce()

    nnz = self._nnz()
    if nnz == 0 or N == 0 or M == 0:
        out_indices = torch.empty((2, 0), dtype=torch.int64, device=self.device)
        out_values = torch.empty((0,), dtype=torch.float32, device=self.device)
        return torch.sparse_coo_tensor(
            out_indices, out_values, size=(M, N), device=self.device
        )

    indices = self._indices()
    rows = indices[0]
    cols = indices[1].contiguous()
    vals = self._values().contiguous()

    # Group the coalesced nonzeros into per-row segments.
    active_rows, counts = torch.unique_consecutive(rows, return_counts=True)
    R = active_rows.numel()
    row_end = counts.cumsum(0)
    row_start = row_end - counts
    row_start = row_start.to(torch.int64).contiguous()
    row_end = row_end.to(torch.int64).contiguous()

    dense = torch.empty((R, N), dtype=torch.float32, device=self.device)
    if R > 0:
        BLOCK_N = _pick_block_n(N, self.dtype)
        grid = (R, triton.cdiv(N, BLOCK_N) if N > 0 else 1)
        with torch_device_fn.device(self.device):
            _sspaddmm_row_kernel[grid](
                row_start,
                row_end,
                cols,
                vals,
                mat,
                dense,
                1.0,
                N,
                mat.stride(0),
                mat.stride(1),
                IS_FP64=False,
                BLOCK_N=BLOCK_N,
            )

    # torch.smm materializes every column of each structural row.
    row_idx = active_rows.to(torch.int64).repeat_interleave(N)
    col_idx = torch.arange(N, dtype=torch.int64, device=self.device).repeat(R)
    out_indices = torch.stack([row_idx, col_idx], dim=0)
    out_values = dense.reshape(-1)

    return torch.sparse_coo_tensor(
        out_indices, out_values, size=(M, N), is_coalesced=True
    )
