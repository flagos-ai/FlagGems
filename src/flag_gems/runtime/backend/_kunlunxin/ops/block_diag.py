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

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# tle.raw lets us embed a hand-written cluster-C payload (bd_raw.xpu) that
# drives per-core GM2LM/LM2GM DMA directly. Measured on P800 this moves the
# output about 2.4x faster than compiler-generated cluster kernels (875GB/s
# vs 364GB/s for the zero fill) because it bypasses the gm2lm/lm2gm offset
# analysis and vectorization limits. Fall back to plain Triton kernels when
# the extension is unavailable.
try:
    import triton.experimental.tle as tle

    _TLE_OK = True
except ImportError:
    tle = None
    _TLE_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))

# P800 (xpu3): 12 clusters, 64 cores each; one Triton program == one cluster.
_NCLUSTER = 12
# The payload keeps up to 64 block pointers in per-core local memory.
_RAW_MAX_BLOCKS = 64

if _TLE_OK:

    @tle.raw.dialect("xpu3", file=os.path.join(_HERE, "bd_raw.xpu"))
    def bd_raw(ptrs, out, n, br, bc, esz, total_rows, rows_start, rows_count):
        ...

    @triton.jit(do_not_specialize=["n", "br", "bc", "esz", "per"])
    def block_diag_raw_kernel(Ptrs, Out, n, br, bc, esz, per):
        pid = tl.program_id(0)
        tle.raw.call(
            bd_raw, (Ptrs, Out, n, br, bc, esz, n * br, pid * per, per)
        )


@libentry()
@triton.jit
def block_diag_strided_row_kernel(
    out_ptr,
    base_ptr,
    base_offset,
    input_stride,
    block_rows,
    block_cols,
    total_cols,
    LOG2_BC: tl.constexpr,
    TC: tl.constexpr,
):
    """Write full output rows: diagonal-block data in place, zeros elsewhere.

    Grid is (block_rows, num_blocks, col_chunks) so the block id and the
    row-within-block come directly from program ids - no per-program integer
    division, which is expensive on XPU. Sources are read from one regularly
    strided allocation (or a contiguous staging buffer) via direct pointer
    arithmetic, and stores cover whole rows contiguously: store bandwidth on
    XPU depends on the contiguous store width per program, so segmented
    (block-wide) stores must be avoided.

    When block_cols is a power of two, the segment membership test reduces to
    one shift plus one compare, which is markedly cheaper on wide vectors than
    two comparisons plus and.
    """
    r = tl.program_id(0)
    blk = tl.program_id(1)
    pid_c = tl.program_id(2)
    row = blk * block_rows + r
    cols = pid_c * TC + tl.arange(0, TC)
    col_off = blk * block_cols
    if LOG2_BC >= 0:
        in_seg = (cols >> LOG2_BC) == blk
    else:
        in_seg = (cols >= col_off) & (cols < col_off + block_cols)
    src_base = base_offset + blk * input_stride + r * block_cols
    val = tl.load(base_ptr + src_base + (cols - col_off), mask=in_seg)
    val = tl.where(in_seg, val, 0)
    tl.store(
        out_ptr + row.to(tl.int64) * total_cols + cols,
        val,
        mask=cols < total_cols,
    )


@libentry()
@triton.jit
def block_diag_stage_kernel(
    staging_ptr,
    ptrs_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    """Copy separately allocated blocks into one contiguous staging buffer.

    Grid is (num_blocks, tiles_per_block); each program loads its block
    pointer once (scalar loads are synchronous on XPU, so their count is
    kept proportional to the number of blocks instead of the number of
    output rows) and copies one contiguous chunk with contiguous 1D loads
    and stores. BLOCK trades per-chunk size against the number of
    synchronous pointer loads; 16K elements per program measured fastest.
    """
    blk = tl.program_id(0)
    j = tl.program_id(1)
    offs = j * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    ptr_val = tl.load(ptrs_ptr + blk)
    src_ptr = ptr_val.to(tl.pointer_type(staging_ptr.dtype.element_ty))
    val = tl.load(src_ptr + offs, mask=m)
    tl.store(staging_ptr + blk.to(tl.int64) * numel + offs, val, mask=m)


@libentry()
@triton.jit
def block_diag_varlen_general_kernel(
    out_ptr,
    ptrs_ptr,
    meta_ptr,
    total_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy variable-sized blocks using pointer array (no torch.cat needed)."""
    tile_id = ext.program_id(0)
    block_id = ext.program_id(1)

    base = block_id * 4
    row_off = tl.load(meta_ptr + base + 0)
    col_off = tl.load(meta_ptr + base + 1)
    rows = tl.load(meta_ptr + base + 2)
    cols = tl.load(meta_ptr + base + 3)

    block_numel = rows * cols
    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < block_numel

    r = offs // cols
    c = offs % cols
    out_idx = (row_off + r) * total_cols + (col_off + c)

    # Load pointer for this block and cast to correct element type
    ptr_val = tl.load(ptrs_ptr + block_id)
    src_ptr = ptr_val.to(tl.pointer_type(out_ptr.dtype.element_ty))

    val = tl.load(src_ptr + offs, mask=mask)
    tl.store(out_ptr + out_idx, val, mask=mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _log2_pow2(n):
    """log2(n) if n is a power of two (and n > 0), else -1."""
    if n > 0 and (n & (n - 1)) == 0:
        return n.bit_length() - 1
    return -1


def _row_tile_width(total_cols):
    """Contiguous store width per program. Wide stores are essential for
    store bandwidth on XPU, so cover whole rows when feasible."""
    return max(64, min(4096, _next_pow2(total_cols)))


# Cache of device-side pointer arrays for the fast path. Entries keep the
# source tensors alive so their data pointers stay valid while cached; the
# cache is bounded, so at most a few input sets are retained.
_ptrs_cache = {}
_PTRS_CACHE_MAX = 8

_STAGE_BLOCK = 16384


def _get_ptrs_tensor(tensors, device):
    key = tuple(t.data_ptr() for t in tensors)
    entry = _ptrs_cache.get(key)
    if entry is None:
        if len(_ptrs_cache) >= _PTRS_CACHE_MAX:
            _ptrs_cache.pop(next(iter(_ptrs_cache)))
        entry = (
            torch.tensor(key, dtype=torch.int64, device=device),
            tensors,
        )
        # The host-to-device copy of a fresh pointer array is not reliably
        # ordered before a Triton kernel launch on every backend; make the
        # values visible before the kernel can observe them.
        torch_device_fn.synchronize()
        _ptrs_cache[key] = entry
    return entry[0]


def block_diag(*tensors):
    """Block diagonal matrix construction using Triton kernel."""
    logger.debug("GEMS_KUNLUNXIN BLOCK_DIAG")

    # Handle case where tensors is passed as a single list/tuple
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])

    if len(tensors) == 0:
        return torch.tensor([])

    n = len(tensors)

    # Fast check: are all 2D, same shape, same dtype, contiguous?
    t0 = tensors[0]
    if t0.ndim == 2:
        shape0 = t0.shape
        dtype0 = t0.dtype
        fast_path = t0.is_contiguous() and (
            n == 1
            or all(
                t.ndim == 2
                and t.shape == shape0
                and t.dtype == dtype0
                and t.is_contiguous()
                for t in tensors[1:]
            )
        )
    else:
        fast_path = False

    if fast_path:
        block_rows, block_cols = shape0
        block_numel = block_rows * block_cols
        total_rows = n * block_rows
        total_cols = n * block_cols
        device = t0.device

        if block_numel == 0:
            return torch.zeros((total_rows, total_cols), dtype=dtype0, device=device)

        if n == 1:
            return t0.clone()

        # Preferred path: one launch of the tle.raw cluster-C payload. It
        # drives per-core DMA directly (fast writes), reads block pointers
        # from a device array (no staging pass) and needs a single kernel.
        if _TLE_OK and n <= _RAW_MAX_BLOCKS:
            ptrs = _get_ptrs_tensor(tensors, device)
            out = torch.empty(
                (total_rows, total_cols), dtype=dtype0, device=device
            )
            per = (total_rows + _NCLUSTER - 1) // _NCLUSTER
            with torch_device_fn.device(device):
                block_diag_raw_kernel[(_NCLUSTER,)](
                    ptrs,
                    out.view(torch.uint8),
                    n,
                    block_rows,
                    block_cols,
                    t0.element_size(),
                    per,
                )
            return out

        # Try strided path: check if tensors are regularly spaced
        base_ptr_val = t0.data_ptr()
        elem_bytes = t0.element_size()

        off1 = (tensors[1].data_ptr() - base_ptr_val) // elem_bytes
        stride = off1
        if n <= 2:
            regular_stride = stride != 0
        elif stride == 0:
            regular_stride = False
        else:
            regular_stride = all(
                (tensors[i].data_ptr() - base_ptr_val) // elem_bytes == i * stride
                for i in range(2, n)
            )

        out = torch.empty((total_rows, total_cols), dtype=dtype0, device=device)
        TC = _row_tile_width(total_cols)
        LOG2_BC = _log2_pow2(block_cols)
        grid = (block_rows, n, (total_cols + TC - 1) // TC)

        if regular_stride:
            # Zero-copy: read directly from original memory with stride
            with torch_device_fn.device(device):
                block_diag_strided_row_kernel[grid](
                    out,
                    t0,
                    0,
                    stride,
                    block_rows,
                    block_cols,
                    total_cols,
                    LOG2_BC=LOG2_BC,
                    TC=TC,
                    num_warps=4,
                )
        else:
            # Separately allocated blocks: gather them into one contiguous
            # staging buffer with a dedicated kernel, then write the output
            # rows via a direct pointer. torch.cat / copy_ based staging is
            # avoided on purpose: those ops may be patched to slower kernels
            # under use_gems(), and per-row pointer loads inside the row
            # kernel serialize on XPU because of synchronous scalar loads.
            ptrs = _get_ptrs_tensor(tensors, device)
            staging = torch.empty(
                n * block_numel, dtype=dtype0, device=device
            )
            stage_grid = (
                n,
                (block_numel + _STAGE_BLOCK - 1) // _STAGE_BLOCK,
            )
            with torch_device_fn.device(device):
                block_diag_stage_kernel[stage_grid](
                    staging,
                    ptrs,
                    block_numel,
                    BLOCK=_STAGE_BLOCK,
                    num_warps=4,
                )
                block_diag_strided_row_kernel[grid](
                    out,
                    staging,
                    0,
                    block_numel,
                    block_rows,
                    block_cols,
                    total_cols,
                    LOG2_BC=LOG2_BC,
                    TC=TC,
                    num_warps=4,
                )
        return out

    # General path: normalize, compute dtype, handle mixed shapes
    tensors_2d = []
    for t in tensors:
        if t.ndim == 0:
            tensors_2d.append(t.unsqueeze(0).unsqueeze(0))
        elif t.ndim == 1:
            tensors_2d.append(t.unsqueeze(0))
        else:
            assert t.ndim == 2, f"Expected 0D, 1D, or 2D tensor, got {t.ndim}D"
            tensors_2d.append(t)

    total_rows = sum(t.shape[0] for t in tensors_2d)
    total_cols = sum(t.shape[1] for t in tensors_2d)

    out_dtype = tensors_2d[0].dtype
    for t in tensors_2d[1:]:
        out_dtype = torch.result_type(
            torch.empty(0, dtype=out_dtype), torch.empty(0, dtype=t.dtype)
        )
    device = tensors_2d[0].device

    out = torch.zeros((total_rows, total_cols), dtype=out_dtype, device=device)

    meta_list = []
    ptrs_list = []
    src_tensors = []  # Keep references to prevent GC
    cur_row = 0
    cur_col = 0
    max_numel = 0

    for t in tensors_2d:
        rows, cols = t.shape
        numel = rows * cols
        meta_list.extend([cur_row, cur_col, rows, cols])
        if numel > 0:
            src = (
                t
                if (t.is_contiguous() and t.dtype == out_dtype)
                else t.contiguous().to(out_dtype)
            )
            ptrs_list.append(src.data_ptr())
            src_tensors.append(src)
        else:
            ptrs_list.append(0)
        max_numel = max(max_numel, numel)
        cur_row += rows
        cur_col += cols

    if max_numel == 0:
        return out

    ptrs = torch.tensor(ptrs_list, dtype=torch.int64, device=device)
    meta = torch.tensor(meta_list, dtype=torch.int64, device=device)

    BLOCK_SIZE = 1024
    num_tiles = (max_numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_tiles, len(tensors_2d))

    with torch_device_fn.device(device):
        block_diag_varlen_general_kernel[grid](
            out,
            ptrs,
            meta,
            total_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out
