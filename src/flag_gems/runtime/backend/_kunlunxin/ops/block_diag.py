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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# tle.gpu (cluster-path TensorDescriptor DMA) copies each source block straight
# into its diagonal slot in the output. This sidesteps the two compiler limits
# that cap the plain-Triton path on XPU: the interval store-mask degrades
# gm2lm/lm2gm to offsetState=-1 (conservative DMA) and the masked-select cone
# makes the whole-row store vectorization-ineligible (keyState=Conflict). A
# descriptor copy is a regular rectangular tile the hardware DMA engine runs at
# memcpy-class bandwidth. Guarded import so non-XPU / older triton still work.
try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _TLE_GPU_OK = True
except ImportError:
    tle = None
    TensorDescriptor = None
    _TLE_GPU_OK = False

# P800 cluster has 64 cores; the legacy TLE tile planner requires the per-core
# slice to fit in one innermost row, i.e. RT * bc / 64 <= bc, so cap the row
# slab at the core count.
_TLE_CORE_NUM = 64


@libentry()
@triton.jit
def _zero_fill_kernel(out_ptr, numel, BLOCK: tl.constexpr):
    """Contiguous zero fill (fully vectorizable prefix store) for the
    off-diagonal region before the tle.gpu block copies land."""
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, 0, mask=offs < numel)


if _TLE_GPU_OK:

    @triton.jit
    def block_diag_tlegpu_kernel(
        src_desc, out_desc, BR: tl.constexpr, BC: tl.constexpr, RT: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        """grid (n, block_rows // RT): each program stages an RT x BC row slab
        of one block from the contiguous (n*BR, BC) source into local memory,
        then writes it to the diagonal position [blk*BR + rt*RT, blk*BC] of the
        output via a descriptor copy. RT <= core count keeps the per-core slice
        within one innermost row (legacy TLE tile planner constraint)."""
        blk = tl.program_id(0)
        rt = tl.program_id(1)
        buf = tle.gpu.alloc([RT, BC], dtype=DTYPE, layout=None, scope=tle.gpu.lmem)
        src_row = blk * BR + rt * RT
        tle.gpu.copy(src_desc, buf, [RT, BC], [src_row, 0])
        tle.gpu.copy(buf, out_desc, [RT, BC], [src_row, blk * BC])


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


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def _row_tile_width(total_cols):
    """Contiguous store width per program. Wide stores are essential for
    store bandwidth on XPU, so cover whole rows when feasible."""
    return max(64, min(4096, _next_pow2(total_cols)))


_TL_DTYPE = {}
if _TLE_GPU_OK:
    _TL_DTYPE = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }

_ZERO_FILL_BLOCK = 16384


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

        # Pure-Triton strided path: check if tensors are regularly spaced
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

        # Preferred path: tle.gpu descriptor copy. Needs power-of-two block
        # dims (TensorDescriptor block_shape must be pow2, last dim stride 1)
        # and a supported float dtype. The output off-diagonal is zeroed by a
        # contiguous vectorizable fill; each block is then DMA'd to its
        # diagonal slot. The source blocks are gathered into one contiguous
        # (n*br, bc) staging buffer first (separate torch tensors do not share
        # storage, so an as_strided view across blocks is not valid even when
        # they happen to be packed).
        if (
            _TLE_GPU_OK
            and dtype0 in _TL_DTYPE
            and _is_pow2(block_rows)
            and _is_pow2(block_cols)
        ):
            rt = min(_TLE_CORE_NUM, block_rows)
            ptrs = _get_ptrs_tensor(tensors, device)
            staging = torch.empty(n * block_numel, dtype=dtype0, device=device)
            stage_grid = (
                n,
                (block_numel + _STAGE_BLOCK - 1) // _STAGE_BLOCK,
            )
            with torch_device_fn.device(device):
                block_diag_stage_kernel[stage_grid](
                    staging, ptrs, block_numel, BLOCK=_STAGE_BLOCK, num_warps=4
                )
            src2d = staging.view(total_rows, block_cols)

            out = torch.empty(
                (total_rows, total_cols), dtype=dtype0, device=device
            )
            out_numel = total_rows * total_cols
            src_desc = TensorDescriptor.from_tensor(
                src2d, block_shape=[rt, block_cols]
            )
            out_desc = TensorDescriptor.from_tensor(
                out, block_shape=[rt, block_cols]
            )
            with torch_device_fn.device(device):
                _zero_fill_kernel[
                    ((out_numel + _ZERO_FILL_BLOCK - 1) // _ZERO_FILL_BLOCK,)
                ](out, out_numel, BLOCK=_ZERO_FILL_BLOCK, num_warps=4)
                block_diag_tlegpu_kernel[(n, block_rows // rt)](
                    src_desc,
                    out_desc,
                    block_rows,
                    block_cols,
                    rt,
                    DTYPE=_TL_DTYPE[dtype0],
                )
            return out

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
