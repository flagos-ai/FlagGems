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

logger = logging.getLogger(__name__)

# Payload of one program. The real/imag pair of a complex element is stored
# adjacently and the destination is contiguous, so a dense run of reals is
# copied as one coalesced vector pass. The streaming throughput of that pass is
# bounded by the tile a program keeps resident, not by the vector width: 16 KiB
# tiles reach ~0.6 TB/s while 512 KiB tiles reach the ~3 TB/s of the DTE-backed
# aten copy (measured on a 1 GiB fp32 payload). 512 KiB stays below the 917504
# byte DSM budget of a program, and the block is additionally clamped to the
# row length so small inputs do not reserve a huge tile.
_TILE_BYTES = 524288
_MIN_BLOCK_SIZE = 1024
# The generic gather path keeps the block width that the 2-D masked load path
# was verified with; it is only used for layouts whose rows cannot be merged.
_STRIDED_BLOCK_SIZE = 1024
# GCU allows grid.x up to 65535 but only 255 blocks on each of grid.y/grid.z,
# far below what a large tensor needs, so the flat program count is folded into
# (grid.x, grid.y) and the kernel recovers a program id as x + y * grid_x.
_MAX_GRID_X = 65535
_MAX_GRID_Y = 255
# Total programs are additionally bounded by x * y; anything larger falls back
# to the generic gather path, which splits its own work across the same axes.
_MAX_LAUNCH = _MAX_GRID_X * _MAX_GRID_Y


def _launch_grid(n_blocks):
    """Fold a flat program count into (grid_x, grid_y, 1) with minimal waste."""
    grid_y = triton.cdiv(n_blocks, _MAX_GRID_X)
    grid_x = triton.cdiv(n_blocks, grid_y)
    if grid_x > _MAX_GRID_X:
        # ``n_blocks`` sits just above a multiple of grid.x (e.g. 65536 blocks
        # with a 65535-wide axis): rounding the row up keeps every block
        # covered, the padded programs are dropped by the kernel's block mask.
        grid_x = _MAX_GRID_X
        grid_y = triton.cdiv(n_blocks, grid_x)
    return (grid_x, grid_y, 1)


def _merge_real_rows(shape, strides):
    """Split a ``(..., 2)`` layout into (n_rows, row_len, src_row_stride).

    Trailing dimensions whose stride equals the dense run accumulated so far
    are merged, so ``row_len`` is the number of contiguous real elements that
    can be copied as one flat stream and ``src_row_stride`` is the row pitch in
    real elements. Returns ``None`` when the remaining leading dimensions do not
    form their own dense row-major layout, i.e. when the move is not a set of
    uniformly strided dense rows and the generic gather kernel is required.
    """
    row_len = shape[-1]
    last = len(shape) - 2
    while last >= 0 and strides[last] == row_len:
        row_len *= shape[last]
        last -= 1
    for dim in range(last - 1, -1, -1):
        if strides[dim] != strides[dim + 1] * shape[dim + 1]:
            return None
    n_rows = 1
    for dim in range(0, last + 1):
        n_rows *= shape[dim]
    src_row_stride = strides[last] if last >= 0 else row_len
    return n_rows, row_len, src_row_stride


@triton.jit
def _view_as_complex_copy_rows_kernel(
    src_ptr,
    dst_ptr,
    total_blocks,
    blocks_per_row,
    row_len,
    src_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy dense real rows of ``row_len`` elements each.

    Every source row is a contiguous run of reals (the trailing dimensions,
    including the size-2 real/imag pair, were merged on the host) and the
    destination is contiguous with the same row length, so one program copies
    one dense slice. The flat program id is split into a row index and a block
    index inside that row; a fully contiguous input degenerates to a single row
    with ``row_len == numel``, i.e. one flat pass. ``total_blocks`` masks out
    the padding programs left over by the grid folding.
    """
    pid = (tl.program_id(axis=0) + tl.program_id(axis=1) * tl.num_programs(axis=0)).to(
        tl.int64
    )
    row = pid // blocks_per_row
    block = pid - row * blocks_per_row
    offset = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offset < row_len) & (pid < total_blocks)
    values = tl.load(src_ptr + row * src_row_stride + offset, mask=mask)
    tl.store(dst_ptr + row * row_len + offset, values, mask=mask)


_COMPLEX_DTYPE = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}
_COMPLEX_TO_REAL = {complex: real for real, complex in _COMPLEX_DTYPE.items()}

# Leading (batch) dimensions are decomposed with a fixed number of slots so the
# generic gather kernel always receives its full argument sets; higher-rank
# inputs that need that path are rejected by the wrapper.
_MAX_LEAD_RANK = 5


@triton.jit
def _view_as_complex_copy_strided_kernel(
    src_ptr,
    dst_ptr,
    d0,
    d1,
    d2,
    d3,
    d4,
    s0,
    s1,
    s2,
    s3,
    s4,
    n_complex,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy a non-contiguous ``(..., 2)`` real buffer into complex storage.

    The output is contiguous, so the flat complex index decomposes row-major
    over the leading (batch) dimensions; each recovered coordinate is scaled
    by the matching *input* stride (expressed in real elements). The trailing
    dimension keeps stride 1, so the (real, imag) pair of a complex element is
    always adjacent and is loaded as a two-wide vector.

    The slots carry the leading dimensions in input order: ``d0``/``s0`` is
    the outermost lead dimension and ``d[RANK-1]``/``s[RANK-1]`` the innermost
    one (the dimension immediately before the complex pair). The generic path
    peels digits from the innermost slot outward, so both paths share the same
    convention.

    A ``RANK == 2`` specialization is kept because it is the overwhelmingly
    common non-contiguous case (a row/column slice of an otherwise flat
    buffer); the tight two-step ``div``/``sub`` decomposition beats the generic
    static loop by ~40% on that layout.
    """
    pid = tl.program_id(axis=0) + tl.program_id(axis=1) * tl.num_programs(axis=0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i < n_complex
    i = i.to(tl.int64)

    if RANK == 2:
        # d1/s1 is the innermost lead dim (fastest digit), d0/s0 the outer one.
        outer = i // d1
        src_base = (i - outer * d1) * s1 + outer * s0
    else:
        lead_dims = [d0, d1, d2, d3, d4]
        lead_strides = [s0, s1, s2, s3, s4]
        rem = i
        src_base = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        for j in tl.static_range(RANK - 1, -1, -1):
            idx = rem % lead_dims[j]
            rem = rem // lead_dims[j]
            src_base += idx * lead_strides[j]

    lane = tl.arange(0, 2)
    src_offsets = src_base[:, None] + lane[None, :]
    dst_offsets = i[:, None] * 2 + lane[None, :]
    mask2 = mask[:, None]
    values = tl.load(src_ptr + src_offsets, mask=mask2)
    tl.store(dst_ptr + dst_offsets, values, mask=mask2)


def view_as_complex_copy(self: torch.Tensor) -> torch.Tensor:
    r"""Copy a real tensor whose last dimension has size 2 into a complex tensor.

    Mirrors ``aten::view_as_complex_copy``: the out-of-place counterpart of
    ``torch.view_as_complex``. The input's last dimension must have size 2
    and stride 1 (batch dimensions may be non-contiguous); ``float32`` maps
    to ``complex64``, ``float64`` to ``complex128`` and ``float16`` to
    ``complex32``. The result is a contiguous, freshly allocated tensor that
    never aliases the input. The same errors PyTorch raises are raised: 0-dim
    inputs, a wrong last-dimension size or stride, and unsupported dtypes
    (bfloat16 and every non-float type).
    """
    logger.debug("GEMS_ENFLAME VIEW_AS_COMPLEX_COPY")

    if self.dim() == 0:
        raise RuntimeError("Input tensor must have one or more dimensions")
    if self.size(-1) != 2:
        raise RuntimeError("Tensor must have a last dimension of size 2")
    complex_dtype = _COMPLEX_DTYPE.get(self.dtype)
    if complex_dtype is None:
        raise RuntimeError(
            "view_as_complex is only supported for half, float and double "
            "tensors, but got a tensor of scalar type: "
            f"{str(self.dtype).replace('torch.', '').capitalize()}"
        )
    if self.stride(-1) != 1:
        raise RuntimeError("Tensor must have a last dimension with stride 1")

    # ``view_as_real`` refuses tensors carrying a lazy conjugate or negative
    # bit; resolving them first is an exact identity for the reinterpreted
    # values and keeps the copy on the logical data.
    if self.is_conj():
        self = self.resolve_conj()
    if self.is_neg():
        self = self.resolve_neg()

    leading = tuple(self.shape[:-1])
    out = torch.empty(leading, dtype=complex_dtype, device=self.device)
    n_complex = out.numel()
    if n_complex == 0:
        return out

    # float64 is copied in 32-bit lanes: the GCU compiler aborts the whole
    # process on *any* float64 Triton kernel (the vectorization pass asserts on
    # the emulated double type) and the 64-bit integer lane faults at runtime,
    # while 32-bit is the backend's native width. float64/complex128 -> int32
    # keeps the byte count (4 int32 per real, 8 per complex), so the move stays
    # bit-exact.
    if self.dtype == torch.float64:
        src_view = self.view(torch.int32)
        dst_view = out.view(torch.int32)
    else:
        src_view = self
        dst_view = torch.view_as_real(out)
    # Dense runs of reals (everything from the last dimension down to the first
    # non-dense one) are copied as flat rows; only the residual layouts that
    # cannot be expressed as uniformly strided dense rows fall back to the
    # generic index-decomposing gather.
    layout = _merge_real_rows(tuple(src_view.shape), tuple(src_view.stride()))
    with torch_device_fn.device(self.device):
        if layout is not None:
            n_rows, row_len, src_row_stride = layout
            # The block is derived from the element width so that every program
            # moves the same ~16 KiB of payload regardless of the dtype; work
            # that does not fit one grid row is folded by _launch_grid instead
            # of widening the tile further.
            block_size = max(_MIN_BLOCK_SIZE, _TILE_BYTES // src_view.element_size())
            # Never reserve more lanes than the row can use: the tile is a
            # per-program DSM allocation, so a long row keeps the full payload
            # while a short row only pays for its own length. Splitting a
            # short row further was measured *slower* (a 16 KiB row per
            # program beat four 4 KiB programs per row by 3x on a 268 MB
            # strided-row payload), so the row length stays the hard cap.
            block_size = min(block_size, triton.next_power_of_2(row_len))
            blocks_per_row = triton.cdiv(row_len, block_size)
            total_blocks = n_rows * blocks_per_row
        if layout is not None and total_blocks <= _MAX_LAUNCH:
            _view_as_complex_copy_rows_kernel[_launch_grid(total_blocks)](
                src_view,
                dst_view,
                total_blocks,
                blocks_per_row,
                row_len,
                src_row_stride,
                BLOCK_SIZE=block_size,
            )
        else:
            rank = self.dim() - 1
            if rank > _MAX_LEAD_RANK:
                raise RuntimeError(
                    f"view_as_complex_copy supports at most {_MAX_LEAD_RANK + 1} "
                    f"dimensions, got {self.dim()}"
                )
            lead_dims = list(src_view.shape[:-1]) + [1] * (_MAX_LEAD_RANK - rank)
            lead_strides = list(src_view.stride()[:-1]) + [0] * (_MAX_LEAD_RANK - rank)
            grid = _launch_grid(triton.cdiv(n_complex, _STRIDED_BLOCK_SIZE))
            _view_as_complex_copy_strided_kernel[grid](
                src_view,
                dst_view,
                *lead_dims,
                *lead_strides,
                n_complex,
                RANK=rank,
                BLOCK_SIZE=_STRIDED_BLOCK_SIZE,
            )
    return out


# ---------------------------------------------------------------------------
# torch.Tensor.to complex32-cast shim (torch_gcu P21 defect workaround)
#
# topsatenCopy aborts the whole process for any on-device dtype cast whose
# *source* is complex32 (ComplexHalf -> ComplexFloat/ComplexDouble hits
# TOPSATEN_STATUS_NOT_SUPPORT and SIGABRTs instead of raising), which kills
# every consumer that upcasts a complex32 gcu tensor -- e.g. the accuracy
# harness comparing float16->complex32 results through a lossless complex64
# upcast (complex32 has no CPU equal kernel).  ENFLAME_PT_OP_DEBUG_CONFIG
# cannot intercept this: _to_copy has a gcu kernel registration and never
# enters the fallback_cpu list (P21).  The unsupported cast decomposes into
# steps topsaten fully supports: view the complex tensor as its real dtype,
# widen the (real, imag) pair to the target's real dtype, and view back as
# complex -- bit-exact and on device.  Layouts whose view_as_real would be
# rejected fall back to a value-identical CPU round trip.  Every other call
# shape passes through to the original ``to`` untouched, and only
# complex32-on-gcu sources are intercepted (complex64/complex128 casts are
# natively supported).  Drop once topsatenCopy handles ComplexHalf.
# ---------------------------------------------------------------------------


def _install_complex_cast_to_shim():
    current = torch.Tensor.to
    if getattr(current, "_flaggems_gcu_complex_cast_safe", False):
        return

    def _complex_cast_safe_to(self, *args, **kwargs):
        if args:
            target_dtype = args[0] if isinstance(args[0], torch.dtype) else None
        elif kwargs.get("device") is None:
            target_dtype = kwargs.get("dtype")
        else:
            target_dtype = None
        if (
            target_dtype is not None
            and target_dtype != self.dtype
            and target_dtype in _COMPLEX_TO_REAL
            and self.dtype == torch.complex32
            and self.device.type == "gcu"
        ):
            try:
                real_pair = torch.view_as_real(self)
                real_pair = current(real_pair, _COMPLEX_TO_REAL[target_dtype])
                return torch.view_as_complex(real_pair)
            except RuntimeError:
                on_host = current(self, "cpu")
                return current(current(on_host, target_dtype), self.device)
        return current(self, *args, **kwargs)

    _complex_cast_safe_to._flaggems_gcu_complex_cast_safe = True
    torch.Tensor.to = _complex_cast_safe_to


_install_complex_cast_to_shim()
