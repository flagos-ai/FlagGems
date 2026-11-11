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

from flag_gems.ops.cast_Byte import cast_Byte as _generic_cast_byte
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _cast_byte_value(x):
    # The vendor runtime narrows floating inputs through a saturating int32
    # conversion (observed in test_cast_Byte_fractional: 2**31 -> 255, 1e20 ->
    # 255, nan -> 255), unlike int64 truncation used by the generic op. The
    # XPU's f32->i32 (and f64->i32) hardware conversion saturates exactly like
    # the vendor cast kernel, so the plain `.to(tl.int32)` matches it
    # bit-for-bit.
    #
    # bfloat16 sources are loaded through a uint16 alias of the input (see
    # cast_Byte) and rebuilt as f32 from the raw bits: the natural
    # bf16 -> f32 -> i32 chain is folded by LLVM InstCombine into
    # `fptosi bfloat`, which the XPU LLVM shipped with flagtree 0.7.0 (used by
    # CI) cannot select -- it aborts with "Cannot select: fp_to_sint ...
    # bf16" (Fatal Python error: Aborted). Rebuilding from integer bits keeps
    # the conversion as `fptosi float`. Same workaround as the bf16 -> integer
    # path of to.py (_to_copy_from_bf16_contiguous_kernel).
    if x.dtype == tl.uint16:
        x = (x.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    elif x.dtype == tl.float16:
        x = x.to(tl.float32)
    if x.dtype.is_floating():
        x = x.to(tl.int32)
    return x.to(tl.uint8)


@libentry()
@triton.jit
def _cast_byte_kernel(X, Y, N, MULT: tl.constexpr, BLOCK: tl.constexpr):
    # MULT == 1 is the non-overlapping-and-dense case: the output keeps the
    # input strides, so the copy is the identity on the flat index. A layout
    # whose strides are a constant multiple of the dense output's (a uniform
    # subsample such as x[::2] or x[:, ::2]) uses MULT == c and stays a single
    # affine pass with no per-lane index arithmetic.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets * MULT, mask, other=0)
    tl.store(Y + offsets, _cast_byte_value(x), mask)


@libentry()
@triton.jit
def _cast_byte_row_kernel(
    X,
    Y,
    SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    ROW_DIM: tl.constexpr,
    ROW_LEN: tl.constexpr,
    NCOL_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program copies BLOCK consecutive columns of one "row": ROW_DIM is the
    # stride-1 dimension of the dense output, so the store is coalesced. The row
    # index is decomposed into the remaining dimensions with scalar arithmetic
    # and both addresses are affine in the column index. No per-lane div/mod and
    # no scatter/gather offset tensor is emitted: the previous flat
    # gather+scatter formulation was lowered as a "Discrete" per-line access on
    # the XPU backend, whose fast path silently dropped stores when the mocked
    # program-id range in OffsetAnalysis never observed the wrap of the offset
    # chain (see XPUTC-8029 / XPUTC-7980).
    pid = tl.program_id(0)
    row = pid // NCOL_BLOCKS
    col_block = pid % NCOL_BLOCKS
    src_base = 0
    dst_base = 0
    remaining = row
    for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
        if dim != ROW_DIM:
            idx = remaining % SHAPE[dim]
            src_base += idx * X_STRIDES[dim]
            dst_base += idx * Y_STRIDES[dim]
            remaining //= SHAPE[dim]
    cols = col_block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < ROW_LEN
    x = tl.load(X + src_base + cols * X_STRIDES[ROW_DIM], mask, other=0)
    tl.store(Y + dst_base + cols * Y_STRIDES[ROW_DIM], _cast_byte_value(x), mask)


def _infer_dense_strides(sizes, strides):
    """Port of ATen's infer_dense_strides (aten/src/ATen/ExpandUtils.cpp).

    ``_cast_Byte`` follows ``empty_like``'s Preserve semantics: for a strided
    input that is not non-overlapping-and-dense, the reference output layout is
    the "dense"-ified strides that keep the input's layout permutation. The
    vendor runtime's ``torch.empty_like`` instead falls back to a contiguous
    allocation for such inputs, so the generic ``cast_Byte`` reports
    contiguous strides and fails the stride assertions of tests/test_cast_Byte.
    This port reproduces the reference algorithm exactly (stable insertion sort
    with 0-stride dims pinned, equal strides resolved by smaller size first).
    """
    ndim = len(sizes)
    if ndim == 0:
        return []
    if ndim == 1:
        return [1]

    perm = list(range(ndim - 1, -1, -1))

    def should_swap(dim0, dim1):
        stride0 = strides[dim0]
        stride1 = strides[dim1]
        if stride0 == 0 or stride1 == 0:
            return 0
        if stride0 < stride1:
            return -1
        if stride0 > stride1:
            return 1
        if sizes[dim0] > sizes[dim1]:
            return 1
        return 0

    for i in range(1, ndim):
        dim1 = i
        for j in range(1, i + 1):
            dim0 = i - j
            comparison = should_swap(perm[dim0], perm[dim1])
            if comparison > 0:
                perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
                dim1 = dim0
            elif comparison < 0:
                break

    out_strides = [0] * ndim
    curr_stride = 1
    for i in range(ndim):
        idx = perm[i]
        out_strides[idx] = curr_stride
        if sizes[idx] > 1:
            curr_stride *= sizes[idx]
    return out_strides


def _uniform_multiple(sizes, in_strides, out_strides):
    """Return ``c`` if ``in_strides == c * out_strides`` on every dimension > 1.

    The dense output index then maps to ``c * out_flat`` in the input, because
    the output is contiguous in its own dense order (see _infer_dense_strides).
    """
    mult = None
    for size, in_stride, out_stride in zip(sizes, in_strides, out_strides):
        if size == 1:
            continue
        if in_stride == 0 or in_stride % out_stride:
            return None
        c = in_stride // out_stride
        if mult is None:
            mult = c
        elif mult != c:
            return None
    return mult


def _row_partition(sizes, out_strides):
    """Split the dense output into rows along its stride-1 dimension.

    ``out_strides`` comes from ``_infer_dense_strides``, so exactly one
    dimension of size > 1 has stride 1. Walking that dimension with a
    contiguous ``tl.arange`` keeps the store coalesced and every address affine.
    """
    row_dim = max(
        dim for dim in range(len(sizes)) if out_strides[dim] == 1 and sizes[dim] > 1
    )
    row_len = sizes[row_dim]
    rows = 1
    for dim, size in enumerate(sizes):
        if dim != row_dim:
            rows *= size
    block = min(1024, triton.next_power_of_2(row_len))
    return row_dim, row_len, rows, block, triton.cdiv(row_len, block)


def cast_Byte(self, non_blocking=False):
    logger.debug("GEMS_KUNLUNXIN CAST_BYTE")
    if self.dtype == torch.uint8:
        return self
    if (
        self.device.type == "cpu"
        or self.layout != torch.strided
        or self.is_complex()
        or self.is_quantized
        or self.is_neg()
        or self.dtype
        not in (
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        )
    ):
        # Unsupported inputs (complex/quantized/negative-view/sparse dtypes)
        # take the reference-vendor path via the generic implementation.
        return _generic_cast_byte(self, non_blocking=non_blocking)
    n = self.numel()
    if n == 0:
        return torch.empty_like(self, dtype=torch.uint8)
    with torch_device_fn.device(self.device):
        # bfloat16 is read through a uint16 alias (same element size, so the
        # reinterpretation keeps size/strides) and widened from its raw bits in
        # the kernel -- see _cast_byte_value.
        src = self.view(torch.uint16) if self.dtype == torch.bfloat16 else self
        if self.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(self):
            # Non-overlapping-and-dense inputs keep the input strides (ATen
            # empty_like Preserve), so the identity linear copy is layout-correct.
            out = torch.empty_strided(
                self.size(), self.stride(), dtype=torch.uint8, device=self.device
            )
            _cast_byte_kernel[(triton.cdiv(n, 1024),)](src, out, n, 1, BLOCK=1024)
        else:
            # Match the reference _cast_Byte output layout (infer_dense_strides).
            strides = _infer_dense_strides(tuple(self.size()), self.stride())
            out = torch.empty_strided(
                self.size(), strides, dtype=torch.uint8, device=self.device
            )
            mult = _uniform_multiple(tuple(self.size()), self.stride(), strides)
            if mult is not None:
                # Uniform subsample: the dense output index maps to `mult` times
                # itself in the input, so one flat affine pass is enough.
                block = min(2048, triton.next_power_of_2(n))
                _cast_byte_kernel[(triton.cdiv(n, block),)](
                    src, out, n, mult, BLOCK=block
                )
            else:
                # Copy row by row along the dense output's stride-1 dimension.
                row_dim, row_len, rows, block, ncol = _row_partition(
                    tuple(self.size()), strides
                )
                _cast_byte_row_kernel[(rows * ncol,)](
                    src,
                    out,
                    tuple(self.size()),
                    self.stride(),
                    tuple(strides),
                    row_dim,
                    row_len,
                    ncol,
                    BLOCK=block,
                )
    return out
