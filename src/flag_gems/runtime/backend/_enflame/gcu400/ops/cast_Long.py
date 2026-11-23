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
#
# GCU400 specialization of ``aten::_cast_Long`` (cast to int64).
#
# On GCU400 a kernel is billed mostly per *program*, not per element: the flat
# cast takes 2.03 ms for 16.7 M elements with 1024-element programs and 0.11 ms
# (the ATen level) once a program covers 65536 elements, i.e. the cost is
# proportional to the grid size. The kernels below are therefore built around
# "few, fat programs":
#   * operands whose element offsets form a dense span (contiguous, or a dense
#     permutation such as ``x.t()``) use one flat kernel, because the output
#     mirrors the layout and the element mapping is the identity in memory order;
#   * other rank-2 operands (strided slices such as ``x[:, ::2]``) use a row
#     kernel that walks the fastest-varying axis and gives every program 16 rows,
#     which removes the per-element rank-2 divmod;
#   * every other layout falls back to the generic rank-8 stride kernel.
# float64 never reaches a kernel: GCU400 has no fp64 vector lane, so even the
# fp64 *load* fails to legalize ("failed to legalize operation
# 'triton_gcu.elementwise_fusion_region'"). It is served by ATen's composite
# ``_to_copy`` kernel, which yields the same int64 image as the float32 path
# because torch_gcu stores Double as Float on the device.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.cast_Long import cast_Long as _generic_cast_Long

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

_FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )
    if dtype is not None
)

# Measured on GCU400 (float32, vs the ATen level of 5.8/14.0/109.7 us for
# 64 K/1 M/16.7 M elements): 8 K-element programs win while the grid stays small
# (1.41x at 64 K), 64 K-element programs win above that (1.07x at 1 M, 1.02x at
# 16.7 M).
_FLAT_BLOCK_SMALL = 8192
_FLAT_BLOCK_LARGE = 65536
_FLAT_WARPS = 4
_SMALL_NUMEL = 262144

# Strided rank-2 operands: rows per program. Re-measured on the 1024x512
# stride-2 slice (1024 rows): 32 rows -> 32 programs in a single wave (21.7 us)
# beats 16 rows -> 64 programs (24.7 us) and 64 rows (38.8 us).
_ROW_ROWS = 32
_ROW_BLOCK = 1024
_ROW_WARPS = 4

_INT32_LIMIT = 0x7FFFFFFF


@triton.jit
def _cast_long_flat_kernel(
    x_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        # numel is a multiple of BLOCK: skip predication entirely.
        v = tl.load(x_ptr + offs)
        if MODE == 1:
            r = v.to(tl.float32).to(tl.int64)
        else:
            r = v.to(tl.int64)
        tl.store(out_ptr + offs, r)
    else:
        mask = offs < numel
        v = tl.load(x_ptr + offs, mask=mask, other=0)
        if MODE == 1:
            r = v.to(tl.float32).to(tl.int64)
        else:
            r = v.to(tl.int64)
        tl.store(out_ptr + offs, r, mask=mask)


@triton.jit
def _cast_long_row_kernel(
    x_ptr,
    out_ptr,
    inner,
    x_row_stride,
    x_step,
    o_row_stride,
    o_step,
    row_count,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    MODE: tl.constexpr,
    X_STEP1: tl.constexpr,
    O_STEP1: tl.constexpr,
):
    # Each program owns ROWS consecutive rows of the fastest-varying axis.
    # Input and output carry their own strides: ``empty_like`` returns a
    # contiguous tensor for a strided source, so the two layouts must not be
    # assumed to match (using the source strides for the store would both
    # misplace the results and write past the end of the output buffer).
    row_base = tl.program_id(0) * ROWS
    col = tl.program_id(1)
    offs = col * BLOCK + tl.arange(0, BLOCK)
    mask = offs < inner
    if X_STEP1:
        x_offs = offs
    else:
        x_offs = offs * x_step
    if O_STEP1:
        o_offs = offs
    else:
        o_offs = offs * o_step
    for row_delta in tl.static_range(ROWS):
        row = row_base + row_delta
        if row < row_count:
            v = tl.load(x_ptr + row * x_row_stride + x_offs, mask=mask, other=0)
            if MODE == 1:
                r = v.to(tl.float32).to(tl.int64)
            else:
                r = v.to(tl.int64)
            tl.store(out_ptr + row * o_row_stride + o_offs, r, mask=mask)


def _cast_mode(dtype):
    """Return the conversion MODE for ``dtype`` (1 = via float32)."""
    if dtype in _FP8_DTYPES:
        return 1
    if dtype.is_floating_point:
        return 1
    return 0


def _flat_block(numel):
    if numel > _SMALL_NUMEL:
        return _FLAT_BLOCK_LARGE
    # Small operands stay at their own power-of-two size: a 1024-element cast in a
    # 8192-wide program wastes 7/8 of the lanes (measured 4.3 us vs 3.4 us).
    return min(_FLAT_BLOCK_SMALL, max(1024, triton.next_power_of_2(numel)))


def _covers_dense_span(shape, stride):
    """True when the element offsets form a dense [0, numel) span.

    Sorting the axes by stride must reproduce a contiguous layout, which is the
    case for ``x.t()`` (strides (1, 1024) over (1024, 1024)) but not for a strided
    slice such as ``x[:, ::2]``. torch_gcu does not expose
    ``Tensor.is_non_overlapping_and_dense``, so the check is done by hand.
    """
    axes = [axis for axis in range(len(shape)) if shape[axis] > 1]
    expected = 1
    for axis in sorted(axes, key=lambda item: stride[item]):
        if stride[axis] != expected:
            return False
        expected *= shape[axis]
    return True


def cast_Long(self, non_blocking=False):
    logger.debug("GEMS_ENFLAME CAST_LONG")
    if not isinstance(self, torch.Tensor):
        raise TypeError(
            "_cast_Long() received an invalid combination of arguments - got %r, "
            "but expected Tensor self" % type(self).__name__
        )
    if not isinstance(non_blocking, bool):
        raise TypeError(
            "non_blocking must be a bool, got %r" % type(non_blocking).__name__
        )

    if self.dtype == torch.int64:
        return self

    if self.dtype == torch.float64:
        # No fp64 vector lane on GCU400: the load itself does not legalize.
        return torch.ops.aten._to_copy.default.redispatch(
            _FALLBACK_KEYSET,
            self,
            dtype=torch.int64,
            non_blocking=non_blocking,
        )

    out = torch.empty_like(self, dtype=torch.int64)
    numel = self.numel()
    if numel == 0:
        return out

    if numel > _INT32_LIMIT:
        # int32 offsets would wrap: the generic kernel has int64 addressing.
        return _generic_cast_Long(self, non_blocking)

    mode = _cast_mode(self.dtype)

    # ``empty_like`` preserves the layout, so a dense footprint (contiguous or a
    # dense permutation such as ``x.t()``) can be swept in memory order.
    if self.is_contiguous() or _covers_dense_span(list(self.shape), list(self.stride())):
        block = _flat_block(numel)
        grid = (triton.cdiv(numel, block),)
        _cast_long_flat_kernel[grid](
            self,
            out,
            numel,
            BLOCK=block,
            MODE=mode,
            EVEN=(numel % block) == 0,
            num_warps=_FLAT_WARPS,
        )
        return out

    shape = list(self.shape)
    stride = list(self.stride())
    if len(shape) == 2:
        fast = 0 if stride[0] <= stride[1] else 1
        other = 1 - fast
        inner = shape[fast]
        row_count = shape[other]
        # The element offset bound is decided by stride and can be far above numel: once it is
        # crossed the int64 addressing path has to be taken.
        max_offset = (row_count - 1) * stride[other] + (inner - 1) * stride[fast]
        if (
            inner > 0
            and row_count > 0
            and stride[other] > stride[fast]
            and max_offset <= _INT32_LIMIT
        ):
            out_stride = list(out.stride())
            block = min(_ROW_BLOCK, triton.next_power_of_2(inner))
            rows = max(1, min(_ROW_ROWS, row_count))
            grid = (triton.cdiv(row_count, rows), triton.cdiv(inner, block))
            _cast_long_row_kernel[grid](
                self,
                out,
                inner,
                stride[other],
                stride[fast],
                out_stride[other],
                out_stride[fast],
                row_count,
                BLOCK=block,
                ROWS=rows,
                MODE=mode,
                X_STEP1=(stride[fast] == 1),
                O_STEP1=(out_stride[fast] == 1),
                num_warps=_ROW_WARPS,
            )
            return out

    return _generic_cast_Long(self, non_blocking)
