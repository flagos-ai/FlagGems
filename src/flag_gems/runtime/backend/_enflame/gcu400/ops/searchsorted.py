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

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_CUDA_BLOCK_SIZE = 256
# Kept for the generic vector path used by the sorter / ascend branches.
SCALAR_BLOCK_SIZE = 64
# Use int32 indices instead of int64 on 1D sequences to reduce address-computation overhead.
GCU400_SCALAR_INT32_INDEX = True
# The GCU400 tensor path also uses int32 indices (software emulation of the i64 address state chain is costly); when
# numel exceeds 2^31
# it automatically falls back to i64 for correctness; the scalar entry point uses _searchsorted_scalar_kernel (fully
# scalar binary search,
# no idle lanes), with the block width chosen by _select_launch_config based on dtype/workload.
GCU400_TENSOR_INT32_INDEX = True
# Single-scalar queries below this length use the counting kernel instead of the
# binary search: one coalesced load + a full reduction beats LOG2(n) dependent
# global loads on GCU400. Measured (median, fp16/int32, warm cache):
#   n=1024 4.85us, 4096 4.46us, 8192 5.31us, 16384 5.77us, 32768 6.99us
# versus the binary-search kernel 7.37/8.11/8.92/... us. Beyond 32768 the
# per-query O(n) traffic loses to O(log n) probes, so the binary search keeps
# serving long sequences.
_SCALAR_COUNT_MAX_ELEMENTS = 32768
_GCU_BLOCK_SIZE = 512
_GCU_NUM_WARPS = 4
_ASCEND_BLOCK_SIZE = 512
_SUPPORTED_INPUT_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


@triton.jit
def _searchsorted_kernel(
    sorted_sequence,
    values,
    sorter,
    out,
    total_values,
    values_per_row,
    sequence_len,
    LOG_SEQUENCE_LEN: tl.constexpr,
    RIGHT: tl.constexpr,
    HAS_SORTER: tl.constexpr,
    IS_1D_SEQUENCE: tl.constexpr,
    USE_INT32_INDEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_values
    values_in = tl.load(values + offsets, mask=mask, other=0)

    if IS_1D_SEQUENCE:
        if USE_INT32_INDEX:
            row_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        else:
            row_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    else:
        row_offsets = (offsets // values_per_row) * sequence_len
        if USE_INT32_INDEX:
            row_offsets = row_offsets.to(tl.int32)

    if USE_INT32_INDEX:
        low = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    else:
        low = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    high = low + sequence_len

    for _ in range(LOG_SEQUENCE_LEN):
        active = mask & (low < high)
        mid = low + (high - low) // 2
        sorted_offsets = row_offsets + mid
        if HAS_SORTER:
            sorted_index = tl.load(sorter + sorted_offsets, mask=active, other=0)
            if USE_INT32_INDEX:
                sorted_index = sorted_index.to(tl.int32)
            sorted_offsets = row_offsets + sorted_index

        mid_values = tl.load(sorted_sequence + sorted_offsets, mask=active, other=0)
        if RIGHT:
            go_left = values_in < mid_values
        else:
            go_left = values_in <= mid_values

        high = tl.where(active & go_left, mid, high)
        low = tl.where(active & ~go_left, mid + 1, low)

    # NaN query lanes: every ``<`` / ``<=`` against a boundary is unordered
    # (False), which drives ``low`` to sequence_len anyway on CPU, but the GCU
    # compare does not reproduce that, so the lanes are pinned explicitly
    # (PyTorch semantics: NaN is greater than every element).
    low = tl.where(values_in == values_in, low, sequence_len)
    tl.store(out + offsets, low, mask=mask)


@triton.jit
def _searchsorted_scalar_kernel(
    sorted_sequence,
    out,
    value,
    sequence_len,
    LOG_SEQUENCE_LEN: tl.constexpr,
    RIGHT: tl.constexpr,
    OUT_INT32: tl.constexpr,
):
    """Binary search for a single scalar value: all-scalar operations, no arange / no idle vector lanes.

    Pointer offsets by dtype width are handled by Triton itself; mid is clamped to [0, sequence_len-1]
    to avoid out-of-bounds when low>=high, in which case the comparison result is masked by active.
    """
    low = tl.zeros((), dtype=tl.int32)
    high = low + sequence_len
    last = sequence_len - 1
    for _ in range(LOG_SEQUENCE_LEN):
        active = low < high
        mid = low + (high - low) // 2
        probe = tl.minimum(mid, last)
        mid_value = tl.load(sorted_sequence + probe)
        if RIGHT:
            go_left = value < mid_value
        else:
            go_left = value <= mid_value
        high = tl.where(active & go_left, mid, high)
        low = tl.where(active & ~go_left, mid + 1, low)
    # Same NaN rule as the vector kernel: a NaN query lands at sequence_len.
    low = tl.where(value == value, low, sequence_len)
    if OUT_INT32:
        tl.store(out, low.to(tl.int32))
    else:
        tl.store(out, low.to(tl.int64))


@triton.jit
def _searchsorted_scalar_count_kernel(
    sorted_sequence,
    out,
    value,
    sequence_len,
    RIGHT: tl.constexpr,
    OUT_INT32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Counting form of a single-scalar search on a 1D sorted 1D sequence.

    For a sorted sequence, ``left`` is the number of elements ``< value`` and
    ``right`` the number of elements ``<= value``, so one coalesced load plus a
    sum reduction replaces LOG2(n) dependent global loads.  Only valid when
    ``sequence_len <= BLOCK_SIZE`` (caller keeps n below
    ``_SCALAR_COUNT_MAX_ELEMENTS`` and picks BLOCK_SIZE accordingly).
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < sequence_len
    sequence_values = tl.load(sorted_sequence + offsets, mask=mask, other=0)
    # Masked-out lanes load 0, which would compare as "smaller than value"
    # whenever value > 0, so the predicate must be re-masked before reduction.
    if RIGHT:
        hits = (sequence_values <= value) & mask
    else:
        hits = (sequence_values < value) & mask
    count = tl.sum(hits.to(tl.int32), axis=0)
    count = tl.where(value == value, count, sequence_len)
    if OUT_INT32:
        tl.store(out, count)
    else:
        tl.store(out, count.to(tl.int64))


def _scalar_count_block_size(sequence_len: int) -> int:
    """Smallest power-of-two tile that covers ``sequence_len``."""
    return 1 << (sequence_len - 1).bit_length()


def _select_launch_config(dtype, numel, is_1d_sequence):
    """Picks (BLOCK_SIZE, num_warps) from element width and work size.

    The bottleneck is the non-contiguous gather on sorted_sequence (one per step):
    - 2-byte floats (fp16/bf16) move half a transaction per gather, so a larger block
      amortizes the fixed per-step cost; large cases measured 2048/w4 1.6x faster than 512/w4;
    - 4/8-byte elements get slower with larger blocks (double the gather transactions, more
      register pressure), so 512/w4 stays; on 1-D sequences 2-byte floats are best at 512/w2.
    All from a gcu400 kernel-only sweep (256/512/1024/2048 x w2/w4).
    """
    if dtype in (torch.float16, torch.bfloat16):
        if numel >= 65536:
            return 2048, 4
        if not is_1d_sequence:
            return 1024, 4
        return 512, 2
    return _GCU_BLOCK_SIZE, _GCU_NUM_WARPS


def _normalize_right(right: bool, side: str | None) -> bool:
    if side is None:
        return bool(right)
    if side == "left":
        if right:
            raise RuntimeError(
                "torch.searchsorted(): side and right can't be set to opposites, "
                "got side of left while right was True"
            )
        return False
    if side == "right":
        return True
    raise RuntimeError(
        f"torch.searchsorted(): side can only be 'left' or 'right' but got {side}"
    )


def _check_dtype(tensor: torch.Tensor, name: str):
    if tensor.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise NotImplementedError(
            f"searchsorted is not implemented for {name} dtype {tensor.dtype}"
        )


def _check_tensor_values_shape(sorted_sequence: torch.Tensor, values: torch.Tensor):
    if sorted_sequence.dim() == 0:
        raise RuntimeError(
            "torch.searchsorted(): boundaries tensor should be 1 dimension or "
            "the first N-1 dimensions of boundaries tensor and input value tensor "
            "must match"
        )
    if sorted_sequence.dim() == 1:
        return
    if values.dim() != sorted_sequence.dim() or (
        tuple(values.shape[:-1]) != tuple(sorted_sequence.shape[:-1])
    ):
        raise RuntimeError(
            "torch.searchsorted(): boundaries tensor should be 1 dimension or "
            "the first N-1 dimensions of boundaries tensor and input value tensor "
            "must match, but we got boundaries tensor "
            f"{list(sorted_sequence.shape)} and input value tensor {list(values.shape)}"
        )


def _check_scalar_values_shape(sorted_sequence: torch.Tensor):
    if sorted_sequence.dim() != 1:
        raise RuntimeError(
            "torch.searchsorted(): input value can be a scalar only when boundaries "
            "tensor dimension is 1, but we got boundaries tensor "
            f"dim({sorted_sequence.dim()}) and input value's dim(0) numel(1)"
        )


def _check_sorter(sorted_sequence: torch.Tensor, sorter: torch.Tensor | None):
    if sorter is None:
        return
    if tuple(sorter.shape) != tuple(sorted_sequence.shape):
        raise RuntimeError(
            "torch.searchsorted(): boundary and sorter must have the same size, "
            f"but got boundary tensor {list(sorted_sequence.shape)}"
            f"and got sorter tensor {list(sorter.shape)}"
        )
    if sorter.dtype != torch.int64:
        raise RuntimeError(
            "torch.searchsorted(): sorter must be a tensor of long dtype but got "
            f"dtype {sorter.dtype}"
        )
    if sorter.device != sorted_sequence.device:
        raise RuntimeError(
            "torch.searchsorted(): sorter and boundary tensors must be on the same device"
        )
    sequence_len = sorted_sequence.shape[-1]
    if sorter.numel() != 0 and (
        torch.any(sorter < 0).item() or torch.any(sorter >= sequence_len).item()
    ):
        raise RuntimeError("torch.searchsorted(): sorter index out of range")


def _prepare_out(
    values: torch.Tensor,
    out_int32: bool,
    out: torch.Tensor | None,
):
    out_dtype = torch.int32 if out_int32 else torch.int64
    if out is None:
        return torch.empty(values.shape, dtype=out_dtype, device=values.device)
    if out.dtype != out_dtype:
        raise RuntimeError(
            "torch.searchsorted(): output tensor's dtype is wrong, it can only be "
            "Int(int32) or Long(int64) depending on whether out_int32 flag is True"
        )
    if out.device != values.device:
        raise RuntimeError(
            "torch.searchsorted(): output tensor must be on the same device as input"
        )
    if tuple(out.shape) != tuple(values.shape):
        out.resize_(values.shape)
    return out


def _prepare_scalar_out(
    sorted_sequence: torch.Tensor,
    out_int32: bool,
    out: torch.Tensor | None,
):
    """out preparation for the scalar entry: 0-dim output, no values tensor needed."""
    out_dtype = torch.int32 if out_int32 else torch.int64
    if out is None:
        return torch.empty((), dtype=out_dtype, device=sorted_sequence.device)
    if out.dtype != out_dtype:
        raise RuntimeError(
            "torch.searchsorted(): output tensor's dtype is wrong, it can only be "
            "Int(int32) or Long(int64) depending on whether out_int32 flag is True"
        )
    if out.device != sorted_sequence.device:
        raise RuntimeError(
            "torch.searchsorted(): output tensor must be on the same device as input"
        )
    if tuple(out.shape) != ():
        out.resize_(())
    return out


def _searchsorted_scalar_impl(
    sorted_sequence: torch.Tensor,
    value,
    *,
    out_int32: bool,
    right: bool,
    side: str | None,
    sorter: torch.Tensor | None,
    out: torch.Tensor | None = None,
):
    """python scalar entry.

    The original first built a 0-dim device tensor with `torch.scalar_tensor(value, device=...)` (one H2D copy
    per call), then ran the vector kernel (many idle lanes). Passing the scalar directly as a
    kernel argument with all-scalar binary search removes the H2D and idle lanes; with sorter or ascend
    it still uses the generic tensor path.
    """
    right = _normalize_right(right, side)
    _check_dtype(sorted_sequence, "sorted_sequence")
    _check_scalar_values_shape(sorted_sequence)
    _check_sorter(sorted_sequence, sorter)
    if sorter is not None or runtime_device.vendor_name == "ascend":
        values = torch.scalar_tensor(value, device=sorted_sequence.device)
        return _searchsorted_impl(
            sorted_sequence,
            values,
            out_int32=out_int32,
            right=right,
            side=None,
            sorter=sorter,
            out=out,
        )

    out = _prepare_scalar_out(sorted_sequence, out_int32, out)
    sequence_len = sorted_sequence.shape[-1]
    if sequence_len == 0:
        out.zero_()
        return out
    sorted_sequence_contiguous = sorted_sequence.contiguous()
    kernel_out = out if out.is_contiguous() else torch.empty_like(out)
    out_int32_kernel = kernel_out.dtype == torch.int32
    with torch_device_fn.device(sorted_sequence.device):
        if sequence_len <= _SCALAR_COUNT_MAX_ELEMENTS:
            _searchsorted_scalar_count_kernel[(1,)](
                sorted_sequence_contiguous,
                kernel_out,
                value,
                sequence_len,
                RIGHT=right,
                OUT_INT32=out_int32_kernel,
                BLOCK_SIZE=max(_scalar_count_block_size(sequence_len), 16),
                num_warps=1,
            )
        else:
            _searchsorted_scalar_kernel[(1,)](
                sorted_sequence_contiguous,
                kernel_out,
                value,
                sequence_len,
                LOG_SEQUENCE_LEN=sequence_len.bit_length(),
                RIGHT=right,
                OUT_INT32=out_int32_kernel,
                num_warps=1,
            )
    if kernel_out is not out:
        out.copy_(kernel_out)
    return out


def _searchsorted_impl(
    sorted_sequence: torch.Tensor,
    values: torch.Tensor,
    *,
    out_int32: bool,
    right: bool,
    side: str | None,
    sorter: torch.Tensor | None,
    out: torch.Tensor | None = None,
):
    right = _normalize_right(right, side)
    _check_dtype(sorted_sequence, "sorted_sequence")
    _check_dtype(values, "values")
    _check_tensor_values_shape(sorted_sequence, values)
    _check_sorter(sorted_sequence, sorter)
    if values.device != sorted_sequence.device:
        raise RuntimeError(
            "torch.searchsorted(): sorted_sequence and values must be on the same device"
        )

    out = _prepare_out(values, out_int32, out)
    if values.numel() == 0:
        return out
    if sorted_sequence.shape[-1] == 0:
        out.zero_()
        return out

    sorted_sequence_contiguous = sorted_sequence.contiguous()
    values_contiguous = values.contiguous()
    sorter_contiguous = sorter.contiguous() if sorter is not None else None
    is_ascend = runtime_device.vendor_name == "ascend"
    if sorter_contiguous is not None and is_ascend:
        sorted_sequence_contiguous = torch.gather(
            sorted_sequence_contiguous, -1, sorter_contiguous
        )
        sorter_contiguous = None
    kernel_out = (
        out
        if out.is_contiguous()
        else torch.empty(out.shape, dtype=out.dtype, device=out.device)
    )

    sequence_len = sorted_sequence.shape[-1]
    values_per_row = values.shape[-1] if sorted_sequence.dim() != 1 else values.numel()
    is_1d_sequence = sorted_sequence.dim() == 1
    scalar_values = values.numel() == 1
    if is_ascend:
        block_size = (
            _ASCEND_BLOCK_SIZE
            if sorted_sequence.dtype.is_floating_point
            else _CUDA_BLOCK_SIZE
        )
        num_warps = _GCU_NUM_WARPS
    else:
        block_size, num_warps = _select_launch_config(
            sorted_sequence.dtype, values.numel(), is_1d_sequence
        )
        if scalar_values:
            # Single elements still using the vector kernel (integer dtype fallback) get a smaller block width
            block_size, num_warps = SCALAR_BLOCK_SIZE, _GCU_NUM_WARPS
    int32_index_safe = (
        values.numel() < torch.iinfo(torch.int32).max
        and sorted_sequence.numel() < torch.iinfo(torch.int32).max
    )
    use_int32_index = (
        (is_ascend or GCU400_TENSOR_INT32_INDEX) and int32_index_safe
    ) or (
        scalar_values
        and GCU400_SCALAR_INT32_INDEX
        and sequence_len <= torch.iinfo(torch.int32).max
    )

    with torch_device_fn.device(sorted_sequence.device):
        grid = (triton.cdiv(values.numel(), block_size),)
        _searchsorted_kernel[grid](
            sorted_sequence_contiguous,
            values_contiguous,
            (
                sorter_contiguous
                if sorter_contiguous is not None
                else sorted_sequence_contiguous
            ),
            kernel_out,
            values.numel(),
            values_per_row,
            sequence_len,
            LOG_SEQUENCE_LEN=sequence_len.bit_length(),
            RIGHT=right,
            HAS_SORTER=sorter_contiguous is not None,
            IS_1D_SEQUENCE=is_1d_sequence,
            USE_INT32_INDEX=use_int32_index,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

    if kernel_out is not out:
        out.copy_(kernel_out)
    return out


def searchsorted(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
):
    logger.debug("GEMS_ENFLAME SEARCHSORTED")
    return _searchsorted_impl(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
    )


def searchsorted_out(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
    out,
):
    logger.debug("GEMS_ENFLAME SEARCHSORTED_OUT")
    return _searchsorted_impl(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
        out=out,
    )


# ---------------- native reference bridge (torch_gcu topsaten searchsorted broken domains) -----------
# Probe results (2026-11-12, L600, TORCH_GCU_ENABLE_INT64_AND_UINT64=1, per-case subprocess):
#   - side="right" (right=False) -> gcu_utils.cpp:74 Check failed: 0,
#     TOPSATEN_STATUS_BAD_PARAM hard abort; right=True alone works, side="left" works;
#   - empty boundaries (sorted_sequence.shape[-1] == 0) -> BAD_PARAM hard abort (all dtypes);
#   - empty values (values.numel() == 0) -> BAD_PARAM hard abort;
#   - python float scalar (.Scalar / .Scalar_out overloads) -> NOT_SUPPORT (F64 scalar) hard
#     abort; int scalar works;
#   - .Tensor_out / .Scalar_out -> BAD_PARAM hard abort for every out layout, contiguous included;
#   - values containing NaN -> wrong values (NaN treated as less than all -> 0; CPU/FlagGems semantics put it last).
# These are C++ CHECK aborts, not error codes, so they cannot be caught in Python: the accuracy


def searchsorted_scalar(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
):
    logger.debug("GEMS_ENFLAME SEARCHSORTED_SCALAR")
    return _searchsorted_scalar_impl(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
    )


def searchsorted_scalar_out(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
    out,
):
    logger.debug("GEMS_ENFLAME SEARCHSORTED_SCALAR_OUT")
    return _searchsorted_scalar_impl(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
        out=out,
    )
