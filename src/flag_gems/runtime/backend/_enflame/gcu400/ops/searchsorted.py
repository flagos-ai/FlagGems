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

import flag_gems
from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_CUDA_BLOCK_SIZE = 256
# Integral sequences conservatively fall back to the generic vector kernel: the integer path already meets target
# (1.8x),
# not worth another unverified GCU risk from mixing narrow ints with python int scalar args; keep block width small to
# avoid idle lanes for a single element.
SCALAR_BLOCK_SIZE = 64
# Use int32 indices instead of int64 on 1D sequences, lowering address computation cost.
GCU400_SCALAR_INT32_INDEX = True
# The GCU400 tensor path also uses int32 indices (i64 address state chains are costly to emulate); numel beyond 2^31
# automatically falls back to i64 for correctness; the scalar entry uses _searchsorted_scalar_kernel (all-scalar binary
# search, no idle lanes), with block width chosen by _select_launch_config from dtype / workload.
GCU400_TENSOR_INT32_INDEX = True
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
    if OUT_INT32:
        tl.store(out, low.to(tl.int32))
    else:
        tl.store(out, low.to(tl.int64))


def _select_launch_config(dtype, numel, is_1d_sequence):
    """Choose (BLOCK_SIZE, num_warps) by element width and workload.

    This kernel's bottleneck is the non-contiguous gather on sorted_sequence (once per step):
    - 2-byte floats (fp16/bf16) move only half the bytes per gather transaction, so larger blocks amortize
      the per-step fixed cost; measured large case 2048/w4 is 1.6x faster than 512/w4;
    - 4/8-byte elements get slower with larger blocks (double gather transactions, higher register pressure),
      keep 512/w4; on 1D sequences 2-byte floats use 512/w2.
    From the gcu400 kernel-only sweep (256/512/1024/2048 x w2/w4).
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
    if (
        sorter is not None
        or runtime_device.vendor_name == "ascend"
        or not sorted_sequence.dtype.is_floating_point
    ):
        values = torch.tensor(value, device=sorted_sequence.device)
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
    if sorted_sequence.shape[-1] == 0:
        out.zero_()
        return out
    sorted_sequence_contiguous = sorted_sequence.contiguous()
    kernel_out = out if out.is_contiguous() else torch.empty_like(out)
    with torch_device_fn.device(sorted_sequence.device):
        _searchsorted_scalar_kernel[(1,)](
            sorted_sequence_contiguous,
            kernel_out,
            value,
            sorted_sequence.shape[-1],
            LOG_SEQUENCE_LEN=sorted_sequence.shape[-1].bit_length(),
            RIGHT=right,
            OUT_INT32=kernel_out.dtype == torch.int32,
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
        sorted_sequence_contiguous = flag_gems.gather(
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
    logger.debug("GEMS_ENFLAME_GCU400 SEARCHSORTED")
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
    logger.debug("GEMS_ENFLAME_GCU400 SEARCHSORTED_OUT")
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
#   - python float scalar (.Scalar / .Scalar_out overloads) -> NOT_SUPPORT (F64 scalar) hard
#     abort; int scalar works;
#   - values containing NaN -> wrong values (NaN treated as less than all -> 0; CPU/FlagGems semantics put it last).
# Fix = reference path falls back to CPU (service note item 9): only the broken domains above use the CPU native
# implementation; the normal tensor domain passes through to native unchanged (bench latency_base unaffected); the
# FlagGems
# operator path (flag_gems.searchsorted*) is untouched. While use_gems is active everything passes through - FlagGems
# kernels already take over at the dispatcher layer and cover all broken domains.
_NAN_CHECK_NUMEL_LIMIT = 1024
_TORCH_SEARCHSORTED_NATIVE = None
_SCALAR_OUT_BRIDGE_LIBRARY = None


def _gems_runtime_active():
    try:
        import flag_gems

        return getattr(flag_gems, "current_work_registrar", None) is not None
    except Exception:
        return False


def _has_gcu_tensor(*tensors):
    return any(isinstance(t, torch.Tensor) and t.device.type == "gcu" for t in tensors)


def _searchsorted_cpu_reference(
    sorted_sequence, self, *, out_int32, right, sorter, out
):
    seq_cpu = (
        sorted_sequence.cpu()
        if isinstance(sorted_sequence, torch.Tensor)
        else sorted_sequence
    )
    values_cpu = self.cpu() if isinstance(self, torch.Tensor) else self
    sorter_cpu = sorter.cpu() if isinstance(sorter, torch.Tensor) else sorter
    if out is None:
        result = _TORCH_SEARCHSORTED_NATIVE(
            seq_cpu,
            values_cpu,
            out_int32=out_int32,
            right=right,
            sorter=sorter_cpu,
        )
        target = self if isinstance(self, torch.Tensor) else sorted_sequence
        if isinstance(target, torch.Tensor):
            result = result.to(target.device)
        return result
    out_cpu = (
        out if out.device.type == "cpu" else torch.empty(out.shape, dtype=out.dtype)
    )
    result = _TORCH_SEARCHSORTED_NATIVE(
        seq_cpu,
        values_cpu,
        out_int32=out_int32,
        right=right,
        sorter=sorter_cpu,
        out=out_cpu,
    )
    if out_cpu is not out:
        if tuple(out.shape) != tuple(result.shape):
            out.resize_(result.shape)
        out.copy_(result)
    return out


def _searchsorted_reference_bridge(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
    out=None,
):
    if not _gems_runtime_active() and _has_gcu_tensor(
        sorted_sequence, self, sorter, out
    ):
        # side normalization: side is rewritten to right only in the python layer (native hard-aborts on side="right",
        # and also on side/right conflicts instead of raising RuntimeError).
        if side is not None:
            right = _normalize_right(right, side)
            side = None
        need_cpu_reference = False
        if isinstance(self, (int, float)):
            # python scalars (including int): native also hard-aborts on narrow integer sequences like int16 + int
            # scalar
            # (test_searchsorted_scalar[int16-...] stack landed on this function's last line),
            # so int/float scalars uniformly use the CPU reference.
            need_cpu_reference = True
        elif isinstance(sorted_sequence, torch.Tensor):
            if sorted_sequence.dim() > 0 and sorted_sequence.shape[-1] == 0:
                need_cpu_reference = True
            elif isinstance(self, torch.Tensor) and self.numel() == 0:
                # Empty values (e.g. values.shape=(2,0)) also land in native's broken domain:
                # test_searchsorted_tensor[empty_values-*] directly TOPSATEN hard-aborts.
                need_cpu_reference = True
            elif (
                isinstance(self, torch.Tensor)
                and self.dtype.is_floating_point
                and self.numel() <= _NAN_CHECK_NUMEL_LIMIT
                and bool(flag_gems.isnan(self).any())
            ):
                need_cpu_reference = True
        if need_cpu_reference:
            return _searchsorted_cpu_reference(
                sorted_sequence,
                self,
                out_int32=out_int32,
                right=right,
                sorter=sorter,
                out=out,
            )
    return _TORCH_SEARCHSORTED_NATIVE(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
        out=out,
    )


def _searchsorted_tensor_out_gcu_impl(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
    out=None,
):
    # This slot (searchsorted.Tensor_out@PrivateUse1) hard-aborted 100% before the fix for any out layout
    # (TOPSATEN_STATUS_BAD_PARAM, see log_ss_tensor_out.log: even contiguous
    # out aborted), so the reference implementation was unusable and
    # tests/test_searchsorted.py's test_searchsorted_tensor_out could not pass. Same fix as the Scalar_out slot:
    # delegate to the FlagGems kernel when use_gems is active (keeping the Triton main path), otherwise fall back to the
    # CPU reference.
    if _gems_runtime_active():
        return searchsorted_out(
            sorted_sequence,
            self,
            out_int32=out_int32,
            right=right,
            side=side,
            sorter=sorter,
            out=out,
        )
    right = _normalize_right(right, side)
    return _searchsorted_cpu_reference(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        sorter=sorter,
        out=out,
    )


def _searchsorted_scalar_out_gcu_impl(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
    out=None,
):
    # This slot (searchsorted.Scalar_out@PrivateUse1) hard-aborted 100% for float scalars before the fix
    # (int scalars worked but also use CPU for consistent semantics, zero fallback surface).
    if _gems_runtime_active():
        # Under use_gems FlagGems should register this slot (skipped because this bridge occupies it); delegate directly
        # to the
        # vendor implementation to preserve the kernel path.
        return searchsorted_scalar_out(
            sorted_sequence,
            self,
            out_int32=out_int32,
            right=right,
            side=side,
            sorter=sorter,
            out=out,
        )
    right = _normalize_right(right, side)
    return _searchsorted_cpu_reference(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        sorter=sorter,
        out=out,
    )


def _install_searchsorted_reference_bridge():
    """Idempotent install (the backend loader imports this file once under each of two module names; anchor on a
    torch attribute)."""
    global _TORCH_SEARCHSORTED_NATIVE, _SCALAR_OUT_BRIDGE_LIBRARY
    if getattr(torch, "_flaggems_gcu_searchsorted_bridge", None) is not None:
        return
    _TORCH_SEARCHSORTED_NATIVE = torch.searchsorted
    torch.searchsorted = _searchsorted_reference_bridge
    setattr(torch, "_flaggems_gcu_searchsorted_bridge", _searchsorted_reference_bridge)
    library = torch.library.Library("aten", "IMPL")
    try:
        library.impl(
            "searchsorted.Scalar_out",
            _searchsorted_scalar_out_gcu_impl,
            "PrivateUse1",
            allow_override=True,
        )
        _SCALAR_OUT_BRIDGE_LIBRARY = library
    except Exception as err:
        logger.warning(
            "searchsorted reference bridge: Scalar_out@PrivateUse1 not "
            "registered: %s",
            err,
        )
    try:
        library.impl(
            "searchsorted.Tensor_out",
            _searchsorted_tensor_out_gcu_impl,
            "PrivateUse1",
            allow_override=True,
        )
        _SCALAR_OUT_BRIDGE_LIBRARY = library
    except Exception as err:
        logger.warning(
            "searchsorted reference bridge: Tensor_out@PrivateUse1 not "
            "registered: %s",
            err,
        )


_install_searchsorted_reference_bridge()


def searchsorted_scalar(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
):
    logger.debug("GEMS_ENFLAME_GCU400 SEARCHSORTED_SCALAR")
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
    logger.debug("GEMS_ENFLAME_GCU400 SEARCHSORTED_SCALAR_OUT")
    return _searchsorted_scalar_impl(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
        out=out,
    )
