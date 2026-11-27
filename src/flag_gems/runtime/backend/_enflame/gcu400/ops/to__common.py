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
from typing import Optional

import torch
import triton
import triton.language as tl

# NOTE: this import must stay absolute.  pointwise_dynamic's codegen copies the
# module-level ImportFrom statements of this file into its generated kernel
# wrapper (utils/pointwise_dynamic.py _collect_jit_deps) and drops the relative
# level, so `from .copy import ...` would be rewritten to a stdlib `copy`
# import inside the generated file and crash it.
from flag_gems.runtime.backend._enflame.gcu400.ops.copy import (
    _flat_copy,
    _flat_copy_plan,
)
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

# Check if float8_e8m0fnu dtype is available in current PyTorch version
_FLOAT8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)


@pointwise_dynamic(
    is_tensor=[
        True,
    ],
    promotion_methods=[(0, "DEFAULT")],
)
@triton.jit
def _to_copy_func(x):
    return x


# Flat grid-stride cast kernel for dense contiguous casts.
#
# The generic pointwise path launches the nd-tile block-pointer (DTE) kernel
# with the top-level ENFLAME codegen config (max tile 4096, grid capped at 12
# CTAs), which under-parallelizes a pure 1R1W stream by 4x on the 48-SPU
# gcu400: measured 0.25-0.33x vs native aten::_to_copy at 2^24 elements.  The
# flat kernel below follows the gcu400 copy-stream family recipe (see
# ops/copy.py and the _indices_copy/deg2rad cases): plain offset
# load -> cast -> store on the synchronous port, fixed 48-CTA grid-stride,
# EVEN specialization to drop tail masks.  Plain-offset access also avoids
# the DTE tail-overrun MMU hazard documented in ops/copy.py.
#
# Config ladder (measured on GCU400/L600, kernel-mode do_bench median):
#   n <= 2^14: keep the generic pointwise kernel (single-tile DTE wins at
#              tiny sizes; both sit on the ~3.3us launch floor)
#   n <= 2^22: BLOCK 8192, nw 4 for >=4-byte sources / nw 2 for <=2-byte
#              sources (2^20 f32->bf16 1.51x, bf16->f32 1.55x, i64->f32 1.21x)
#   n >  2^22: BLOCK 65536, nw 2 (2^30 f32->bf16 1.42x, bf16->f32 1.50x,
#              u8->f32 1.08x, f32<->i32 1.01x); i64 involved -> nw 4
#              (i64->f32 0.92x, f32->i64 1.00x, i32->i64 0.99x: i64 reads
#              saturate the same ~2.5TB/s every config reaches)
_FLAT_CAST_GRID = 48
_FLAT_CAST_TINY_MAX = 1 << 14
_FLAT_CAST_MID_MAX = 1 << 22
_FLAT_CAST_BLOCK_MID = 8192
_FLAT_CAST_BLOCK_WIDE = 65536
# Offsets are int32, so the last tile must still fit in an int32.
_FLAT_CAST_MAX_ELEMS = 2**31 - 1 - _FLAT_CAST_BLOCK_WIDE

# Triton-representable element dtypes routed to the flat kernel; everything
# else (bool, fp8, ...) keeps the generic pointwise path.
_FLAT_CAST_DTYPES = frozenset(
    (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    )
)


@triton.jit
def _flat_cast_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in tl.range(pid, num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if EVEN:
            value = tl.load(src_ptr + offsets)
            tl.store(dst_ptr + offsets, value.to(dst_ptr.dtype.element_ty))
        else:
            mask = offsets < n_elements
            value = tl.load(src_ptr + offsets, mask=mask, other=0)
            tl.store(dst_ptr + offsets, value.to(dst_ptr.dtype.element_ty), mask=mask)


def _flat_cast(x: torch.Tensor, out: torch.Tensor) -> bool:
    """Launch the flat cast kernel; return False when the case is not eligible."""
    if not x.is_contiguous() or not out.is_contiguous():
        return False
    if x.dtype == out.dtype:
        plan = _flat_copy_plan(out, x)
        return plan is not None and _flat_copy(out, x, plan)
    if x.dtype not in _FLAT_CAST_DTYPES or out.dtype not in _FLAT_CAST_DTYPES:
        return False
    n_elements = x.numel()
    if n_elements == 0 or n_elements > _FLAT_CAST_MAX_ELEMS:
        return False
    src_esize = x.element_size()
    dst_esize = out.element_size()
    is_wide = src_esize == 8 or dst_esize == 8
    # int64-involved casts bypass the tiny-size pointwise fallback: the generic
    # rank-1 pointwise kernel makes the gcu400 backend compiler abort during
    # SelectionDAG lowering on e.g. int64->int8 (llc "DAG->DAG Pattern
    # Instruction Selection" fatal error, pre-existing in the generic path).
    if n_elements <= _FLAT_CAST_TINY_MAX and not is_wide:
        return False
    if n_elements <= _FLAT_CAST_MID_MAX:
        block_size = _FLAT_CAST_BLOCK_MID
        num_warps = 2 if src_esize <= 2 else 4
    else:
        block_size = _FLAT_CAST_BLOCK_WIDE
        num_warps = 4 if is_wide else 2
    grid = (min(_FLAT_CAST_GRID, triton.cdiv(n_elements, block_size)),)
    _flat_cast_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        GRID_SIZE=_FLAT_CAST_GRID,
        EVEN=(n_elements % block_size == 0),
        num_warps=num_warps,
    )
    return True


def _resolve_dtype(x: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is None:
        return x.dtype
    if isinstance(dtype, torch.dtype):
        return dtype
    raise TypeError(f"Unsupported dtype argument type: {type(dtype)!r}")


def _resolve_device(x: torch.Tensor, device: Optional[torch.device]) -> torch.device:
    if device is None:
        return x.device
    return torch.device(device)


def _normalize_memory_format(
    memory_format: Optional[torch.memory_format],
) -> torch.memory_format:
    if memory_format is None:
        return torch.preserve_format
    return memory_format


def _allocate_preserve_format(x: torch.Tensor, empty_kwargs: dict) -> torch.Tensor:
    """Recreate tensor storage while honoring preserve_format semantics."""
    if torch.ops.aten.is_non_overlapping_and_dense(x):
        return torch.empty_strided(x.size(), x.stride(), **empty_kwargs)
    # Fall back to PyTorch's best-effort layout suggestion when stride replication is unsafe.
    return torch.empty_like(x, memory_format=torch.preserve_format, **empty_kwargs)


# func: _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None,
#   bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
def to_copy(
    x,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
):
    # We only implement the dense strided kernel today; all other layouts fall back to PyTorch.
    if (layout is not None and layout != torch.strided) or x.layout != torch.strided:
        raise NotImplementedError(
            "FlagGems to_copy currently supports strided tensors only."
        )
    if pin_memory is not None:
        raise NotImplementedError(
            "FlagGems to_copy does not yet support pin_memory=True."
        )
    if x.is_quantized:
        raise NotImplementedError(
            "Quantized tensors are not supported in FlagGems to_copy yet."
        )

    target_dtype = _resolve_dtype(x, dtype)
    target_device = _resolve_device(x, device)
    target_memory_format = _normalize_memory_format(memory_format)

    # Triton does not support complex dtypes; fall back to PyTorch.
    if x.dtype.is_complex or target_dtype.is_complex:
        return torch.ops.aten._to_copy.default.redispatch(
            _FALLBACK_KEYSET,
            x,
            dtype=target_dtype,
            layout=layout,
            device=target_device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=target_memory_format,
        )

    # Triton does not support float8_e8m0fnu dtypes; fall back to PyTorch.
    if _FLOAT8_E8M0FNU is not None and (
        x.dtype == torch.float8_e8m0fnu or target_dtype == torch.float8_e8m0fnu
    ):
        return torch.ops.aten._to_copy.default.redispatch(
            _FALLBACK_KEYSET,
            x,
            dtype=target_dtype,
            layout=layout,
            device=target_device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=target_memory_format,
        )

    # int64 -> int8/uint8 casts abort the gcu400 backend compiler during
    # SelectionDAG lowering: the i64-lane truncate-to-i8 value chain is
    # unselectable in every kernel shape tried (pointwise nd-tile and flat
    # plain-offset alike, all BLOCK/num_warps).  Fall back to PyTorch for
    # these two dtype pairs instead of crashing (see compiler_problems/P05).
    if x.dtype == torch.int64 and target_dtype in (torch.int8, torch.uint8):
        return torch.ops.aten._to_copy.default.redispatch(
            _FALLBACK_KEYSET,
            x,
            dtype=target_dtype,
            layout=layout,
            device=target_device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=target_memory_format,
        )

    if target_device != x.device or (
        x.device.type == "cpu" and target_device.type == "cpu"
    ):
        # Device transfer (d2h/h2d etc.) relies on PyTorch's implementation.
        return torch.ops.aten._to_copy.default.redispatch(
            _FALLBACK_KEYSET,
            x,
            dtype=target_dtype,
            layout=layout,
            device=target_device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=target_memory_format,
        )

    logger.debug("GEMS_ENFLAME TO_COPY")
    empty_kwargs = {"dtype": target_dtype, "device": target_device}

    if target_memory_format is torch.preserve_format:
        out = _allocate_preserve_format(x, empty_kwargs)
    else:
        out = torch.empty_like(x, memory_format=target_memory_format, **empty_kwargs)

    if _flat_cast(x, out):
        return out
    return _to_copy_func(x, out0=out)
