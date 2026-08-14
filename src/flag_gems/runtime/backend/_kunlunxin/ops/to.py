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
from typing import Optional

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

copy_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    # Large int32 -> fp16 tiles abort the current XPU LLVM lowering.
    # Limiting each cluster buffer to 512 bytes caps that specialization at 8K
    # lanes while retaining enough parallel CTAs for bandwidth-bound copies.
    buffer_size_limit=512,
)


@pointwise_dynamic(
    is_tensor=[
        True,
    ],
    promotion_methods=[(0, "DEFAULT")],
    config=copy_config,
)
@triton.jit
def _to_copy_func(x):
    return x


close_interleave_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseInterleave=True,
    buffer_size_limit=512,
)


@pointwise_dynamic(
    is_tensor=[
        True,
    ],
    promotion_methods=[(0, "DEFAULT")],
    config=close_interleave_config,
)
@triton.jit
def _to_copy_func_close_interleave(x):
    return x


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

    # A non-dense view cannot preserve its holes in a new allocation.  PyTorch's
    # device-copy reference materializes such a view as a contiguous tensor, so
    # use the canonical contiguous layout instead of retaining the source's
    # dimension order.
    return torch.empty(x.size(), **empty_kwargs)


@triton.jit
def _real_to_complex_contiguous_kernel(
    source, destination, n_elements, BLOCK: tl.constexpr
):
    # Iterate over the real-valued storage of the complex destination so stores
    # remain contiguous. A pair of stride-2 stores is pathologically slow on
    # XPU, while tl.interleave itself does not lower reliably on this backend.
    destination_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = destination_offsets < n_elements * 2
    source_offsets = destination_offsets // 2
    value = tl.load(source + source_offsets, mask=mask)
    value = tl.where(destination_offsets % 2 == 0, value, 0.0)
    tl.store(destination + destination_offsets, value, mask=mask)


@triton.jit
def _real_to_complex_strided_kernel(
    source,
    destination,
    shape,
    source_strides,
    destination_strides,
    n_elements,
    ndim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    logical_offset = offsets.to(tl.int64)
    source_offset = tl.zeros((BLOCK,), dtype=tl.int64)
    destination_offset = tl.zeros((BLOCK,), dtype=tl.int64)
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        coordinate = logical_offset % dim_size
        logical_offset //= dim_size
        source_offset += coordinate * tl.load(source_strides + dim)
        destination_offset += coordinate * tl.load(destination_strides + dim)

    value = tl.load(source + source_offset, mask=mask)
    tl.store(destination + destination_offset, value, mask=mask)
    tl.store(destination + destination_offset + 1, 0.0, mask=mask)


def _real_to_complex(
    x: torch.Tensor,
    target_dtype: torch.dtype,
    target_device: torch.device,
    target_memory_format: torch.memory_format,
) -> torch.Tensor:
    empty_kwargs = {"dtype": target_dtype, "device": target_device}
    if target_memory_format is torch.preserve_format:
        out = _allocate_preserve_format(x, empty_kwargs)
    else:
        out = torch.empty_like(x, memory_format=target_memory_format, **empty_kwargs)

    n_elements = x.numel()
    if n_elements == 0:
        return out

    # Passing the real view gives Triton a supported scalar pointer while still
    # writing the complex tensor's interleaved [real, imag] storage directly.
    destination = torch.view_as_real(out)
    block = 8192
    with torch_device_fn.device(x.device):
        if x.is_contiguous() and destination.is_contiguous():
            grid = (triton.cdiv(n_elements * 2, block),)
            _real_to_complex_contiguous_kernel[grid](
                x, destination, n_elements, BLOCK=block, num_warps=8
            )
        else:
            grid = (triton.cdiv(n_elements, block),)
            shape = torch.tensor(x.shape, dtype=torch.int64, device=x.device)
            source_strides = torch.tensor(
                x.stride(), dtype=torch.int64, device=x.device
            )
            destination_strides = torch.tensor(
                destination.stride()[:-1], dtype=torch.int64, device=x.device
            )
            _real_to_complex_strided_kernel[grid](
                x,
                destination,
                shape,
                source_strides,
                destination_strides,
                n_elements,
                ndim=x.ndim,
                BLOCK=block,
                num_warps=8,
            )
    return out


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

    # The XPU interleave pass corrupts either side of a BF16 conversion unless
    # explicitly disabled.  This applies when BF16 is the destination as well
    # as when it is the source (for example int16 -> bfloat16).
    if x.dtype == torch.bfloat16 or target_dtype == torch.bfloat16:
        to_dtype_fn = _to_copy_func_close_interleave
    else:
        to_dtype_fn = _to_copy_func

    # Same-device real-to-complex conversion cannot use XDNN's dtype-converting
    # copy kernel on all Kunlunxin runtimes. Write the interleaved complex storage
    # through a real view instead. Other complex conversions still use PyTorch.
    if x.dtype.is_complex or target_dtype.is_complex:
        if (
            not x.dtype.is_complex
            and target_dtype.is_complex
            and target_device == x.device
            and x.device.type != "cpu"
        ):
            return _real_to_complex(
                x, target_dtype, target_device, target_memory_format
            )
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

    # Direct int32 -> fp16/bf16 casts can abort during XPU LLVM lowering, while
    # XDNN's dtype-converting copy kernel is unavailable on some runtimes
    # (cudaErrorInvalidDeviceFunction).  Split the cast into two supported
    # Triton kernels instead.
    if x.dtype == torch.int32 and target_dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        intermediate = _allocate_preserve_format(
            x, {"dtype": torch.float32, "device": x.device}
        )
        _to_copy_func(x, out0=intermediate)
        x = intermediate

    # Direct casts from either 16-bit representation to BF16 are broken in the
    # current toolchain: fp16 -> bf16 is rejected by xpu3-elfconv, while
    # int16 -> bf16 turns almost every nonzero value into zero.  Route through
    # a supported wider representation for each source dtype.
    if x.dtype in (torch.float16, torch.int16) and target_dtype == torch.bfloat16:
        intermediate_dtype = torch.float32 if x.dtype == torch.float16 else torch.int32
        intermediate = _allocate_preserve_format(
            x, {"dtype": intermediate_dtype, "device": x.device}
        )
        _to_copy_func(x, out0=intermediate)
        x = intermediate

    logger.debug("GEMS_KUNLUNXIN TO_COPY")
    empty_kwargs = {"dtype": target_dtype, "device": target_device}

    if target_memory_format is torch.preserve_format:
        out = _allocate_preserve_format(x, empty_kwargs)
    else:
        out = torch.empty_like(x, memory_format=target_memory_format, **empty_kwargs)

    if out.element_size() == 8:
        os.environ["TRITONXPU_ELEMBYTES"] = "8"
        os.environ["TRITONXPU_BF16_FAST"] = "1"
        res = to_dtype_fn(x, out0=out)
        del os.environ["TRITONXPU_ELEMBYTES"]
        del os.environ["TRITONXPU_BF16_FAST"]
    else:
        os.environ["TRITONXPU_BF16_FAST"] = "1"
        res = to_dtype_fn(x, out0=out)
        del os.environ["TRITONXPU_BF16_FAST"]
    return res
