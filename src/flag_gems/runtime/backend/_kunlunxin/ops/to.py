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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

@pointwise_dynamic(
    is_tensor=[
        True,
    ],
    promotion_methods=[(0, "DEFAULT")],
)
@triton.jit
def _to_copy_func(x):
    return x


@triton.jit
def _to_copy_contiguous_kernel(inp, out, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(inp + offsets, mask=mask)
    tl.store(out + offsets, values.to(out.dtype.element_ty), mask=mask)


@triton.jit
def _to_copy_to_complex_kernel(inp, out, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(inp + offsets, mask=mask)
    real_offsets = offsets * 2
    tl.store(out + real_offsets, values.to(out.dtype.element_ty), mask=mask)
    tl.store(out + real_offsets + 1, 0.0, mask=mask)


@triton.jit
def _to_copy_contiguous_to_bf16_kernel(
    inp, out, n_elements, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(inp + offsets, mask=mask).to(tl.float32)
    fp32_bits = values.to(tl.uint32, bitcast=True)
    rounding_bias = 0x7FFF + ((fp32_bits >> 16) & 1)
    bf16_bits = ((fp32_bits + rounding_bias) >> 16).to(tl.uint16)
    out_bits = out.to(tl.pointer_type(tl.uint16))
    tl.store(out_bits + offsets, bf16_bits, mask=mask)


def _to_copy_contiguous(inp, out, target_dtype):
    n_elements = inp.numel()
    if n_elements == 0:
        return out
    block_size = 256
    with torch_device_fn.device(inp.device):
        kernel = (
            _to_copy_contiguous_to_bf16_kernel
            if target_dtype == torch.bfloat16
            else _to_copy_contiguous_kernel
        )
        kernel[(triton.cdiv(n_elements, block_size),)](
            inp, out, n_elements, block_size
        )
    return out


def _to_copy_to_complex(inp, out):
    n_elements = inp.numel()
    if n_elements == 0:
        return out
    block_size = 256
    with torch_device_fn.device(inp.device):
        _to_copy_to_complex_kernel[(triton.cdiv(n_elements, block_size),)](
            inp, torch.view_as_real(out), n_elements, block_size
        )
    return out


close_interleave_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseInterleave=True,
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
    if x.dtype == torch.bfloat16:
        to_dtype_fn = _to_copy_func_close_interleave
    else:
        to_dtype_fn = _to_copy_func

    # The specialized kernel supports dense strided tensors only.
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
    if x.dtype == torch.bfloat16:
        to_dtype_fn = _to_copy_func_close_interleave
    else:
        to_dtype_fn = _to_copy_func

    if x.dtype.is_complex:
        raise NotImplementedError(
            "FlagGems to_copy does not support complex source tensors on Kunlunxin."
        )

    if target_device != x.device or (
        x.device.type == "cpu" and target_device.type == "cpu"
    ):
        raise NotImplementedError(
            "FlagGems to_copy only supports copies within one Kunlunxin device."
        )

    logger.debug("GEMS_KUNLUNXIN TO_COPY")
    empty_kwargs = {"dtype": target_dtype, "device": target_device}

    if target_memory_format is torch.preserve_format:
        out = _allocate_preserve_format(x, empty_kwargs)
    else:
        out = torch.empty_like(x, memory_format=target_memory_format, **empty_kwargs)

    if target_dtype.is_complex:
        if not x.is_contiguous() or not out.is_contiguous():
            raise NotImplementedError(
                "FlagGems to_copy only supports contiguous real-to-complex copies on Kunlunxin."
            )
        return _to_copy_to_complex(x, out)

    if x.is_contiguous() and out.is_contiguous():
        return _to_copy_contiguous(x, out, target_dtype)

    return to_dtype_fn(x, out0=out)
