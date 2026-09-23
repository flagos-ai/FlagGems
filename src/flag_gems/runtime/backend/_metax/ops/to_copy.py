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

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

# Check if float8_e8m0fnu dtype is available in current PyTorch version
_FLOAT8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)

# Integer/boolean dtypes that produce illegal uitofp/sitofp to bf16/fp16 on MetaX.
# MetaX represents bf16 as i16 in LLVM IR, so uitofp i1/i8/i16/i32/i64 -> bf16
# generates an illegal instruction (uitofp requires a floating-point result type).
_INTEGER_DTYPES = frozenset(
    {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)
_PROBLEMATIC_TARGET_DTYPES = frozenset({torch.bfloat16, torch.float16})


@pointwise_dynamic(
    is_tensor=[
        True,
    ],
    promotion_methods=[(0, "DEFAULT")],
)
@triton.jit
def _to_copy_func(x):
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
    return torch.empty_like(x, memory_format=torch.preserve_format, **empty_kwargs)


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

    if target_device != x.device or (
        x.device.type == "cpu" and target_device.type == "cpu"
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

    # MetaX-specific: integer/bool -> bf16/fp16 generates illegal uitofp/sitofp
    # because MetaX represents bf16 as i16 in LLVM IR. Fall back to a two-step
    # conversion: int -> float32 -> target_dtype.
    if x.dtype in _INTEGER_DTYPES and target_dtype in _PROBLEMATIC_TARGET_DTYPES:
        logger.debug("GEMS_METAX TO_COPY (int->bf16/fp16 two-step)")
        # Step 1: convert integer to float32 using Triton kernel
        empty_kwargs_f32 = {"dtype": torch.float32, "device": target_device}
        if target_memory_format is torch.preserve_format:
            intermediate = _allocate_preserve_format(x, empty_kwargs_f32)
        else:
            intermediate = torch.empty_like(
                x, memory_format=target_memory_format, **empty_kwargs_f32
            )
        _to_copy_func(x, out0=intermediate)

        # Step 2: convert float32 to target bf16/fp16 using Triton kernel
        empty_kwargs = {"dtype": target_dtype, "device": target_device}
        if target_memory_format is torch.preserve_format:
            out = _allocate_preserve_format(x, empty_kwargs)
        else:
            out = torch.empty_like(
                x, memory_format=target_memory_format, **empty_kwargs
            )
        _to_copy_func(intermediate, out0=out)
        return out

    logger.debug("GEMS_METAX TO_COPY")
    empty_kwargs = {"dtype": target_dtype, "device": target_device}

    if target_memory_format is torch.preserve_format:
        out = _allocate_preserve_format(x, empty_kwargs)
    else:
        out = torch.empty_like(x, memory_format=target_memory_format, **empty_kwargs)

    return _to_copy_func(x, out0=out)
