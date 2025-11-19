import logging
from typing import Optional

import torch
import triton

from flag_gems.utils import pointwise_dynamic

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
    logger.debug("GEMS _TO_COPY")

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

    empty_kwargs = {"dtype": target_dtype, "device": target_device}

    if target_memory_format is torch.preserve_format:
        out = _allocate_preserve_format(x, empty_kwargs)
    else:
        out = torch.empty_like(x, memory_format=target_memory_format, **empty_kwargs)

    if target_device == x.device and x.device.type != "cpu":
        return _to_copy_func(x, out0=out)

    out.copy_(x, non_blocking=non_blocking)
    return out
