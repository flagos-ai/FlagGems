import logging
import os
from typing import Optional

import torch
import triton
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@pointwise_dynamic(
    is_tensor=[
        True,
    ],
    promotion_methods=[(0, "DEFAULT")],
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

    logger.debug("GEMS _TO_COPY")
    empty_kwargs = {"dtype": target_dtype, "device": target_device}

    if target_memory_format is torch.preserve_format:
        out = _allocate_preserve_format(x, empty_kwargs)
    else:
        out = torch.empty_like(x, memory_format=target_memory_format, **empty_kwargs)

    out = torch.empty_like(x, dtype=dtype, memory_format=memory_format)
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
