import logging
from typing import Optional

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

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
    if torch.ops.aten.is_non_overlapping_and_dense(x):
        return torch.empty_strided(x.size(), x.stride(), **empty_kwargs)
    return torch.empty_like(x, memory_format=torch.preserve_format, **empty_kwargs)


def _fallback_to_copy(
    x: torch.Tensor,
    *,
    dtype: torch.dtype,
    layout,
    device: torch.device,
    pin_memory,
    non_blocking: bool,
    memory_format: torch.memory_format,
):
    return torch.ops.aten._to_copy.default.redispatch(
        _FALLBACK_KEYSET,
        x,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        memory_format=memory_format,
    )


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

    # PTPU can hold complex tensors, but the same-device ptpu cast path drops into
    # a backend copy_ implementation that does not handle complex. Stage through CPU
    # to avoid ptpu complex copy_/view_as_real gaps.
    if x.dtype.is_complex or target_dtype.is_complex:
        logger.debug("GEMS_SUNRISE _TO_COPY COMPLEX VIA CPU")
        cpu_x = x
        if x.device.type != "cpu":
            cpu_x = _fallback_to_copy(
                x,
                dtype=x.dtype,
                layout=layout,
                device=torch.device("cpu"),
                pin_memory=pin_memory,
                non_blocking=non_blocking,
                memory_format=target_memory_format,
            )
        cpu_res = _fallback_to_copy(
            cpu_x,
            dtype=target_dtype,
            layout=layout,
            device=torch.device("cpu"),
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=target_memory_format,
        )
        if target_device.type == "cpu":
            return cpu_res
        return _fallback_to_copy(
            cpu_res,
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
        return _fallback_to_copy(
            x,
            dtype=target_dtype,
            layout=layout,
            device=target_device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=target_memory_format,
        )

    logger.debug("GEMS_SUNRISE _TO_COPY")
    empty_kwargs = {"dtype": target_dtype, "device": target_device}

    if target_memory_format is torch.preserve_format:
        out = _allocate_preserve_format(x, empty_kwargs)
    else:
        out = torch.empty_like(x, memory_format=target_memory_format, **empty_kwargs)

    return _to_copy_func(x, out0=out)
