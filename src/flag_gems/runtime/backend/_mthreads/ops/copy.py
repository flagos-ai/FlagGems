import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
    ],
    key=["n_elements"],
)
@triton.jit
def copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, data, mask=mask)


def _can_use_triton(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.layout != torch.strided or src.layout != torch.strided:
        return False
    if dst.device != src.device:
        return False
    if dst.is_quantized or src.is_quantized:
        return False
    if src.is_complex() or dst.is_complex():
        return False
    return True


def _expand_like(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if src.shape == target_shape:
        return src
    return src.expand(target_shape)


def copy(
    template: torch.Tensor, src: torch.Tensor, *, non_blocking: Optional[bool] = False
):
    logger.debug("GEMS_MTHREADS COPY (functional)")
    out = torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )
    copy_(out, src, non_blocking=bool(non_blocking))
    return out


def copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    logger.debug("GEMS_MTHREADS COPY_")

    if isinstance(src, (int, float, bool)):
        src = torch.tensor(src, device=dst.device)
    elif not isinstance(src, torch.Tensor):
        raise TypeError("unsupport src type for copy_: ", type(src))

    if dst._is_zerotensor():
        raise RuntimeError("ZeroTensors are immutable. Call clone() before copy_.")
    if src._is_zerotensor():
        return dst.zero_()

    if torch._C._is_alias_of(dst, src):
        if (
            dst.storage_offset() == src.storage_offset()
            and dst.stride() == src.stride()
            and dst.size() == src.size()
            and dst.dtype == src.dtype
            and dst.device == src.device
            and dst.is_conj() == src.is_conj()
            and dst.is_neg() == src.is_neg()
        ):
            return dst
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if src.numel() > 2**31 - 1 or dst.numel() > 2**31 - 1:
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if not _can_use_triton(dst, src):
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if dst.numel() == 0:
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    try:
        broadcast_shape = torch.broadcast_shapes(dst.shape, src.shape)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    if torch.Size(broadcast_shape) != dst.shape:
        raise RuntimeError(
            f"The broadcast shape {broadcast_shape} does not match destination shape {tuple(dst.shape)}"
        )

    expanded_src = _expand_like(src, dst.shape)

    # Use optimized path for contiguous tensors with same dtype
    if (
        dst.is_contiguous()
        and expanded_src.is_contiguous()
        and dst.dtype == expanded_src.dtype
    ):
        n_elements = dst.numel()
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        with torch_device_fn.device(dst.device):
            copy_kernel[grid](expanded_src, dst, n_elements)
        return dst

    # Fallback for non-contiguous or dtype conversion cases
    return torch.ops.aten.copy_.default.redispatch(
        _FALLBACK_KEYSET, dst, src, non_blocking
    )
