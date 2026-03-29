import logging
from typing import Optional

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _copy_kernel(src):
    return src


def _can_use_triton(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.layout != torch.strided or src.layout != torch.strided:
        return False
    if dst.device != src.device:
        return False
    if dst.is_quantized or src.is_quantized:
        return False
    if src.dtype.is_complex != dst.dtype.is_complex:
        # Preserve PyTorch behaviour/warnings for complex<->real casts.
        return False
    return True


def _expand_like(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if src.shape == target_shape:
        return src
    return src.expand(target_shape)


def _to_physical_tensor(tensor: torch.Tensor) -> torch.Tensor:
    physical = tensor
    if physical.is_conj():
        physical = physical._conj()
    if physical.is_neg():
        physical = torch._neg_view(physical)
    return physical


def copy(
    template: torch.Tensor, src: torch.Tensor, *, non_blocking: Optional[bool] = False
):
    logger.debug("GEMS COPY (functional)")
    out = torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )
    copy_(out, src, non_blocking=bool(non_blocking))
    return out


def copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    if isinstance(src, (int, float, bool)):
        src = torch.tensor(src, device=dst.device)
    elif not isinstance(src, torch.Tensor):
        raise TypeError("unsupport src type for copy_: ", type(src))

    # this is the same as PyTorch's check
    if dst._is_zerotensor():
        raise RuntimeError("ZeroTensors are immutable. Call clone() before copy_.")
    if src._is_zerotensor():
        return dst.zero_()

    if torch._C._is_alias_of(dst, src):
        # Align with PyTorch: if metadata fully matches, this is a no-op.
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
        # Otherwise defer to PyTorch for well-defined semantics on overlapping writes.
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if src.numel() > 2**31 - 1 or dst.numel() > 2**31 - 1:
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    work_dst = dst
    work_src = src

    if dst.dtype.is_complex and src.dtype.is_complex:
        dst_is_conj = dst.is_conj()
        src_is_conj = src.is_conj()
        dst_is_neg = dst.is_neg()
        src_is_neg = src.is_neg()

        work_dst = _to_physical_tensor(dst)
        transformed_src = _to_physical_tensor(src)

        need_conj_transform = src_is_conj ^ dst_is_conj
        need_neg_transform = src_is_neg ^ dst_is_neg

        work_dst = torch.view_as_real(work_dst)
        work_src = torch.view_as_real(transformed_src)

        if need_conj_transform:
            work_src = work_src.clone()
            work_src[..., 1].neg_()

        if need_neg_transform:
            work_src = -work_src

    if not _can_use_triton(work_dst, work_src):
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if work_dst.numel() == 0:
        # Respect PyTorch behaviour: empty tensors should still validate broadcast.
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    logger.debug("GEMS COPY_")

    try:
        broadcast_shape = torch.broadcast_shapes(work_dst.shape, work_src.shape)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    if torch.Size(broadcast_shape) != work_dst.shape:
        raise RuntimeError(
            f"The broadcast shape {broadcast_shape} does not match destination shape {tuple(work_dst.shape)}"
        )

    expanded_src = _expand_like(work_src, work_dst.shape)

    overload = _copy_kernel.instantiate(expanded_src.ndim)
    overload(expanded_src, out0=work_dst)
    return dst
