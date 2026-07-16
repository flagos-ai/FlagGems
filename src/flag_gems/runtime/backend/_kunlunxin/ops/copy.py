import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_FLOAT8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    is_scatter_slice=True,
)


# @pointwise_dynamic(is_tensor=(True,), promotion_methods=[(0, "DEFAULT")])
# @triton.jit
# def copy(src):
#     return src


@pointwise_dynamic(
    is_tensor=(True,), promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def copy_slice(src):
    return src


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _copy_kernel(src):
    return src


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _copy_e8m0_to_float_kernel(src):
    src_i32 = src.to(tl.int32)
    # e8m0 stores the float32 exponent directly. Code zero represents 2^-127
    # and needs the corresponding subnormal encoding; 255 represents NaN.
    bits = src_i32 << 23
    bits = tl.where(src_i32 == 0, 1 << 22, bits)
    bits = tl.where(src_i32 == 255, 0x7FC00000, bits)
    return bits.to(tl.float32, bitcast=True)


def _can_use_triton(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.layout != torch.strided or src.layout != torch.strided:
        return False
    if dst.device != src.device:
        return False
    if dst.is_quantized or src.is_quantized:
        return False
    if src.is_complex() or dst.is_complex():
        # Triton on kunlunxin does not support complex dtypes; fall back to PyTorch.
        return False
    # A one-dimensional expanded scalar has stride zero. The pointwise kernel
    # supports this read, while aten.copy_ may dispatch an unavailable CUDA
    # kernel on Kunlunxin. Keep other non-contiguous layouts on the fallback.
    is_expanded_scalar = src.ndim == 1 and src.numel() > 0 and src.stride(0) == 0
    if not src.is_contiguous() and not is_expanded_scalar:
        return False
    return True


def _expand_like(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if src.shape == target_shape:
        return src
    return src.expand(target_shape)


def copy(
    template: torch.Tensor, src: torch.Tensor, *, non_blocking: Optional[bool] = False
):
    logger.debug("GEMS_KUNLUNXIN COPY")
    out = torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )
    copy_(out, src, non_blocking=bool(non_blocking))
    return out


def copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    if not isinstance(src, torch.Tensor):
        raise TypeError("src must be a Tensor")

    # this is the same as PyTorch's check
    if dst._is_zerotensor():
        raise RuntimeError("ZeroTensors are immutable. Call clone() before copy_.")
    if src._is_zerotensor():
        return dst.zero_()

    src_is_e8m0 = _FLOAT8_E8M0FNU is not None and src.dtype == _FLOAT8_E8M0FNU
    dst_is_e8m0 = _FLOAT8_E8M0FNU is not None and dst.dtype == _FLOAT8_E8M0FNU

    if src_is_e8m0 and dst_is_e8m0:
        # Triton cannot bind e8m0 tensors, but same-dtype copy is bitwise.
        return copy_(dst.view(torch.uint8), src.view(torch.uint8), non_blocking)

    if src_is_e8m0 and dst.dtype == torch.float32:
        if src.shape != dst.shape:
            src = src.expand(dst.shape)
        overload = _copy_e8m0_to_float_kernel.instantiate(src.ndim)
        overload(src.view(torch.uint8), out0=dst)
        return dst

    if src_is_e8m0 or dst_is_e8m0:
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

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

    if not _can_use_triton(dst, src):
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if dst.numel() == 0:
        # Respect PyTorch behaviour: empty tensors should still validate broadcast.
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    logger.debug("GEMS_KUNLUNXIN COPY_")

    try:
        broadcast_shape = torch.broadcast_shapes(dst.shape, src.shape)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    if torch.Size(broadcast_shape) != dst.shape:
        raise RuntimeError(
            f"The broadcast shape {broadcast_shape} does not match destination shape {tuple(dst.shape)}"
        )

    expanded_src = _expand_like(src, dst.shape)

    overload = _copy_kernel.instantiate(expanded_src.ndim)
    overload(expanded_src, out0=dst)
    return dst
