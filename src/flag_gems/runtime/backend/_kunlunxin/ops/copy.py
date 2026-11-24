import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic
from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    is_scatter_slice=True,
)


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


@triton.jit
def _copy_e8m0_to_fp32_kernel(src, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    exponent_bits = tl.load(src + offsets, mask=mask).to(tl.uint32) << 23
    values = exponent_bits.to(tl.float32, bitcast=True)
    tl.store(dst + offsets, values, mask=mask)


@triton.jit
def _copy_flat_kernel(
    src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr, NEED_MASK: tl.constexpr
):
    """Bounded-tile flat block-DMA copy for contiguous same-dtype tensors.

    The pointwise codegen widens the 1d tile to next_pow2(numel/12) (12-CTAs
    "XPU BLOCK_NUM" partition), which measures 0.073-0.16ms on a 16M-element
    copy but 1.08ms on bool (i1 bytes cannot reuse the wide-tile path).  A
    fixed bounded tile with a full grid keeps every access a contiguous
    block DMA.  NEED_MASK constexpr splits the always-true-mask case (the
    slow masked-memory path on XPU) from the true-tail case.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < n_elements
        tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets, mask=mask), mask=mask)
    else:
        tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets))


def _pick_flat_block(n_elements: int) -> int:
    if n_elements >= 2**19:
        return 65536
    if n_elements >= 2**16:
        return 32768
    if n_elements >= 2**13:
        return 8192
    if n_elements >= 2**10:
        return 4096
    return 1024


def _is_e8m0(tensor: torch.Tensor) -> bool:
    return hasattr(torch, "float8_e8m0fnu") and tensor.dtype is torch.float8_e8m0fnu


def _validate_triton_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    if dst.layout != torch.strided or src.layout != torch.strided:
        raise NotImplementedError("copy_ only supports strided tensors on Kunlunxin")
    if dst.is_quantized or src.is_quantized:
        raise NotImplementedError(
            "copy_ for quantized tensors is not supported on Kunlunxin"
        )
    if src.is_complex() or dst.is_complex():
        raise NotImplementedError(
            "copy_ for complex tensors is not supported on Kunlunxin"
        )


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

    if dst._is_zerotensor():
        raise RuntimeError("ZeroTensors are immutable. Call clone() before copy_.")
    if src._is_zerotensor():
        return dst.zero_()

    aliases = torch._C._is_alias_of(dst, src)
    if aliases and (
        dst.storage_offset() == src.storage_offset()
        and dst.stride() == src.stride()
        and dst.size() == src.size()
        and dst.dtype == src.dtype
        and dst.device == src.device
        and dst.is_conj() == src.is_conj()
        and dst.is_neg() == src.is_neg()
    ):
        return dst

    if dst.device != src.device:
        raise NotImplementedError("copy_ across devices is not supported on Kunlunxin")

    _validate_triton_copy(dst, src)
    logger.debug("GEMS_KUNLUNXIN COPY_")

    if src.shape != dst.shape:
        try:
            src.expand(dst.shape)
        except RuntimeError:
            try:
                broadcast_shape = torch.broadcast_shapes(dst.shape, src.shape)
            except RuntimeError as exc:
                raise RuntimeError(str(exc)) from exc
            raise RuntimeError(
                f"The broadcast shape {broadcast_shape} does not match destination shape {tuple(dst.shape)}"
            ) from None
    if dst.numel() == 0:
        return dst

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
    if _is_e8m0(expanded_src):
        if _is_e8m0(dst):
            overload = _copy_kernel.instantiate(expanded_src.ndim)
            overload(expanded_src.view(torch.uint8), out0=dst.view(torch.uint8))
            return dst
        if (
            dst.dtype is torch.float32
            and expanded_src.is_contiguous()
            and dst.is_contiguous()
        ):
            block_size = 256
            _copy_e8m0_to_fp32_kernel[(triton.cdiv(expanded_src.numel(), block_size),)](
                expanded_src.view(torch.uint8),
                dst,
                expanded_src.numel(),
                BLOCK_SIZE=block_size,
            )
            return dst
        raise NotImplementedError(
            "copy_ from float8_e8m0fnu only supports float8_e8m0fnu and contiguous float32 destinations on Kunlunxin"
        )

    if (
        not aliases
        and expanded_src.is_contiguous()
        and dst.is_contiguous()
        and expanded_src.dtype == dst.dtype
    ):
        n_elements = expanded_src.numel()
        item_size = expanded_src.element_size()
        if (
            item_size <= 4
            and (n_elements * item_size) % 4 == 0
            and n_elements * item_size >= 2**20
            and expanded_src.data_ptr() % 4 == 0
            and dst.data_ptr() % 4 == 0
        ):
            src_view = expanded_src.reshape(-1).view(torch.int32)
            dst_view = dst.reshape(-1).view(torch.int32)
            n32 = src_view.numel()
            block_size = _pick_flat_block(n32)
            _copy_flat_kernel[(triton.cdiv(n32, block_size),)](
                src_view,
                dst_view,
                n32,
                BLOCK_SIZE=block_size,
                NEED_MASK=(n32 % block_size != 0),
                num_warps=32,
                unroll_num=8,
                buffer_size_limit=1024,
            )
        else:
            block_size = _pick_flat_block(n_elements)
            _copy_flat_kernel[(triton.cdiv(n_elements, block_size),)](
                expanded_src,
                dst,
                n_elements,
                BLOCK_SIZE=block_size,
                NEED_MASK=(n_elements % block_size != 0),
                num_warps=32,
                unroll_num=8,
                buffer_size_limit=1024,
            )
        return dst

    # [KT2 2026-09-17] Strided / broadcast layouts: try the TLE DMA path before
    # the elementwise fallback below. The upstream rewrite dropped the
    # tle_copy branch here; on a transposed source (e.g. dim_compress's
    # permute().contiguous()) the elementwise `copy_slice` path below measures
    # ~28 ms for a [64, 512, 512] f16 transpose vs 247 us for tle_copy (which
    # moves it as an SDNN 2D row transfer / TMA tile). That cascaded into a
    # 55x regression on logsumexp's mid-dim reduce (regression_0917).
    # tle_copy self-checks layout/dtype/alignment and returns False when it
    # cannot express the copy -> falls through unchanged.
    if not aliases and tle_copy(expanded_src, dst):
        return dst

    overload = copy_slice.instantiate(expanded_src.ndim)
    if aliases:
        snapshot = torch.empty(dst.shape, dtype=src.dtype, device=src.device)
        overload(expanded_src, out0=snapshot)
        expanded_src = snapshot
    overload(expanded_src, out0=dst)
    return dst
