# Copyright 2026 FlagOS Contributors

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutogradNonFunctional
)


@triton.jit
def _as_strided_scatter_kernel(
    src,
    out,
    sizes,
    strides,
    n_elements,
    storage_offset,
    BLOCK_SIZE: tl.constexpr,
    NDIM: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(src + offsets, mask=mask)

    remaining = offsets.to(tl.int64)
    target = tl.full(offsets.shape, storage_offset, tl.int64)
    for reverse_dim in tl.static_range(NDIM):
        dim = NDIM - reverse_dim - 1
        dim_size = tl.load(sizes + dim)
        dim_stride = tl.load(strides + dim)
        target += (remaining % dim_size) * dim_stride
        remaining //= dim_size

    tl.store(out + target, values, mask=mask)


def _native_as_strided_scatter(self, src, size, stride, storage_offset):
    return torch.ops.aten.as_strided_scatter.default.redispatch(
        _FALLBACK_KEYSET, self, src, size, stride, storage_offset
    )


def _can_use_triton(self, src, size, stride, storage_offset):
    if self.device.type != "cuda" or src.device != self.device:
        return False
    if not self.is_contiguous() or self.storage_offset() != 0:
        return False
    if self.dtype != src.dtype or self.is_complex() or self.is_quantized:
        return False
    if len(size) != len(stride) or len(size) == 0:
        return False
    if any(int(value) < 0 for value in size) or any(int(value) < 0 for value in stride):
        return False
    if storage_offset is None:
        storage_offset = 0
    try:
        target = torch.as_strided(self, size, stride, int(storage_offset))
    except RuntimeError:
        return False
    return torch._debug_has_internal_overlap(target) == 0


def as_strided_scatter(
    self: torch.Tensor,
    src: torch.Tensor,
    size,
    stride,
    storage_offset=None,
) -> torch.Tensor:
    logger.debug("GEMS AS_STRIDED_SCATTER")
    if not _can_use_triton(self, src, size, stride, storage_offset):
        return _native_as_strided_scatter(self, src, size, stride, storage_offset)

    storage_offset = 0 if storage_offset is None else int(storage_offset)
    src = src.contiguous()
    expected_numel = 1
    for value in size:
        expected_numel *= int(value)
    if src.numel() != expected_numel:
        return _native_as_strided_scatter(self, src, size, stride, storage_offset)

    out = self.clone(memory_format=torch.preserve_format)
    if src.numel() == 0:
        return out
    sizes = torch.tensor(size, dtype=torch.int64, device=self.device)
    strides = torch.tensor(stride, dtype=torch.int64, device=self.device)
    grid = (triton.cdiv(src.numel(), 256),)
    _as_strided_scatter_kernel[grid](
        src,
        out,
        sizes,
        strides,
        src.numel(),
        storage_offset,
        BLOCK_SIZE=256,
        NDIM=len(size),
    )
    return out


def as_strided_scatter_out(
    self: torch.Tensor,
    src: torch.Tensor,
    size,
    stride,
    storage_offset=None,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS AS_STRIDED_SCATTER_OUT")
    result = as_strided_scatter(self, src, size, stride, storage_offset)
    out.resize_as_(result)
    out.copy_(result)
    return out
