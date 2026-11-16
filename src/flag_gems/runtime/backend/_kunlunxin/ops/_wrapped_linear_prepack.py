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
#
# Kunlunxin (XPU) override of ``flag_gems.ops._wrapped_linear_prepack``.
#
# The generic kernel is dominated by its round-half-to-even helper: it is built
# from ``tl.floor`` + a float ``% 2.0`` parity test + nested ``tl.where`` and
# micro-benchmarks at 2048x1024 show the pure elementwise math costs ~650-880us
# while a load+int8-store of the same tensor costs ~50-65us.  On XPU the
# libdevice ``rint``/``nearbyint`` are also slow stubs (~580us) because they do
# not vectorize, so this override uses the IEEE "magic number" identity
#
#     round_half_even(q) == (q + 1.5 * 2**23) - 1.5 * 2**23
#
# which is exact for ``|q| <= 2**22`` and only requires two FP add/sub ops.  Any
# ``|q|`` above that bound already clamps to +-128 (the int8 range), so the trick
# is bit-exact for the whole int8 quantization domain (verified tie-by-tie against
# ``torch.quantize_per_tensor``).
#
# Two further XPU-specific facts shaped the layout below:
#   * Consecutive masked stores in one kernel accumulate their masks (the second
#     store's effective mask is ``mask1 & mask2``).  The generic empty-weight
#     kernel (``N == 0``) therefore drops the metadata store entirely because it
#     runs ``bias(mask<0)`` before ``metadata(mask<4)``.  The fast path below keeps
#     a single masked store or orders masks so that the previous mask is a
#     superset; the edge path uses one store per kernel.
#   * The packed buffer is ``[16B metadata][4N B bias][N*K B int8 weight]``, so the
#     metadata + bias region is a single contiguous float32 array of ``4 + N``
#     elements.  The fast path writes it with one masked store.
#
# Numeric semantics (round-half-to-even then ``+ zero_point`` then clamp to
# ``[-128, 127]``, plus the argument validation) are identical to the generic
# implementation.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_HEADER_BYTES = 16
_PACK_MAGIC = 260821.0
_PACK_VERSION = 1.0


@triton.jit
def _wlp_quantize(values, scale, zero_point):
    q = values / scale
    magic: tl.constexpr = 12582912.0
    q = (q + magic) - magic + zero_point
    return tl.minimum(tl.maximum(q, -128.0), 127.0)


@triton.jit
def _wlp_head_values(
    offsets,
    scale,
    zero_point,
    bias,
    stride_bias,
    N,
    PACK_MAGIC: tl.constexpr,
    PACK_VERSION: tl.constexpr,
):
    # metadata[0:4] = (scale, zero_point, magic, version); then bias[0:N].
    #
    # ``bias_idx`` is clamped into ``[0, N-1]``: XPU does not always predicate a
    # masked load whose result feeds a store with a *different* mask, so the
    # address itself must stay in bounds.  The masked lanes select ``other=0.0``
    # and are never stored.
    head_len = 4 + N
    bias_idx = tl.minimum(tl.maximum(offsets - 4, 0), N - 1)
    bias_vals = tl.load(
        bias + bias_idx * stride_bias,
        mask=(offsets >= 4) & (offsets < head_len),
        other=0.0,
    ).to(tl.float32)
    meta_vals = tl.where(
        offsets == 0,
        scale,
        tl.where(
            offsets == 1,
            zero_point,
            tl.where(offsets == 2, PACK_MAGIC, PACK_VERSION),
        ),
    )
    return tl.where(offsets < 4, meta_vals, bias_vals)


@triton.jit
def _wrapped_linear_prepack_compact_kernel(
    weight,
    weight_scale,
    weight_zero_point,
    bias,
    head,
    packed_weight,
    N,
    K,
    stride_bias,
    PACK_MAGIC: tl.constexpr,
    PACK_VERSION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Contiguous-weight fast path: one weight store, then one head store.

    Precondition (enforced on the host): ``weight.is_contiguous()`` and
    ``N * K >= 4 + N`` so the head store's accumulated mask (``N*K & (4+N)``)
    equals ``4 + N``.
    """
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    scale = tl.load(weight_scale).to(tl.float32)
    zero_point = tl.load(weight_zero_point).to(tl.float32)

    weight_numel = N * K
    # Affine ``weight + offsets`` keeps the block-DMA / vectorized load path.  The
    # load and the store below use the *same* mask, so XPU predicates it.
    values = tl.load(weight + offsets, mask=offsets < weight_numel, other=0.0)
    quantized = _wlp_quantize(values, scale, zero_point)
    # First masked store in the kernel: its mask is not modified.
    tl.store(
        packed_weight + offsets, quantized.to(tl.int8), mask=offsets < weight_numel
    )

    head_vals = _wlp_head_values(
        offsets, scale, zero_point, bias, stride_bias, N, PACK_MAGIC, PACK_VERSION
    )
    tl.store(head + offsets, head_vals, mask=offsets < 4 + N)


@triton.jit
def _wrapped_linear_prepack_weight_kernel(
    weight,
    weight_scale,
    weight_zero_point,
    packed_weight,
    N,
    K,
    stride_wn,
    stride_wk,
    PACK_MAGIC: tl.constexpr,
    PACK_VERSION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Strided-weight path: exactly one masked store, so no mask accumulation."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    scale = tl.load(weight_scale).to(tl.float32)
    zero_point = tl.load(weight_zero_point).to(tl.float32)

    weight_numel = N * K
    # Clamp indices so neither load can leave its allocation even when the
    # masked-load predication is dropped.
    weight_idx = tl.minimum(offsets, weight_numel - 1)
    rows = weight_idx // K
    cols = weight_idx % K
    values = tl.load(
        weight + rows * stride_wn + cols * stride_wk,
        mask=offsets < weight_numel,
        other=0.0,
    ).to(tl.float32)
    quantized = _wlp_quantize(values, scale, zero_point)
    tl.store(
        packed_weight + offsets, quantized.to(tl.int8), mask=offsets < weight_numel
    )


@triton.jit
def _wrapped_linear_prepack_head_kernel(
    weight_scale,
    weight_zero_point,
    bias,
    head,
    N,
    stride_bias,
    PACK_MAGIC: tl.constexpr,
    PACK_VERSION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Metadata + bias only (edge shapes / empty weight): one masked store."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    scale = tl.load(weight_scale).to(tl.float32)
    zero_point = tl.load(weight_zero_point).to(tl.float32)
    head_vals = _wlp_head_values(
        offsets, scale, zero_point, bias, stride_bias, N, PACK_MAGIC, PACK_VERSION
    )
    tl.store(head + offsets, head_vals, mask=offsets < 4 + N)


@triton.jit
def _wrapped_linear_prepack_meta_kernel(
    weight_scale,
    weight_zero_point,
    head,
    PACK_MAGIC: tl.constexpr,
    PACK_VERSION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Metadata only (``N == 0``): no bias argument at all.

    A zero-element ``bias`` has a null data pointer; XPU faults if such a tensor
    takes part in a masked load whose result is consumed by a store with another
    mask.  With ``N == 0`` there is nothing to write but the 4-word header.
    """
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    scale = tl.load(weight_scale).to(tl.float32)
    zero_point = tl.load(weight_zero_point).to(tl.float32)
    meta_vals = tl.where(
        offsets == 0,
        scale,
        tl.where(
            offsets == 1,
            zero_point,
            tl.where(offsets == 2, PACK_MAGIC, PACK_VERSION),
        ),
    )
    tl.store(head + offsets, meta_vals, mask=offsets < 4)


def _pick_block(total):
    if total <= 512:
        return 256
    if total <= 2048:
        return 1024
    if total <= 8192:
        return 4096
    return 8192


def _wrapped_linear_prepack(weight, weight_scale, weight_zero_point, bias):
    logger.debug("GEMS_KUNLUNXIN _WRAPPED_LINEAR_PREPACK")
    if weight.dtype != torch.float32:
        raise RuntimeError(f"Quantize only works on Float Tensor, got {weight.dtype}")
    if weight.ndim != 2:
        raise RuntimeError("fbgemm weight packing only packs matrices not vectors.")
    N, K = weight.shape
    if weight_scale.numel() != 1 or weight_zero_point.numel() != 1:
        raise RuntimeError("weight scale and zero point must contain one element")
    if bias.dtype != torch.float32 or bias.ndim != 1 or bias.numel() != N:
        raise RuntimeError("bias must be a float32 vector with out_channel elements")
    if not (
        weight.device == weight_scale.device == weight_zero_point.device == bias.device
    ):
        raise RuntimeError("all prepack inputs must be on the same device")

    weight_numel = N * K
    head_len = 4 + N
    bias_end = _HEADER_BYTES + 4 * N
    packed_numel = _HEADER_BYTES + 4 * N + weight_numel

    # torch.empty is pure allocation; no numeric work happens on the host.
    packed = torch.empty(packed_numel, dtype=torch.uint8, device=weight.device)
    quantized_weight = packed[bias_end:].view(torch.int8)
    full_head = packed[:bias_end].view(torch.float32)  # metadata + bias, 4+N floats

    stride_bias = bias.stride(0)
    with torch_device_fn.device(weight.device):
        if N == 0:
            # Nothing but the 4-word header exists; never touch the empty bias.
            meta_block = _pick_block(4)
            _wrapped_linear_prepack_meta_kernel[(triton.cdiv(4, meta_block),)](
                weight_scale,
                weight_zero_point,
                full_head,
                PACK_MAGIC=_PACK_MAGIC,
                PACK_VERSION=_PACK_VERSION,
                BLOCK_SIZE=meta_block,
            )
        elif weight.is_contiguous() and weight_numel >= head_len:
            # Fast path: weight store mask (N*K) superset of head mask (4+N).
            block_size = _pick_block(weight_numel)
            grid = (triton.cdiv(weight_numel, block_size),)
            _wrapped_linear_prepack_compact_kernel[grid](
                weight,
                weight_scale,
                weight_zero_point,
                bias,
                full_head,
                quantized_weight,
                N,
                K,
                stride_bias,
                PACK_MAGIC=_PACK_MAGIC,
                PACK_VERSION=_PACK_VERSION,
                BLOCK_SIZE=block_size,
            )
        else:
            if weight_numel > 0:
                block_size = _pick_block(weight_numel)
                grid = (triton.cdiv(weight_numel, block_size),)
                _wrapped_linear_prepack_weight_kernel[grid](
                    weight,
                    weight_scale,
                    weight_zero_point,
                    quantized_weight,
                    N,
                    K,
                    weight.stride(0),
                    weight.stride(1),
                    PACK_MAGIC=_PACK_MAGIC,
                    PACK_VERSION=_PACK_VERSION,
                    BLOCK_SIZE=block_size,
                )
            head_block = _pick_block(head_len)
            head_grid = (triton.cdiv(head_len, head_block),)
            _wrapped_linear_prepack_head_kernel[head_grid](
                weight_scale,
                weight_zero_point,
                bias,
                full_head,
                N,
                stride_bias,
                PACK_MAGIC=_PACK_MAGIC,
                PACK_VERSION=_PACK_VERSION,
                BLOCK_SIZE=head_block,
            )
    # ``full_head`` writes go through the kernels; no post-processing needed.
    return packed


def _patch_generic_entry():
    """Expose the override through the generic module object.

    ``SpecOpRegistrar`` only rebinds the top-level ``flag_gems.*`` namespace, but
    ``tests/test_wrapped_linear_prepack.py`` imports the function straight from
    ``flag_gems.ops._wrapped_linear_prepack``.  Rebind that module attribute so
    the functional gate exercises this kernel (same technique as
    ``_foreach_binary_tensor._patch_generic_wrapper``).
    """
    import sys

    gen = sys.modules.get("flag_gems.ops._wrapped_linear_prepack")
    if gen is not None:
        gen._wrapped_linear_prepack = _wrapped_linear_prepack


_patch_generic_entry()
