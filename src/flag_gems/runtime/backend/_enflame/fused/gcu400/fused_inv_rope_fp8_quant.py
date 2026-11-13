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

import importlib
import logging
import os
import sys
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# GCU Triton casts and stores float8_e4m3fn bit-exactly (verified on L600),
# so this backend-local copy declares fp8 support directly instead of
# consulting the generic get_device_capability() >= (9, 0) check, which
# reports (4, 1) on GCU and would silently degrade the op into a float32
# pseudo-quant mode (assert fp8_dtype == torch.float8_e4m3fn would fire).
SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn


logger = logging.getLogger(__name__)


def _get_tma_aligned_size(size: int, align: int) -> int:
    return ((size + align - 1) // align) * align


_GCU_I64_WORDS_ARE_VALUES = None


def _gcu_int64_words_are_values() -> bool:
    """One-time check of torch_gcu's int64 storage layout.

    Without TORCH_GCU_ENABLE_INT64_AND_UINT64=1 torch_gcu warns "GCU not
    support Long use Int replace" and stores int64 values as compact int32
    words: ``t.view(torch.int32)`` yields 2*numel words whose FIRST numel
    words are the element values (verified on torch_gcu 2.11.0, including
    negative values and 2**31-1).  A true little-endian int64 build would
    interleave low/high words instead; detect that once and fall back to a
    real cast, so the zero-copy path never silently misreads positions.
    """
    global _GCU_I64_WORDS_ARE_VALUES
    if _GCU_I64_WORDS_ARE_VALUES is None:
        probe = torch.tensor([1, 2], dtype=torch.int64, device="gcu")
        words = probe.view(-1).view(torch.int32)
        _GCU_I64_WORDS_ARE_VALUES = bool(
            torch.equal(words[: probe.numel()].to(torch.int64).cpu(), probe.cpu())
        )
    return _GCU_I64_WORDS_ARE_VALUES


def _positions_as_int32(positions: torch.Tensor) -> torch.Tensor:
    """Re-expose int64 positions as int32 without a device copy on GCU.

    torch_gcu (without TORCH_GCU_ENABLE_INT64_AND_UINT64=1) stores int64
    tensors as compact int32 words, and a scalar i64 tl.load feeding address
    arithmetic aborts the device (compiler issue triton_i64_narrow_load).
    Positions are provably < 2**31, so reinterpret the compact words
    directly: a pure metadata view with no kernel launch, which keeps the
    op single-launch even for benchmark timing closures that pass int64
    positions.  Falls back to a real cast when the storage layout is not
    the compact one and passes non-GCU / non-int64 tensors through.
    """
    if positions.dtype != torch.int64 or positions.device.type != "gcu":
        return positions
    if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64", "") in ("1", "true"):
        return positions
    if not positions.is_contiguous():
        return positions.to(torch.int32)
    if not _gcu_int64_words_are_values():
        return positions.to(torch.int32)
    flat = positions.view(-1)
    words = flat.view(torch.int32)[: flat.numel()]
    return words.view(positions.shape)


@triton.jit
def _fused_inv_rope_fp8_quant_per_head(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
):
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh
    qb_start = head_in_group * CHUNKS_PER_HEAD

    if pid_token >= num_tokens:
        if TMA_ALIGNED_SCALES:
            scale_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + head_in_group * scale_stride_k
            )
            tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
        else:
            block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
            qb_indices = qb_start + block_offsets
            scale_addrs = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    pos = tl.load(positions_ptr + pid_token).to(tl.int64)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)

    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    scales = block_absmax * (1.0 / fp8_max)
    if TMA_ALIGNED_SCALES:
        scales = tl.math.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scales), 1e-10))))

    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

    fp8_base = (
        fp8_ptr
        + g * fp8_stride_group
        + pid_token * fp8_stride_token
        + qb_start * QUANT_GROUP_SIZE
    )
    tl.store(fp8_base + offsets, x_quant)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = qb_start + block_offsets
    if TMA_ALIGNED_SCALES:
        scale_bits = scales.to(tl.int32, bitcast=True)
        ue8m0_bytes = (scale_bits >> 23) & 0xFF
        packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + head_in_group * scale_stride_k
        )
        tl.store(scale_addr, packed_val)
    else:
        scale_addrs = (
            scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
        )
        tl.store(scale_addrs, scales)


def fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    tma_aligned_scales: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton draft of DeepSeek-V4 fused inverse-RoPE + FP8 group quant.

    Args:
        o: [num_tokens, num_heads, head_dim]
        positions: [num_tokens]
        cos_sin_cache: [max_position, rope_dim] laid out as cos || sin

    Returns:
        o_fp8: [num_tokens, n_groups, heads_per_group * head_dim]
        o_scale: [num_tokens, n_groups, num_scale_blocks] or packed UE8M0 view
    """
    logger.debug("GEMS FUSED_INV_ROPE_FP8_QUANT")

    fp8_dtype = SUPPORTED_FP8_DTYPE if dtype is None else dtype
    assert fp8_dtype == torch.float8_e4m3fn, "only torch.float8_e4m3fn is supported"
    assert o.ndim == 3, "`o` must be [num_tokens, num_heads, head_dim]"
    assert positions.ndim == 1, "`positions` must be 1D"
    assert cos_sin_cache.ndim == 2, "`cos_sin_cache` must be 2D"
    assert o.stride(-1) == 1, "head_dim must be contiguous"
    assert positions.shape[0] == o.shape[0], "positions and o token count mismatch"

    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    # GCU: in-kernel int64 index loads poison the scalar load + address
    # arithmetic (compiler issue triton_i64_narrow_load); token positions are
    # provably < 2^31, so normalize to int32 on the host.  Zero-copy storage
    # reinterpretation (see _positions_as_int32): no extra kernel launch.
    if positions.dtype == torch.int64:
        positions = _positions_as_int32(positions)

    chunks_per_head = head_dim // quant_group_size
    if tma_aligned_scales:
        assert (
            chunks_per_head <= 4
        ), "packed UE8M0 path currently expects at most 4 scale blocks per head"

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    tma_aligned_t = _get_tma_aligned_size(num_tokens, 4)

    if tma_aligned_scales:
        scale_inner = (num_scale_blocks + 3) // 4
        scale_dtype = torch.int32
    else:
        scale_inner = num_scale_blocks
        scale_dtype = torch.float32

    finfo = torch.finfo(fp8_dtype)
    fp8_q = torch.empty((n_groups, num_tokens, d), dtype=fp8_dtype, device=o.device)
    scale = torch.empty(
        n_groups * scale_inner * tma_aligned_t,
        dtype=scale_dtype,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_t, 1, tma_aligned_t),
    )

    grid = (tma_aligned_t, n_groups * heads_per_group)
    _fused_inv_rope_fp8_quant_per_head[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_q,
        scale,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_group=fp8_q.stride(0),
        fp8_stride_token=fp8_q.stride(1),
        scale_stride_group=scale.stride(0),
        scale_stride_k=scale.stride(2),
        fp8_max=finfo.max,
        eps=eps,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=nope_dim % quant_group_size,
        HALF_ROPE=rope_dim // 2,
        TMA_ALIGNED_SCALES=tma_aligned_scales,
        num_warps=1,
        num_stages=1,
    )

    return fp8_q.transpose(0, 1), scale.transpose(0, 1)


# ----------------------------------------------------------------------------
# GCU compatibility overlays (installed at backend import time).
# ----------------------------------------------------------------------------


def _install_gcu_exp2_pow2_overlay() -> None:
    """Install a torch.exp2 overlay restoring exact powers of two on GCU.

    torch_gcu's device-side ``torch.exp2`` is a polynomial approximation
    whose result deviates from the exact value by up to ~14 ulp; at integer
    exponents it can land in the binade *below* ``2**k``, so UE8M0-style
    packing (``(bits >> 23) & 0xFF``) yields an exponent byte one too small
    (e.g. ``exp2(-33.0)`` -> 0x2EFFFFF7 instead of 0x2E800000, byte 93 vs 94).
    FlagGems kernels use the exact ``tl.math.exp2``, so op outputs are
    correct, but torch references computed on device (tests/benchmarks, and
    host-side E8M0 decodes such as ``fused_marlin_moe._pack_scale_e8m0``)
    silently disagree with the op at exactly those points.  Rebuild ``2**k``
    from the exponent field for integer-valued float32 exponents in the
    normal range and keep torch_gcu's approximation everywhere else.
    See compiler_issues/torch_gcu_exp2_inaccuracy.md; remove once torch_gcu
    implements exact powers of two in its device exp2 kernel.
    """
    if getattr(torch.exp2, "_flaggems_gcu_pow2_overlay", False):
        return

    original_exp2 = torch.exp2

    def exp2_pow2_exact(x, *, out=None):
        y = original_exp2(x)
        if (
            not isinstance(x, torch.Tensor)
            or x.dtype != torch.float32
            or x.device.type != "gcu"
        ):
            return y
        if torch.is_grad_enabled() and x.requires_grad:
            return y
        exponent = torch.floor(x)
        is_pow2_exponent = (x == exponent) & (exponent >= -126.0) & (exponent <= 127.0)
        code = (exponent.to(torch.int32) + 127) << 23
        exact = code.contiguous().view(torch.float32)
        result = torch.where(is_pow2_exponent, exact, y)
        if out is not None:
            return out.copy_(result)
        return result

    exp2_pow2_exact._flaggems_gcu_pow2_overlay = True
    torch.exp2 = exp2_pow2_exact


_install_gcu_exp2_pow2_overlay()


def _correct_package_fp8_gate() -> None:
    """Correct the package-level fp8 gate consumed by tests/benchmarks.

    GCU Triton casts and stores float8_e4m3fn bit-exactly (verified on
    L600), but the generic ``get_device_capability() >= (9, 0)`` gate does
    not know enflame and reports (4, 1) on GCU, so the package-level
    SUPPORTED_FP8_DTYPE that tests/benchmarks read resolves to float32 and
    every fp8-quant test silently skips.  runtime's SpecOpRegistrar only
    replaces functions, so this backend corrects the gate at import time.

    All per_token_group_quant_fp8 implementations bound to the gate are
    corrected together: the generic module that repo tests compose with
    this op as the unfused reference path, and this backend's arch copy
    that the package-level name resolves to.  The arch copy is imported by
    the arch registrar under the top-level name "gcu400.ops..." only after
    this module runs, so it is pre-imported here (idempotently reused by
    the registrar) to make sure the correction reaches the instance whose
    functions actually get bound.  Leaving any side at float32 would run
    one side of a comparison in a float32 pseudo-quant mode while the
    other runs real fp8.  Only float32 values are upgraded (idempotent,
    never downgrades a deliberate override).
    """
    if getattr(torch, "float8_e4m3fn", None) is None:
        return
    package = sys.modules.get("flag_gems")
    if package is not None and getattr(package, "SUPPORTED_FP8_DTYPE", None) is (
        torch.float32
    ):
        package.SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
    backend_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path.insert(0, backend_dir)
    try:
        importlib.import_module("gcu400.ops.per_token_group_quant_fp8")
    except Exception:  # noqa: BLE001 — best-effort pre-import, see docstring
        pass
    finally:
        try:
            sys.path.remove(backend_dir)
        except ValueError:
            pass
    for module_name in (
        "flag_gems.ops.per_token_group_quant_fp8",
        "flag_gems.runtime.backend._enflame.gcu400.ops.per_token_group_quant_fp8",
        "gcu400.ops.per_token_group_quant_fp8",
    ):
        quant_module = sys.modules.get(module_name)
        if quant_module is not None and getattr(
            quant_module, "SUPPORTED_FP8_DTYPE", None
        ) is (torch.float32):
            quant_module.SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn


_correct_package_fp8_gate()
