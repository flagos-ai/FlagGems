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

# Kunlunxin (XPU) override of feature_alpha_dropout.
#
# Why an override exists
# ----------------------
# The generic implementation (`flag_gems.ops.feature_alpha_dropout`) launches
# two kernels: a tiny mask kernel that materialises one ``bool`` per feature
# (n_batch * n_channels) and an apply kernel over every element.  The apply
# kernel derives the feature index with three *runtime-scalar* integer
# divisions/mods per element::
#
#     channel_batch  = offsets // spatial_size
#     feature_offset = (channel_batch // n_channels) * n_channels
#                      + (channel_batch % n_channels)
#
# On XPU a non-constexpr integer division is scalarised, so the kernel runs at
# ~0.25 GB/s: profiling `harness/perf_ir/prof_feature_alpha_dropout_pre.log`
# attributes 99.4% of the runtime (2248.8 ms / 36 calls) to
# `feature_alpha_dropout_apply_kernel` while the mask kernel is 0.014%.  The
# benchmark therefore went 0.002x-0.005x on every large shape.
#
# Semantics (identical to the generic implementation / ATen)
# ---------------------------------------------------------
# One Bernoulli draw per feature (batch, channel) is shared by the whole
# spatial plane::
#
#     kept    : y = x * scale + shift
#     dropped : y = _ALPHA * scale * (p - 1)
#     scale   = 1 / sqrt((_ALPHA^2 * p + 1) * (1 - p))
#     shift   = _ALPHA * scale * p
#
# The arithmetic runs in f32 (the affine scalars are f32) and is rounded once
# on store, matching ATen's opmath-type computation.
#
# Kernel structure
# ----------------
# A single fused flat kernel replaces the mask+apply pair.  The flat element
# index of a contiguous [N, C, *spatial] tensor maps to its feature through a
# *constant shift* when spatial is a power of two:
# ``feature = offset >> log2(spatial)``; the codegen then knows every lane of a
# feature run shares one counter, draws one philox per feature and there are no
# gathers, no masks and no runtime divisions.  Non-power-of-two spatial uses a
# multiply-shift magic division (exact for the whole tensor).  Each program is
# one wide aligned tile (no mask) and a tail program covers the remainder with
# a single partially-masked store (the known-correct XPU pattern; the
# masked-multi-sub-store legalization bug only affects multi-store programs).
#
# RNG stream: `philox_backend_seed_offset(n_features)` is advanced with the same
# increment as the generic implementation (one counter per feature), so the
# global generator state moves identically and consecutive calls draw fresh
# masks.  The keep test uses the cheap `(r & 0x7FFFFFFF) > K` decode
# (K = int(p * 2**31) - 1) from `_kunlunxin/ops/alpha_dropout.py`: same keep
# probability ``1 - floor(p * 2**31) / 2**31`` without the u2f float chain.
#
# p == 1: ATen returns ``input * 0`` with IEEE signed-zero semantics.  XPU's
# native (and Triton) multiply-by-zero flushes -0.0 to +0.0, so a dedicated
# kernel reconstructs ``copysign(0, x)`` from the sign bit and keeps NaN for
# non-finite inputs (the CPU reference produces NaN there).
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)

# SELU alpha-dropout constant (positive), same value the generic op / tests use.
_ALPHA = 1.7580993408473766

# Flat-path tile.  Power-of-two spatial uses the constant-shift feature index
# and runs one wide aligned tile per program (BLOCK_up to 16K lanes measured
# fastest on the whole benchmark matrix); the non-power-of-two magic-division
# path keeps a smaller tile to bound the per-lane arithmetic.
_FLAT_POW2_BLOCK = 16384
_FLAT_NONPOW2_BLOCK = 2048
_FLAT_WARPS = 8


@libentry()
@triton.jit(
    do_not_specialize=[
        "a",
        "b",
        "c",
        "K",
        "philox_seed",
        "philox_offset",
    ]
)
def _fad_flat_bulk_kernel(
    X,
    Y,
    a,
    b,
    c,
    K,
    philox_seed,
    philox_offset,
    LOG2S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Fully in-bounds prefix, no mask.
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    feat = off >> LOG2S

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32) + feat.to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
    keep = (r0 & 0x7FFFFFFF) > K

    x = tl.load(X + off)
    y = tl.where(keep, x * a + b, c)
    tl.store(Y + off, y)


@libentry()
@triton.jit(
    do_not_specialize=[
        "a",
        "b",
        "c",
        "K",
        "philox_seed",
        "philox_offset",
    ]
)
def _fad_flat_nonpow2_bulk_kernel(
    X,
    Y,
    a,
    b,
    c,
    K,
    philox_seed,
    philox_offset,
    MAGIC: tl.constexpr,
    MSHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Same as `_fad_flat_bulk_kernel` but the feature index is derived with an
    # exact multiply-shift division: ``off // spatial == (off * MAGIC) >> MSHIFT``.
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    feat = (off * MAGIC) >> MSHIFT

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32) + feat.to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
    keep = (r0 & 0x7FFFFFFF) > K

    x = tl.load(X + off)
    y = tl.where(keep, x * a + b, c)
    tl.store(Y + off, y)


@libentry()
@triton.jit(do_not_specialize=["base", "N", "a", "b", "c", "K", "philox_seed", "philox_offset"])
def _fad_flat_tail_kernel(
    X,
    Y,
    base,
    N,
    a,
    b,
    c,
    K,
    philox_seed,
    philox_offset,
    POW2: tl.constexpr,
    LOG2S: tl.constexpr,
    MAGIC: tl.constexpr,
    MSHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Remainder [base, N): one partially-masked load/store per program.
    pid = tl.program_id(0)
    off = base + pid * BLOCK + tl.arange(0, BLOCK)
    if POW2:
        feat = off >> LOG2S
    else:
        feat = (off * MAGIC) >> MSHIFT

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32) + feat.to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
    keep = (r0 & 0x7FFFFFFF) > K

    mask = off < N
    x = tl.load(X + off, mask=mask, other=0.0)
    y = tl.where(keep, x * a + b, c)
    tl.store(Y + off, y, mask=mask)


@libentry()
@triton.jit(
    do_not_specialize=["spatial", "a", "b", "c", "K", "philox_seed", "philox_offset"]
)
def _fad_flat_div_kernel(
    X,
    Y,
    spatial,
    n_elements,
    a,
    b,
    c,
    K,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # Correctness fallback for spatial sizes so large that no int32
    # multiply-shift division fits: per-lane runtime division, single masked
    # load/store per program.
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    feat = off // spatial

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32) + feat.to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
    keep = (r0 & 0x7FFFFFFF) > K

    mask = off < n_elements
    x = tl.load(X + off, mask=mask, other=0.0)
    y = tl.where(keep, x * a + b, c)
    tl.store(Y + off, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["zero"])
def _fad_zero_kernel(X, Y, zero, BLOCK: tl.constexpr):
    # p == 1: reproduce ``x * 0`` with IEEE signed-zero semantics.  XPU's native
    # and Triton multiply-by-zero both flush -0.0 to +0.0, so rebuild the sign
    # bit: result = copysign(0, x) for finite x, NaN otherwise (inf * 0 = NaN).
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + off)
    vf = x.to(tl.float32)
    xb = vf.to(tl.int32, bitcast=True)
    signbit = (xb >> 31) & 1
    signed_zero = (signbit << 31).to(tl.float32, bitcast=True)
    finite = (xb & 0x7FFFFFFF) < 0x7F800000
    out = tl.where(finite, signed_zero, vf * zero)
    tl.store(Y + off, out.to(x.dtype))


def _div_magic(d, n_max):
    """Smallest (M, S) with (x * M) >> S == x // d for all 0 <= x < n_max,
    keeping M * n_max inside int32."""
    for s in range(10, 32):
        m = -(-(1 << s) // d)
        if m * n_max >= (1 << 31):
            continue
        if n_max * (d - 1) < (1 << s):
            return m, s
    return None


def _is_pow2(v):
    return v > 0 and (v & (v - 1)) == 0


def _log2(v):
    return v.bit_length() - 1


def _flat_block(numel, pow2):
    cap = _FLAT_POW2_BLOCK if pow2 else _FLAT_NONPOW2_BLOCK
    b = triton.next_power_of_2(min(numel, cap))
    if b < 64:
        b = 64
    return b


def _feature_alpha_dropout_zero(input):
    out = torch.empty_like(input)
    n = input.numel()
    if n == 0:
        return out
    block = _flat_block(n, True)
    grid = (triton.cdiv(n, block),)
    with torch_device_fn.device(input.device):
        _fad_zero_kernel[grid](input, out, 0.0, BLOCK=block, num_warps=_FLAT_WARPS)
    return out


def feature_alpha_dropout(input, p=0.5, train=True):
    logger.debug("GEMS_KUNLUNXIN FEATURE_ALPHA_DROPOUT")

    if not (0.0 <= p <= 1.0):
        raise RuntimeError(
            f"dropout probability has to be between 0 and 1, but got {p}"
        )
    if p == 0.0 or not train or input.numel() == 0:
        return input
    if p == 1.0:
        return _feature_alpha_dropout_zero(input)
    if input.ndim < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )
    if not input.dtype.is_floating_point:
        raise RuntimeError("feature_alpha_dropout only supports floating-point inputs")

    original_input = input
    input = input.contiguous()
    output = torch.empty_like(input)

    n_elements = input.numel()
    n_batch = input.shape[0]
    n_channels = input.shape[1]
    spatial_size = n_elements // (n_batch * n_channels)
    n_features = n_batch * n_channels

    scale = 1.0 / math.sqrt((_ALPHA * _ALPHA * p + 1.0) * (1.0 - p))
    shift = _ALPHA * scale * p
    dropped_value = _ALPHA * scale * (p - 1.0)
    # keep <=> (r & 0x7FFFFFFF) > K ; probability == 1 - floor(p * 2**31) / 2**31
    K = max(0, int(p * (1 << 31)) - 1)

    pow2 = _is_pow2(spatial_size)
    log2s = _log2(spatial_size) if pow2 else 0

    with torch_device_fn.device(input.device):
        philox_seed, philox_offset = philox_backend_seed_offset(n_features)

        # Flat elementwise kernel: the feature index is a per-lane constant
        # shift (power-of-two spatial) or a multiply-shift magic division.
        block = _flat_block(n_elements, pow2)
        magic, mshift = 0, 0
        if not pow2:
            found = _div_magic(spatial_size, n_elements)
            if found is not None:
                magic, mshift = found
        if not pow2 and not magic:
            # No int32 magic division fits (spatial huge): per-lane div.
            grid = (triton.cdiv(n_elements, block),)
            _fad_flat_div_kernel[grid](
                input,
                output,
                spatial_size,
                n_elements,
                scale,
                shift,
                dropped_value,
                K,
                philox_seed,
                philox_offset,
                BLOCK=block,
                num_warps=_FLAT_WARPS,
            )
            return _fad_finish(original_input, input, output)

        n_full = (n_elements // block) * block
        if n_full > 0:
            if pow2:
                _fad_flat_bulk_kernel[(n_full // block,)](
                    input,
                    output,
                    scale,
                    shift,
                    dropped_value,
                    K,
                    philox_seed,
                    philox_offset,
                    LOG2S=log2s,
                    BLOCK=block,
                    num_warps=_FLAT_WARPS,
                )
            else:
                _fad_flat_nonpow2_bulk_kernel[(n_full // block,)](
                    input,
                    output,
                    scale,
                    shift,
                    dropped_value,
                    K,
                    philox_seed,
                    philox_offset,
                    MAGIC=magic,
                    MSHIFT=mshift,
                    BLOCK=block,
                    num_warps=_FLAT_WARPS,
                )
        n_tail = n_elements - n_full
        if n_tail > 0:
            tblock = _flat_block(n_tail, pow2)
            _fad_flat_tail_kernel[(triton.cdiv(n_tail, tblock),)](
                input,
                output,
                n_full,
                n_elements,
                scale,
                shift,
                dropped_value,
                K,
                philox_seed,
                philox_offset,
                POW2=pow2,
                LOG2S=log2s,
                MAGIC=magic if magic else 1,
                MSHIFT=mshift,
                BLOCK=tblock,
                num_warps=_FLAT_WARPS,
            )

    return _fad_finish(original_input, input, output)


def _fad_finish(original_input, input, output):
    if original_input.is_contiguous():
        return output
    result = torch.empty_like(original_input)
    result.copy_(output)
    return result
