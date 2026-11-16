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

# Kunlunxin (XPU) override of the in-place aten::feature_alpha_dropout_.
#
# This reuses the fused single-kernel scheme of the out-of-place override in
# ``_kunlunxin/ops/feature_alpha_dropout.py`` (see that module for the full
# rationale) but writes the result straight back into ``input``.  The generic
# ``flag_gems.ops.feature_alpha_dropout_`` keeps the two-kernel mask+apply pair
# and derives the feature index per element with a runtime-scalar integer
# division (``offsets // spatial_size``); pre-profiling on the benchmark matrix
# attributes 97.7% of the runtime (1128.4 ms / 63 calls) to
# ``feature_alpha_dropout_inplace_apply_kernel`` while the mask kernel is
# 0.04%.
#
# In-place semantics
# ------------------
# ATen's ``feature_alpha_dropout_`` mutates ``self`` and returns the *same*
# tensor (same ``data_ptr`` and ``stride``).  Each flat kernel below reads and
# writes the exact same lane (``X[off]`` -> ``Y[off]``), so passing the same
# pointer for both operands is race-free: there is no cross-lane dependence.
# A non-contiguous ``input`` cannot be addressed by a flat 1-D kernel, so the
# op runs on a contiguous working copy and then ``copy_``s the result back into
# ``input`` (preserving the original storage/stride/data_ptr), exactly like the
# generic in-place op.
#
# p == 1: ATen returns ``input * 0`` with IEEE signed-zero semantics.  XPU's
# multiply-by-zero flushes -0.0 to +0.0, so a dedicated (masked, in-place)
# kernel reconstructs ``copysign(0, x)`` from the sign bit and keeps NaN for
# non-finite inputs, matching the out-of-place override's fix.
import logging
import math

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset

from .feature_alpha_dropout import (
    _ALPHA,
    _FLAT_WARPS,
    _div_magic,
    _fad_flat_bulk_kernel,
    _fad_flat_div_kernel,
    _fad_flat_nonpow2_bulk_kernel,
    _fad_flat_tail_kernel,
    _flat_block,
    _is_pow2,
    _log2,
)

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["n", "zero"])
def _fad_inplace_zero_kernel(X, n, zero, IS_FLOAT: tl.constexpr, BLOCK: tl.constexpr):
    # p == 1, in place: ``Y = copysign(0, x)`` for finite x, NaN otherwise
    # (inf * 0 == NaN).  Masked so the final block never touches memory past
    # the tensor bounds.  ``zero`` is a runtime scalar so the inf/NaN branch
    # is not constant-folded away.  ATen's p==1 path is ``x * 0`` for any dtype
    # (it short-circuits before the ndim/float checks), so non-floating inputs
    # just get exact integer zeros (no signed-zero issue).
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    x = tl.load(X + off, mask=mask, other=0)
    if IS_FLOAT:
        vf = x.to(tl.float32)
        xb = vf.to(tl.int32, bitcast=True)
        signbit = (xb >> 31) & 1
        signed_zero = (signbit << 31).to(tl.float32, bitcast=True)
        finite = (xb & 0x7FFFFFFF) < 0x7F800000
        out = tl.where(finite, signed_zero, vf * zero)
        tl.store(X + off, out.to(x.dtype), mask=mask)
    else:
        tl.store(X + off, x * 0, mask=mask)


def _feature_alpha_dropout_zero_inplace(input):
    n = input.numel()
    block = _flat_block(n, True)
    grid = (triton.cdiv(n, block),)
    with torch_device_fn.device(input.device):
        _fad_inplace_zero_kernel[grid](
            input,
            n,
            0.0,
            IS_FLOAT=input.dtype.is_floating_point,
            BLOCK=block,
            num_warps=_FLAT_WARPS,
        )
    return input


def feature_alpha_dropout_(input, p=0.5, train=True):
    logger.debug("GEMS_KUNLUNXIN FEATURE_ALPHA_DROPOUT_")

    if not (0.0 <= p <= 1.0):
        raise RuntimeError(
            f"dropout probability has to be between 0 and 1, but got {p}"
        )
    if p == 0.0 or not train or input.numel() == 0:
        return input
    if p == 1.0:
        # ATen short-circuits ``p == 1`` to ``x * 0`` *before* the ndim/float
        # validation, so 1-D / 0-D / non-floating inputs are accepted here too.
        work = input if input.is_contiguous() else input.contiguous()
        _feature_alpha_dropout_zero_inplace(work)
        if work is not input:
            input.copy_(work)
        return input
    if input.ndim < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )
    if not input.dtype.is_floating_point:
        raise RuntimeError("feature_alpha_dropout_ only supports floating-point inputs")

    original_input = input
    work = input if input.is_contiguous() else input.contiguous()

    n_elements = work.numel()
    n_batch = work.shape[0]
    n_channels = work.shape[1]
    spatial_size = n_elements // (n_batch * n_channels)
    n_features = n_batch * n_channels

    scale = 1.0 / math.sqrt((_ALPHA * _ALPHA * p + 1.0) * (1.0 - p))
    shift = _ALPHA * scale * p
    dropped_value = _ALPHA * scale * (p - 1.0)
    # keep <=> (r & 0x7FFFFFFF) > K ; probability == 1 - floor(p * 2**31) / 2**31
    K = max(0, int(p * (1 << 31)) - 1)

    pow2 = _is_pow2(spatial_size)
    log2s = _log2(spatial_size) if pow2 else 0

    with torch_device_fn.device(work.device):
        philox_seed, philox_offset = philox_backend_seed_offset(n_features)

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
                work,
                work,
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
        else:
            n_full = (n_elements // block) * block
            if n_full > 0:
                if pow2:
                    _fad_flat_bulk_kernel[(n_full // block,)](
                        work,
                        work,
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
                        work,
                        work,
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
                    work,
                    work,
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

    if work is not original_input:
        original_input.copy_(work)
    return original_input


# Dedicated tests/benchmarks may resolve ``flag_gems.ops.<op>`` directly; the
# SpecOpRegistrar only rewrites top-level ``flag_gems.<op>`` globals.  Publish
# the backend implementation into the ``flag_gems.ops`` namespace too so every
# access path reaches this device-kernel implementation.
import sys as _sys

_generic_ops_module = _sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    setattr(_generic_ops_module, "feature_alpha_dropout_", feature_alpha_dropout_)
