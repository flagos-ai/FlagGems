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

import logging
from typing import List

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
BLOCK_SIZE_SMALL = 1024
BLOCK_SIZE_LARGE = 8192

# XPU codegen knobs for the fused kernel: without them the tile body is
# lowered lane-by-lane with an open vectorization policy.
_LARGE_LAUNCH = dict(
    num_warps=32,
    buffer_size_limit=8192,
    unroll_num=16,
    isCloseVectorization=False,
)
_SMALL_LAUNCH = dict(num_warps=4)
_LARGE_POINTWISE_CONFIG = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
    kunlunAutoGrid=True,
)

# Value at/above which the fp32 product rounds to inf in the output dtype.
# fp32 itself can only overflow to inf at exactly inf, so the threshold is
# +inf (the exponent test already covers that case).
_OVERFLOW_THRESH = {
    torch.float16: 65520.0,
    torch.bfloat16: 3.402823669209385e38,  # 2**128 (bf16 round-to-inf bound)
    torch.float32: float("inf"),
}

REDUCE_BLOCK = 16384


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, "INT_TO_FLOAT"), (0, "INT_TO_FLOAT")],
    num_outputs=2,
    config=_LARGE_POINTWISE_CONFIG,
)
@triton.jit
def _unscale_flag_func(x, scale, thresh):
    s = scale.to(tl.float32)
    xf = x.to(tl.float32)

    # finite = 1 when the fp32 exponent field is not all ones (see the fused
    # kernel below for the rationale of the integer-only 0/1 selector).
    bits = xf.to(tl.int32, bitcast=True)
    expo = (bits >> 23) & 0xFF
    finite = ((expo - 255) >> 31) & 1
    ff = finite.to(tl.float32)
    y = xf * (ff * s + (1.0 - ff))

    # bad mirrors the fused kernel's post-store check: non-finite result, or a
    # finite result that rounds to inf in the output dtype.
    yb = y.to(tl.int32, bitcast=True)
    yexpo = (yb >> 23) & 0xFF
    bad = (((yexpo - 255) >> 31) + 1) | (tl.abs(y) >= thresh).to(tl.int32)
    return y, bad


@triton.jit
def _amp_foreach_flag_reduce_kernel(
    bad_ptr,
    flags_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    # int8 -> fp32 conversion is a pathological path on this backend, so the
    # flags stay integer until the final (scalar) store.
    flags = tl.load(bad_ptr + offsets, mask=mask, other=0).to(tl.int32)
    tl.store(flags_ptr + pid, tl.max(flags, axis=0).to(tl.float32))


@triton.jit
def _strided_copy_kernel(
    dst,
    src,
    N,
    D0,
    D1,
    D2,
    D3,
    D4,
    D5,
    D6,
    D7,
    SS0,
    SS1,
    SS2,
    SS3,
    SS4,
    SS5,
    SS6,
    SS7,
    DS0,
    DS1,
    DS2,
    DS3,
    DS4,
    DS5,
    DS6,
    DS7,
    BLOCK: tl.constexpr,
):
    # Elementwise same-shape strided copy: dst[i] = src[i] over the logical
    # index space (dims D0..D7 outermost-first, front-padded with 1s). No
    # native copy_/copy primitives: every lane's source/destination offset is
    # computed from the two stride vectors.
    pid = tl.program_id(axis=0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N
    idx = offs
    i7 = idx % D7
    idx = idx // D7
    i6 = idx % D6
    idx = idx // D6
    i5 = idx % D5
    idx = idx // D5
    i4 = idx % D4
    idx = idx // D4
    i3 = idx % D3
    idx = idx // D3
    i2 = idx % D2
    idx = idx // D2
    i1 = idx % D1
    i0 = idx // D1
    s_off = (
        i0 * SS0
        + i1 * SS1
        + i2 * SS2
        + i3 * SS3
        + i4 * SS4
        + i5 * SS5
        + i6 * SS6
        + i7 * SS7
    )
    d_off = (
        i0 * DS0
        + i1 * DS1
        + i2 * DS2
        + i3 * DS3
        + i4 * DS4
        + i5 * DS5
        + i6 * DS6
        + i7 * DS7
    )
    vals = tl.load(src + s_off, mask=mask)
    tl.store(dst + d_off, vals, mask=mask)


def _strided_copy(dst, src):
    # Same-shape strided copy without native copy_/copy: gems overrides copy_,
    # so a plain copy_ here would recurse into the override; all offset math
    # happens in the kernel from the two stride vectors.
    assert tuple(dst.shape) == tuple(src.shape), "shape mismatch"
    n = dst.numel()
    if n:
        dims = list(src.shape)
        ss = list(src.stride())
        ds = list(dst.stride())
        pad = 8 - len(dims)
        assert pad >= 0, "strided copy supports up to 8 dims"
        dims = [1] * pad + dims
        ss = [0] * pad + ss
        ds = [0] * pad + ds
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        with torch_device_fn.device(dst.device):
            _strided_copy_kernel[grid](
                dst,
                src,
                n,
                *dims,
                *ss,
                *ds,
                BLOCK=BLOCK,
            )
    return dst


# Fused-kernel tile switch and the element count above which the vectorized
# pointwise path wins. Measured crossover on XPU: the 2-pass pointwise path is
# only faster once tensors reach a few million elements (below that its extra
# launch + flag reduce cost more than the vectorized body saves).
LARGE_TENSOR_NUMEL = 65536
POINTWISE_NUMEL = 1 << 22

# Block size of the tiny found_inf fixup kernel that scans the per-program
# bad-value flags of a whole foreach call (a few int32 loads per program).
FOUND_INF_REDUCE_BLOCK = 1024


@triton.jit
def _amp_foreach_non_finite_check_and_unscale_kernel(
    inp_ptr,
    inv_scale_ptr,
    bad_flags_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    if EVEN:
        # No tail: an unmasked load/store lowers to a block DMA on this
        # backend, while a (even all-true) mask forces the whole tile through
        # a local-memory staging buffer with scalar per-lane accesses.
        inp = tl.load(inp_ptr + offsets)
    else:
        mask = offsets < num_elements
        inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(inv_scale_ptr).to(tl.float32)

    inp_fp32 = inp.to(tl.float32)
    bits = inp_fp32.to(tl.int32, bitcast=True)
    expo = (bits >> 23) & 0xFF
    finite = ((expo - 255) >> 31) & 1  # 1 for finite, 0 for inf/nan
    finite_f = finite.to(tl.float32)
    factor = finite_f * scale + (1.0 - finite_f)
    scaled_fp32 = inp_fp32 * factor

    scaled = scaled_fp32.to(inp.dtype)
    if EVEN:
        tl.store(output_ptr + offsets, scaled)
    else:
        tl.store(output_ptr + offsets, scaled, mask=mask)

    stored_fp32 = scaled.to(tl.float32)
    stored_bits = stored_fp32.to(tl.int32, bitcast=True)
    stored_expo = (stored_bits >> 23) & 0xFF
    is_bad = ((stored_expo - 255) >> 31) + 1
    if not EVEN:
        is_bad = is_bad & mask.to(tl.int32)
    block_bad = tl.max(is_bad, axis=0)
    tl.store(bad_flags_ptr + pid, block_bad.to(tl.float32))


@triton.jit
def _amp_foreach_found_inf_reduce_kernel(
    bad_flags_ptr,
    found_inf_ptr,
    num_flags,
    BLOCK_SIZE: tl.constexpr,
):
    # Single-program fold of the per-program flags into found_inf, mirroring
    # the previous host-side semantics: found_inf <- 1.0 if any non-finite
    # value was seen in any input tensor, otherwise found_inf is untouched.
    any_bad = 0.0
    for start in range(0, num_flags, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        flag_mask = offsets < num_flags
        flags = tl.load(bad_flags_ptr + offsets, mask=flag_mask, other=0.0)
        any_bad = tl.maximum(any_bad, tl.max(flags, axis=0))

    current = tl.load(found_inf_ptr)
    updated = tl.where(any_bad > 0.0, 1.0, current)
    tl.store(found_inf_ptr, updated)


def _unscale_large(tensor, inv_scale, flags_out, num_programs):
    """Vectorized unscale + per-element bad flag + block reduce (large tensors)."""
    thresh = _OVERFLOW_THRESH[tensor.dtype]
    # torch.empty only allocates; every flag byte is unconditionally written.
    bad = torch.empty(tensor.shape, dtype=torch.int8, device=tensor.device)
    if tensor.is_contiguous():
        _unscale_flag_func(tensor, inv_scale, thresh, out0=tensor, out1=bad)
    else:
        output = torch.empty_like(tensor)
        _unscale_flag_func(tensor, inv_scale, thresh, out0=output, out1=bad)
        _strided_copy(tensor, output)

    _amp_foreach_flag_reduce_kernel[(num_programs,)](
        bad,
        flags_out,
        tensor.numel(),
        BLOCK_SIZE=REDUCE_BLOCK,
    )


def _amp_foreach_non_finite_check_and_unscale_(
    tensors: List[torch.Tensor],
    found_inf: torch.Tensor,
    inv_scale: torch.Tensor,
):
    """
    Check for non-finite values in tensors and unscale them (in-place).

    For each tensor in the list:
    - Scale finite values by inv_scale; non-finite (inf, nan) stay unchanged.
    - If any element is non-finite, set found_inf to 1.0.
    """
    logger.debug("GEMS_KUNLUNXIN AMP_FOREACH_NON_FINITE_CHECK_AND_UNSCALE")

    if not isinstance(tensors, (list, tuple)):
        raise TypeError(f"Expected list or tuple of tensors, got {type(tensors)}")

    if len(tensors) == 0:
        return

    # PyTorch expects inv_scale as a float32 tensor.
    inv_scale = inv_scale.to(dtype=torch.float32)

    # The unscale kernel and the found_inf detection are fused into a single
    # launch per tensor. The previous implementation needed ~6 launches per
    # tensor (empty_like alloc + unscale kernel + _copy_from write-back, then
    # isnan/isinf/any post-passes over the full data), which made this foreach
    # op launch- and traffic-bound (full-tensor bool materializations of
    # isnan/isinf alone quadrupled the memory traffic of the op call).
    jobs = []
    for tensor in tensors:
        if not tensor.is_floating_point():
            continue

        num_elements = tensor.numel()
        if num_elements == 0:
            continue

        if num_elements >= POINTWISE_NUMEL:
            # Vectorized pointwise path; one flag slot per reduced block.
            num_programs = triton.cdiv(num_elements, REDUCE_BLOCK)
            jobs.append((tensor, num_elements, num_programs, None))
        else:
            block = (
                BLOCK_SIZE_LARGE
                if num_elements >= LARGE_TENSOR_NUMEL
                else BLOCK_SIZE_SMALL
            )
            num_programs = triton.cdiv(num_elements, block)
            jobs.append((tensor, num_elements, num_programs, block))

    if not jobs:
        return

    total_programs = sum(job[2] for job in jobs)
    # torch.empty only allocates (no kernel launch); every flag slot is
    # unconditionally written by its own program, so no zero-init is needed.
    bad_flags = torch.empty(
        total_programs, dtype=torch.float32, device=inv_scale.device
    )

    flag_offset = 0
    for tensor, num_elements, num_programs, block in jobs:
        if block is None:
            _unscale_large(tensor, inv_scale, bad_flags[flag_offset:], num_programs)
            flag_offset += num_programs
            continue

        grid = (num_programs,)
        launch_opts = _LARGE_LAUNCH if block == BLOCK_SIZE_LARGE else _SMALL_LAUNCH

        if tensor.is_contiguous():
            # Fast path: the store target aliases the load source, which is
            # safe: every program loads its own disjoint region before
            # storing it.
            _amp_foreach_non_finite_check_and_unscale_kernel[grid](
                tensor,
                inv_scale,
                bad_flags[flag_offset:],
                tensor,
                num_elements,
                BLOCK_SIZE=block,
                EVEN=(num_elements % block == 0),
                **launch_opts,
            )
        else:
            # Non-contiguous inputs: the kernel addresses elements by flat
            # offset, so write a dense temporary and scatter it back with the
            # strided-copy kernel (gems overrides copy_, so a plain copy_ here
            # would recurse into the override; the kernel is self-contained).
            output = torch.empty_like(tensor)
            _amp_foreach_non_finite_check_and_unscale_kernel[grid](
                tensor,
                inv_scale,
                bad_flags[flag_offset:],
                output,
                num_elements,
                BLOCK_SIZE=block,
                EVEN=(num_elements % block == 0),
                **launch_opts,
            )
            _strided_copy(tensor, output)

        flag_offset += num_programs

    # One tiny fixup kernel per foreach call folds the flags into found_inf
    # (a few int32 loads per program slot; ordering is guaranteed by the
    # in-stream program order of the launches above).
    _amp_foreach_found_inf_reduce_kernel[(1,)](
        bad_flags,
        found_inf,
        total_programs,
        BLOCK_SIZE=FOUND_INF_REDUCE_BLOCK,
    )
