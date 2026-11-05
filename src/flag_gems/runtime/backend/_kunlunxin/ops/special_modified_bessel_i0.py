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
# Kunlunxin (XPU) override of special_modified_bessel_i0 /
# special_modified_bessel_i0_out (aten::special_modified_bessel_i0).
#
# Why an override exists at all (performance):
#   the generic `flag_gems/ops/special_modified_bessel_i0.py` launches its
#   kernel with `grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)`,
#   i.e. a fresh lambda object per call.  On the xpubin backend the launch
#   grid is part of the compilation cache key, so every call becomes a full
#   recompilation: measured ~150-190 ms/call regardless of tensor size
#   (dtype-balanced Gems Speedup ~0.27-0.32 at comprehensive level), which
#   is the KERNEL_OPT_EXPERIENCE #41 fingerprint exactly.
#
# Fix (single-variable): vendor raw-pointer kernels with a constant-tuple
# grid plus the sweep-measured `_pick_block` tile buckets of the erfc/ndtr
# pointwise family.  The i0 body is the same Abramowitz & Stegun 9.8.1 /
# 9.8.2 rational polynomial as the generic kernel (fp32 compute, A&S error
# < 1.6e-7 / 1.9e-7); only `tl.exp`/`tl.rsqrt`/`tl.fma`/`tl.where` are used,
# all proven to link on XPU (see special_i0e / special_modified_bessel_k0).
# No CPU/ATen/native/composite fallback: `torch.empty*` only allocates.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

UNROLL_NUM = 8
BUFFER_SIZE_LIMIT = 2048
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    # Tile buckets sweep-measured on XPU for the erfc/ndtr pointwise family
    # (identical load/store pattern, see erfc.py / special_ndtr.py).  The
    # masked-memory path costs ~2x on this backend, so an unmasked kernel is
    # used whenever the shape divides the tile exactly.
    if n_elements >= 16_777_216 and n_elements % 65536 == 0:
        return 65536, 16, False
    if n_elements >= 1_048_576 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 65_536 and n_elements % 8192 == 0:
        return 8192, 4, False
    if n_elements <= 65_536:
        return 2048, 4, True
    return 8192, 4, True


@triton.jit
def _i0_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # i0 is even; A&S 9.8.1 (|x| <= 3.75) / 9.8.2 (|x| > 3.75).
    ax = tl.abs(x)
    use_small = ax <= 3.75

    # Small region: i0 = 1 + y*P(y), y = (x/3.75)^2.
    z = ax / 3.75
    y = z * z
    p_small_inner = tl.fma(y, 0.0045813, 0.0360768)
    p_small_inner = tl.fma(y, p_small_inner, 0.2659732)
    p_small_inner = tl.fma(y, p_small_inner, 1.2067492)
    p_small_inner = tl.fma(y, p_small_inner, 3.0899424)
    p_small_inner = tl.fma(y, p_small_inner, 3.5156229)
    res_small = tl.fma(y, p_small_inner, 1.0)

    # Large region: i0 = exp(ax) * P(3.75/ax) / sqrt(ax).  Clamp the small
    # lanes to 3.75 so 3.75/ax can never divide by zero; those lanes are
    # discarded by the final select.
    safe_ax = tl.where(use_small, 3.75, ax)
    yb = 3.75 / safe_ax
    p_big_inner = tl.fma(yb, 0.00392377, -0.01647633)
    p_big_inner = tl.fma(yb, p_big_inner, 0.02635537)
    p_big_inner = tl.fma(yb, p_big_inner, -0.02057706)
    p_big_inner = tl.fma(yb, p_big_inner, 0.00916281)
    p_big_inner = tl.fma(yb, p_big_inner, -0.00157565)
    p_big_inner = tl.fma(yb, p_big_inner, 0.00225319)
    p_big_inner = tl.fma(yb, p_big_inner, 0.01328592)
    p_big = tl.fma(yb, p_big_inner, 0.39894228)
    res_big = tl.exp(safe_ax) * p_big * tl.rsqrt(safe_ax)

    result = tl.where(use_small, res_small, res_big)
    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _i0_kernel_unmasked(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets).to(tl.float32)

    ax = tl.abs(x)
    use_small = ax <= 3.75

    z = ax / 3.75
    y = z * z
    p_small_inner = tl.fma(y, 0.0045813, 0.0360768)
    p_small_inner = tl.fma(y, p_small_inner, 0.2659732)
    p_small_inner = tl.fma(y, p_small_inner, 1.2067492)
    p_small_inner = tl.fma(y, p_small_inner, 3.0899424)
    p_small_inner = tl.fma(y, p_small_inner, 3.5156229)
    res_small = tl.fma(y, p_small_inner, 1.0)

    safe_ax = tl.where(use_small, 3.75, ax)
    yb = 3.75 / safe_ax
    p_big_inner = tl.fma(yb, 0.00392377, -0.01647633)
    p_big_inner = tl.fma(yb, p_big_inner, 0.02635537)
    p_big_inner = tl.fma(yb, p_big_inner, -0.02057706)
    p_big_inner = tl.fma(yb, p_big_inner, 0.00916281)
    p_big_inner = tl.fma(yb, p_big_inner, -0.00157565)
    p_big_inner = tl.fma(yb, p_big_inner, 0.00225319)
    p_big_inner = tl.fma(yb, p_big_inner, 0.01328592)
    p_big = tl.fma(yb, p_big_inner, 0.39894228)
    res_big = tl.exp(safe_ax) * p_big * tl.rsqrt(safe_ax)

    result = tl.where(use_small, res_small, res_big)
    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty))


def _launch_i0(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    n_elements = out.numel()
    if n_elements == 0:
        return out
    x_contig = x.contiguous() if not x.is_contiguous() else x
    out_contig = out.contiguous() if not out.is_contiguous() else out
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        _i0_kernel[grid](
            x_contig,
            out_contig,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        _i0_kernel_unmasked[grid](
            x_contig,
            out_contig,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)
    return out


def special_modified_bessel_i0(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SPECIAL_MODIFIED_BESSEL_I0")
    if not x.is_cuda:
        raise ValueError(
            "special_modified_bessel_i0: input tensor must be on CUDA device"
        )
    out_dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
    out = torch.empty_like(x, dtype=out_dtype, device=x.device)
    with torch_device_fn.device(x.device):
        _launch_i0(x, out)
    return out


def special_modified_bessel_i0_out(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SPECIAL_MODIFIED_BESSEL_I0_OUT")
    if not (x.is_cuda and out.is_cuda):
        raise ValueError(
            "special_modified_bessel_i0_out: input and output tensors must be on CUDA device"
        )
    if not out.is_floating_point():
        raise TypeError(
            "special_modified_bessel_i0_out: output tensor must be a floating point type"
        )
    if x.numel() != out.numel():
        raise ValueError(
            "special_modified_bessel_i0_out: input and output must have the same number of elements"
        )
    with torch_device_fn.device(x.device):
        _launch_i0(x, out)
    return out
