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

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_MAX_GRID = 65535
_MIN_PROGRAMS = 256
_TARGET_ITERS = 16
_BLOCK = 4096
_WARPS = 4


@triton.jit
def _i0_compute(x):
    """Elementwise modified Bessel I0 for an fp32 vector x (same polynomial
    as the generic KernelGen implementation).  No per-lane guards: dead lanes
    are discarded by the final select."""
    ax = tl.abs(x)
    use_small = ax <= 3.75

    # ---- small region: |x| <= 3.75 ----
    z = ax * (1.0 / 3.75)
    y = z * z

    p_small_inner = tl.fma(y, 0.0045813, 0.0360768)
    p_small_inner = tl.fma(y, p_small_inner, 0.2659732)
    p_small_inner = tl.fma(y, p_small_inner, 1.2067492)
    p_small_inner = tl.fma(y, p_small_inner, 3.0899424)
    p_small_inner = tl.fma(y, p_small_inner, 3.5156229)
    res_small = tl.fma(y, p_small_inner, 1.0)

    # ---- large region: |x| > 3.75 ----
    # tl.where does not short-circuit, so use a safe denominator for lanes
    # that will eventually select the small branch.
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

    return tl.where(use_small, res_small, res_big)


@triton.jit
def special_modified_bessel_i0_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    step = nprog * BLOCK
    n_full = (n_elements // BLOCK) * BLOCK
    for base in range(pid * BLOCK, n_full, step):
        offs = base + lane
        x = tl.load(x_ptr + offs).to(tl.float32)
        res = _i0_compute(x)
        tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty))
    # trailing partial tile: finished once, by program 0, with a mask
    if n_full < n_elements:
        if pid == 0:
            offs = n_full + lane
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            res = _i0_compute(x)
            tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


def _launch(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    assert (
        out.numel() == x.numel()
    ), "Input and output must have the same number of elements"
    assert out.device == x.device, "Input and output must be on the same device"

    # Ensure floating point compute
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())

    # Cast input to match the desired output dtype if needed
    # (Compute is done in fp32 inside the kernel; store casts to out dtype)
    if x.dtype != out.dtype:
        x = x.to(out.dtype)

    x_contig = x.contiguous()
    out_was_noncontig = not out.is_contiguous()
    out_contig = out.contiguous() if out_was_noncontig else out

    n_elements = out_contig.numel()
    if n_elements == 0:
        return out

    tiles = triton.cdiv(n_elements, _BLOCK)
    nprog = max(min(tiles, _MIN_PROGRAMS), triton.cdiv(tiles, _TARGET_ITERS))
    with torch_device_fn.device(x_contig.device):
        special_modified_bessel_i0_kernel[(min(nprog, _MAX_GRID),)](
            x_contig, out_contig, n_elements, BLOCK=_BLOCK, num_warps=_WARPS
        )

    if out_was_noncontig:
        out.copy_(out_contig)
    return out


def special_modified_bessel_i0(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME SPECIAL_MODIFIED_BESSEL_I0")
    if x.device.type != flag_gems.device:
        raise ValueError(
            f"special_modified_bessel_i0: input tensor must be on {flag_gems.device} device"
        )
    # Result dtype follows PyTorch's floating type behavior
    out_dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
    out = torch.empty_like(x, dtype=out_dtype)
    return _launch(out, x)


def special_modified_bessel_i0_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME SPECIAL_MODIFIED_BESSEL_I0_OUT")
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(
            f"special_modified_bessel_i0_out: input and output tensors must be on {flag_gems.device} device"
        )
    if not out.is_floating_point():
        raise TypeError(
            "special_modified_bessel_i0_out: output tensor must be a floating point type"
        )
    if x.numel() != out.numel():
        raise ValueError(
            "special_modified_bessel_i0_out: input and output must have the same number of elements"
        )
    return _launch(out, x)
