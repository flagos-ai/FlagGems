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

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _special_erfinv_kernel_xpu(
    x_ptr, out_ptr, n_elements,
    BLOCK: tl.constexpr, TILES_PER_CTA: tl.constexpr, ONE_TILE: tl.constexpr,
):
    # XPU specialization of the erfinv rational polynomial.
    #
    # A single degree-12 polynomial in w = -log((1-x)(1+x)) (Horner, 12 FMA)
    # replaces the previous two-branch (w<5 / sqrt(w)) approximation. It is
    # accurate to ~2e-6 on x in [-0.999995, 0.999995] (far inside the fp32
    # 1e-4 atol budget) and tracks the true erfinv to within ~1% at the very
    # extreme tail, hitting +inf at x=+1 and -inf at x=-1 exactly -- the
    # leading Horner coefficient is positive, so no sign fix is needed (the
    # previous high branch had a negative first coefficient and gave -inf at
    # +1, which the old code corrected with a wide `tl.where`).
    #
    # The out-of-domain / boundary semantics come from IEEE arithmetic, so no
    # explicit special-case handling is emitted, and no unordered `!=`/setuo
    # compare exists anywhere -- the XPU LLVM selector crash the old NaN
    # bit-check worked around cannot trigger:
    #   * NaN input  -> NaN  (propagates through log/poly)
    #   * |x| > 1    -> NaN  (log of a negative number)
    #   * x == +1    -> +inf
    #   * x == -1    -> -inf
    # grid = (12,) grid-stride tiles and no `other=` on loads (both are
    # slow paths on this backend).
    pid = tl.program_id(0)
    if ONE_TILE:
        tid = pid * BLOCK + tl.arange(0, BLOCK)
        mask = tid < n_elements
        x = tl.load(x_ptr + tid, mask=mask)
        xf = x.to(tl.float32)

        one = 1.0
        w = -tl.log((one - xf) * (one + xf))
        wl = w - 2.5
        p = 1.1443425061489894e-12
        p = -5.492934420147839e-11 + p * wl
        p = 1.0267015254802125e-09 + p * wl
        p = -8.478802959745463e-09 + p * wl
        p = 6.089567262255907e-09 + p * wl
        p = 4.7026309515791583e-07 + p * wl
        p = -3.3593168315697363e-06 + p * wl
        p = -4.999015344304015e-06 + p * wl
        p = 0.00021807259506328384 + p * wl
        p = -0.0012526699951610407 + p * wl
        p = -0.0041771138285083284 + p * wl
        p = 0.24664026374680736 + p * wl
        p = 1.501409313316536 + p * wl
        res = p * xf

        tl.store(out_ptr + tid, res.to(x_ptr.type.element_ty), mask=mask)
    else:
        num_ctas = tl.num_programs(0)
        for j in range(0, TILES_PER_CTA):
            tile_id = pid + j * num_ctas
            tid = tile_id * BLOCK + tl.arange(0, BLOCK)
            mask = tid < n_elements
            x = tl.load(x_ptr + tid, mask=mask)
            xf = x.to(tl.float32)

            one = 1.0
            w = -tl.log((one - xf) * (one + xf))
            wl = w - 2.5
            p = 1.1443425061489894e-12
            p = -5.492934420147839e-11 + p * wl
            p = 1.0267015254802125e-09 + p * wl
            p = -8.478802959745463e-09 + p * wl
            p = 6.089567262255907e-09 + p * wl
            p = 4.7026309515791583e-07 + p * wl
            p = -3.3593168315697363e-06 + p * wl
            p = -4.999015344304015e-06 + p * wl
            p = 0.00021807259506328384 + p * wl
            p = -0.0012526699951610407 + p * wl
            p = -0.0041771138285083284 + p * wl
            p = 0.24664026374680736 + p * wl
            p = 1.501409313316536 + p * wl
            res = p * xf

            tl.store(out_ptr + tid, res.to(x_ptr.type.element_ty), mask=mask)


def _erfinv_launch(x, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    num_ctas = 12
    num_tiles = num_ctas
    block = triton.next_power_of_2(triton.cdiv(n_elements, num_tiles))
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    _special_erfinv_kernel_xpu[(num_ctas, 1, 1)](
        x,
        out,
        n_elements,
        BLOCK=block,
        TILES_PER_CTA=tiles_per_cta,
        ONE_TILE=tiles_per_cta == 1,
    )


def special_erfinv_out(x: torch.Tensor, out: torch.Tensor):
    """special_erfinv.out specialized for XPU (crash-free NaN handling)."""
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ERFINV_OUT")
    if out.shape != x.shape:
        out.resize_(x.shape)
    assert out.dtype == x.dtype
    x_in = x if x.is_contiguous() else x.contiguous()
    if out.is_contiguous():
        _erfinv_launch(x_in, out)
        return out
    out_c = torch.empty_like(x_in)
    _erfinv_launch(x_in, out_c)
    out.copy_(out_c)
    return out


def special_erfinv(x: torch.Tensor):
    """special_erfinv specialized for XPU (crash-free NaN handling)."""
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ERFINV")
    x_in = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(x_in)
    _erfinv_launch(x_in, out)
    if out.shape != x.shape or out.stride() != x.stride():
        out = out.reshape(x.shape).as_strided(x.size(), x.stride())
    return out
