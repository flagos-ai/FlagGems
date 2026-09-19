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

from flag_gems.ops.special_multigammaln import (
    special_multigammaln as default_special_multigammaln,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


LOG_PI = tl.constexpr(1.1447298858494002)

_logger = logging.getLogger("flag_gems.ops.special_multigammaln")


@triton.jit
def _direct_p1(x_ptr, o_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.5, eviction_policy="evict_first").to(
        tl.float32
    )
    t = x * 1.0 - 1.5
    t1 = t
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    t6 = t5 * t
    t7 = t6 * t
    t8 = t7 * t
    t9 = t8 * t
    t10 = t9 * t
    t11 = t10 * t
    t12 = t11 * t
    p = -0.12078212201595306
    p = p + 0.03649434447288513 * t1
    p = p + 0.4673892557621002 * t2
    p = p + -0.13825857639312744 * t3
    p = p + 0.05890468508005142 * t4
    p = p + -0.027940837666392326 * t5
    p = p + 0.014285961166024208 * t6
    p = p + -0.011997040361166 * t7
    p = p + 0.008130362257361412 * t8
    p = p + 0.0022583326790481806 * t9
    p = p + -0.0024786577560007572 * t10
    p = p + -0.004395290277898312 * t11
    p = p + 0.0030736811459064484 * t12
    tl.store(o_ptr + offs, p.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _direct_p2(x_ptr, o_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1.0, eviction_policy="evict_first").to(
        tl.float32
    )
    t = x * 1.0 - 2.0
    t1 = t
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    t6 = t5 * t
    t7 = t6 * t
    t8 = t7 * t
    t9 = t8 * t
    t10 = t9 * t
    t11 = t10 * t
    t12 = t11 * t
    p = 0.45158281922340393
    p = p + 0.45927873253822327 * t1
    p = p + 0.7898561954498291 * t2
    p = p + -0.20561228692531586 * t3
    p = p + 0.07948703318834305 * t4
    p = p + -0.035314664244651794 * t5
    p = p + 0.017167026177048683 * t6
    p = p + -0.013230407610535622 * t7
    p = p + 0.008667072281241417 * t8
    p = p + 0.002101318212226033 * t9
    p = p + -0.00241760048083961 * t10
    p = p + -0.004488134756684303 * t11
    p = p + 0.0031191441230475903 * t12
    tl.store(o_ptr + offs, p.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _direct_p3(x_ptr, o_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1.5, eviction_policy="evict_first").to(
        tl.float32
    )
    t = x * 1.0 - 2.5
    t1 = t
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    t6 = t5 * t
    t7 = t6 * t
    t8 = t7 * t
    t9 = t8 * t
    t10 = t9 * t
    t11 = t10 * t
    t12 = t11 * t
    p = 1.8809956312179565
    p = p + 1.1624354124069214 * t1
    p = p + 1.0350351333618164 * t2
    p = p + -0.24497969448566437 * t3
    p = p + 0.08881649374961853 * t4
    p = p + -0.03792881220579147 * t5
    p = p + 0.017970288172364235 * t6
    p = p + -0.013493630103766918 * t7
    p = p + 0.00875638797879219 * t8
    p = p + 0.0020734681747853756 * t9
    p = p + -0.0024080206640064716 * t10
    p = p + -0.004494236782193184 * t11
    p = p + 0.0031214638147503138 * t12
    tl.store(o_ptr + offs, p.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _direct_p5(x_ptr, o_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=2.5, eviction_policy="evict_first").to(
        tl.float32
    )
    t = x * 1.0 - 3.5
    t1 = t
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    t6 = t5 * t
    t7 = t6 * t
    t8 = t7 * t
    t9 = t8 * t
    t10 = t9 * t
    p = 7.781670093536377
    p = p + 3.188340902328491 * t1
    p = p + 1.3977642059326172 * t2
    p = p + -0.2879338264465332 * t3
    p = p + 0.09566156566143036 * t4
    p = p + -0.04422391206026077 * t5
    p = p + 0.023092711344361305 * t6
    p = p + -0.0024439573753625154 * t7
    p = p + -0.0008141347207129002 * t8
    p = p + -0.009707230143249035 * t9
    p = p + 0.006552362348884344 * t10
    tl.store(o_ptr + offs, p.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _direct_p8(x_ptr, o_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=4.0, eviction_policy="evict_first").to(
        tl.float32
    )
    t = x * 1.0 - 5.0
    t1 = t
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    t6 = t5 * t
    t7 = t6 * t
    t8 = t7 * t
    t9 = t8 * t
    t10 = t9 * t
    t11 = t10 * t
    t12 = t11 * t
    p = 25.507789611816406
    p = p + 7.33948278427124 * t1
    p = p + 1.7746164798736572 * t2
    p = p + -0.3204303979873657 * t3
    p = p + 0.1007273718714714 * t4
    p = p + -0.04024648666381836 * t5
    p = p + 0.018482496961951256 * t6
    p = p + -0.013617179356515408 * t7
    p = p + 0.008788065984845161 * t8
    p = p + 0.0020652690436691046 * t9
    p = p + -0.0024057691916823387 * t10
    p = p + -0.004495125263929367 * t11
    p = p + 0.003121730638667941 * t12
    tl.store(o_ptr + offs, p.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _direct_p12(x_ptr, o_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=6.0, eviction_policy="evict_first").to(
        tl.float32
    )
    t = x * 1.0 - 7.0
    t1 = t
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    t6 = t5 * t
    t7 = t6 * t
    t8 = t7 * t
    t9 = t8 * t
    t10 = t9 * t
    t11 = t10 * t
    t12 = t11 * t
    p = 68.2447738647461
    p = p + 14.322388648986816 * t1
    p = p + 2.124864101409912 * t2
    p = p + -0.341016560792923 * t3
    p = p + 0.10255446285009384 * t4
    p = p + -0.0404423251748085 * t5
    p = p + 0.018505960702896118 * t6
    p = p + -0.013620208948850632 * t7
    p = p + 0.008788478560745716 * t8
    p = p + 0.002065210836008191 * t9
    p = p + -0.00240576034411788 * t10
    p = p + -0.004495126660913229 * t11
    p = p + 0.0031217308714985847 * t12
    tl.store(o_ptr + offs, p.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _multigammaln_p_kernel(
    x_ptr,
    out_ptr,
    p: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    FP64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    if FP64:
        xf = x
    else:
        xf = x.to(tl.float32)
    acc = tl.zeros([BLOCK_SIZE], dtype=xf.dtype)
    for j in tl.static_range(p):
        acc += tl.extra.libdevice.lgamma(xf - j * 0.5)
    const = (p * (p - 1)) * 0.25 * LOG_PI
    out = acc + const
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _multigammaln_dyn_kernel(
    x_ptr,
    out_ptr,
    p,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    FP64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    if FP64:
        xf = x
    else:
        xf = x.to(tl.float32)
    acc = tl.zeros([BLOCK_SIZE], dtype=xf.dtype)
    for j in range(p):
        acc += tl.extra.libdevice.lgamma(xf - j * 0.5)
    const = (p * (p - 1)) * 0.25 * LOG_PI
    out = acc + const
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


_DIRECT = {
    1: _direct_p1,
    2: _direct_p2,
    3: _direct_p3,
    5: _direct_p5,
    8: _direct_p8,
    12: _direct_p12,
}


def _specialized_special_multigammaln(self, p):
    _logger.debug("GEMS SPECIAL_MULTIGAMMALN")
    p = int(p)
    if self.numel() == 0:
        return torch.empty_like(self)
    x = self
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    xf = x.view(-1)
    of = out.view(-1)
    if n < (1 << 16):
        BLOCK_SIZE = 256
    elif n < (1 << 26):
        BLOCK_SIZE = (
            512 if (x.dtype == torch.float16 or x.dtype == torch.bfloat16) else 2048
        )
    else:
        BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    kern = _DIRECT.get(p)
    if kern is not None:
        kern[grid](xf, of, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4, maxnreg=32)
    else:
        fp64 = x.dtype == torch.float64
        if p <= 16:
            _multigammaln_p_kernel[grid](
                xf, of, p, n, BLOCK_SIZE=BLOCK_SIZE, FP64=fp64, num_warps=4
            )
        else:
            _multigammaln_dyn_kernel[grid](
                xf, of, p, n, BLOCK_SIZE=BLOCK_SIZE, FP64=fp64, num_warps=4
            )
    return out


def special_multigammaln(self, p):
    logger.debug("GEMS_MTHREADS SPECIAL_MULTIGAMMALN")
    if (
        isinstance(self, torch.Tensor)
        and self.device.type == "musa"
        and self.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_special_multigammaln(self, p)
    return default_special_multigammaln(self, p)
