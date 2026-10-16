# Copyright 2026, The FlagOS Contributors.
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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)

_FAST_BLOCK = 16384
_FAST_WARPS = 32


# Round-towards-zero of an fp32 value, computed on its bit pattern by clearing
# the mantissa bits below the binary point.  `fix` is semantically identical to
# `trunc`, so this is the same integer-only bit trick used by the backend
# `trunc` override.
#
# The generic implementation (floor(x) / ceil(x) / where) is catastrophic on
# this backend: the libdevice tl.floor / tl.ceil externs collapse into a slow
# path that costs ~125 ms *per launch*, independent of tensor size (measured on
# a (64, 64) tile: 128 ms vs 0.11 ms for torch.fix), while this masking path
# runs at memcpy-class throughput.  Integer-only is also the only correct
# choice here: every fp op of this backend behaves as "no signed zero", so no
# arithmetic expression can return the -0.0 that fix(-0.5) / fix(-0.0) owe;
# only the sign bit, i.e. the mask below, can carry it.
#
#   exp (biased) >= 127 : n = 150 - exp fraction bits to clear (0 when the
#                         value is already integral)
#   exp           < 127 : n = 31 -> |x| < 1 truncates to +-0.0, and subnormal
#                         inputs / +-0.0 keep their sign this way
#   exp          == 255 : n = 0 -> +-inf and NaN pass through untouched
@triton.jit
def _fix_fp32(xf):
    xi = xf.to(tl.int32, bitcast=True)
    exp = (xi >> 23) & 255
    n = tl.where(exp < 127, 31, tl.maximum(150 - exp, 0))
    return (xi & (-1 << n)).to(tl.float32, bitcast=True)


@triton.jit
def fix_fast_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)  # numel % BLOCK == 0 guaranteed by caller
    t = _fix_fp32(x.to(tl.float32))
    tl.store(y_ptr + offs, t.to(y_ptr.dtype.element_ty))


@triton.jit
def fix_masked_kernel(x_ptr, y_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask)
    t = _fix_fp32(x.to(tl.float32))
    tl.store(y_ptr + offs, t.to(y_ptr.dtype.element_ty), mask=mask)


# Generic path: any dtype/layout/shape (bf16/fp64 kept in fp32 like the
# original implementation).
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def fix_func(x):
    return _fix_fp32(x.to(tl.float32)).to(x.dtype)


def _fix_impl(A, out=None):
    numel = A.numel()
    if (
        A.dtype in (torch.float16, torch.float32)
        and A.is_contiguous()
        and A.dim() > 0
        and numel > 1024
    ):
        block = min(_FAST_BLOCK, triton.next_power_of_2(numel))
        if out is None:
            out = torch.empty_like(A)
        if numel % block == 0:
            fix_fast_kernel[(numel // block,)](
                A, out, BLOCK=block, num_warps=_FAST_WARPS
            )
        else:
            fix_masked_kernel[(triton.cdiv(numel, block),)](
                A, out, numel, BLOCK=block, num_warps=_FAST_WARPS
            )
        return out
    if out is None:
        return fix_func(A)
    fix_func(A, out0=out)
    return out


def fix(A):
    logger.debug("GEMS_KUNLUNXIN FIX")
    if not A.is_floating_point():
        # Preserve the generic implementation behaviour for non-float dtypes
        # exactly: integer/bool tensors are an identity copy and complex uses
        # the pre-existing torch.trunc path.  Neither is exercised by
        # tests/test_fix.py (FLOAT_DTYPES only); not expanded here.
        from flag_gems.ops.fix import fix as _generic_fix

        return _generic_fix(A)
    return _fix_impl(A)


def fix_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN FIX_OUT")
    if not A.is_floating_point():
        from flag_gems.ops.fix import fix_out as _generic_fix_out

        return _generic_fix_out(A, out)
    return _fix_impl(A, out)
