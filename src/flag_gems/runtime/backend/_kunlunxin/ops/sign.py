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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# sign is a pure memory-bound sign-extract pointwise op. Body evidence (XPU5
# probe, see harness/solution/performance/sign_xpu5_20260813.md): the generic
# fp compare path `(x>0).to(dtype) - (x<0).to(dtype)` lowers to fp compare ->
# i1 -> select -> sub, which on XPU costs ~2.8x the bit-domain formulation
# (fp16 (4096,4096): 810us -> 289us; fp32 713us -> 285us; bf16 757us -> 354us).
# The bit-domain body uses only integer ALU ops (no fp compare, no select):
#   r = (1.0-bits | signbit) & (nonzero-mask) & (not-NaN-mask)
# with the two masks computed arithmetically:
#   nz = ((0 - m) | m) >> (bits-1)     (-1 iff magnitude m != 0, kills +-0)
#   nf = (m - INF_BITS - 1) >> (bits-1) (-1 iff m <= INF_BITS, kills NaN;
#        for fp16 inf==0x7C00, fp32/bf16(wide) inf==0x7F800000; masked m is
#        unsigned so m - (inf+1) is >= 0 exactly for NaN bit patterns)
# NaN -> 0 and +-0 -> 0 match torch.sign semantics; -1.0 is encoded through the
# pure sign mask, +1.0 through 1.0-bits, both exact. The single-tensor kernel
# (no scalar arg) goes through the vectorized 1D-tile DMA path, which is
# ~2.2x faster than the generic masked kernel on numel > 65536 (sweep A/B
# evidence, see sign_out_xpu4_20260812.md). The generic masked kernel
# (BLOCK_SIZE=1024, grid=cdiv(n,1024)) keeps the edge on tiny tensors where
# the vectorized path's coarser grid doubles launch count; keep it as the
# numel <= SMALL_NUMEL fast path.
_SMALL_NUMEL = 65536
_SIGN_CONFIG = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    buffer_size_limit=4096,
    unroll_num=8,
)


@triton.jit
def _sign_bit(x):
    # sign(x) via integer bit domain; exact for +-0/+-inf/NaN/subnormals.
    # NaN -> 0, +-0 -> 0 (torch semantics). Non-float dtypes fall back to the
    # exact compare body (correct for all ints; fp64/others use the same path).
    if x.dtype == tl.float16:
        i = x.to(tl.int16, bitcast=True)
        m = i & 0x7FFF
        nz = ((0 - m) | m) >> 15                        # -1 iff m != 0
        nf = (m - 0x7C01) >> 15                         # -1 iff m <= 0x7C00
        r = (0x3C00 | ((i >> 15) << 15)) & nz & nf
        return r.to(x.dtype, bitcast=True)
    elif x.dtype == tl.bfloat16:
        # bf16 must widen to fp32 (u16 bitcast errors in the Triton frontend)
        u = x.to(tl.float32).to(tl.uint32, bitcast=True)
        i0 = (u & 0x7FFFFFFF).to(tl.int32, bitcast=True)
        nz = ((0 - i0) | i0) >> 31
        nf = (i0 - 0x7F800001) >> 31
        r = (
            (0x3F800000 | (u & 0x80000000))
            & nz.to(tl.uint32, bitcast=True)
            & nf.to(tl.uint32, bitcast=True)
        )
        return r.to(tl.float32, bitcast=True).to(x.dtype)
    elif x.dtype == tl.float32:
        u = x.to(tl.uint32, bitcast=True)
        i0 = (u & 0x7FFFFFFF).to(tl.int32, bitcast=True)
        nz = ((0 - i0) | i0) >> 31
        nf = (i0 - 0x7F800001) >> 31
        r = (
            (0x3F800000 | (u & 0x80000000))
            & nz.to(tl.uint32, bitcast=True)
            & nf.to(tl.uint32, bitcast=True)
        )
        return r.to(x.dtype, bitcast=True)
    else:
        # int / fp64 fallback: exact, same semantics as torch.sign
        return (x > 0).to(x.dtype) - (x < 0).to(x.dtype)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=_SIGN_CONFIG)
@triton.jit
def _sign_func(x):
    return _sign_bit(x)


@triton.jit
def _sign_small_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
):
    # small-numel fast path: 1D masked kernel with minimal launch count
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    if IS_BOOL:
        result = x
    else:
        result = _sign_bit(x)
    tl.store(out_ptr + offsets, result, mask=mask)


def _sign_impl(x: torch.Tensor, out: torch.Tensor):
    if x.is_complex():
        raise NotImplementedError(
            "Complex dtypes are not supported. Use torch.sgn for complex numbers."
        )
    if x.device != out.device:
        raise RuntimeError("input and out must be on the same device")
    if x.dtype != out.dtype:
        raise RuntimeError(f"out must have dtype {x.dtype}, but got {out.dtype}")
    if out.shape != x.shape:
        out.resize_(x.shape)
    if x.numel() == 0:
        return out
    if x.dtype == torch.bool:
        # torch.sign(bool) is the identity; comparisons are meaningless on i1.
        if out.data_ptr() != x.data_ptr():
            out.copy_(x)
        return out
    if x.numel() <= _SMALL_NUMEL and x.is_contiguous() and out.is_contiguous():
        grid = (triton.cdiv(x.numel(), 1024),)
        _sign_small_kernel[grid](
            x.view(-1),
            out.view(-1),
            x.numel(),
            BLOCK_SIZE=1024,
            IS_BOOL=x.dtype == torch.bool,
        )
        return out
    _sign_func(x, out0=out)
    return out


def sign(x: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SIGN")
    return _sign_impl(x, torch.empty_like(x))


def sign_out(x: torch.Tensor, *, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SIGN_OUT")
    return _sign_impl(x, out)