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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)

UNROLL = 4

# 持久化（grid-stride）路径参数：固定 program 数循环覆盖整段。
# 实测 192 programs × BLOCK=4096 相比「一个 program 一块」的朴素 grid 在大 shape 上
# 快 18%~24%（probe_01_uniform_kernels.py gsprog/gsprog2）；program 数取非 2 的幂的
# 192 更贴合 GCU 的 CU 数，负载更均衡。小 shape 仍走朴素 grid（持久化的固定
# program 开销比收益大）。
PERSIST_BLOCK = 4096
PERSIST_NUM_PROG = 192
PERSIST_NUM_WARPS = 4
PERSIST_MIN_NUMEL = 1 << 21  # 2M 元素以下朴素 grid 更划算


@triton.jit
def _next_below(x, DT: tl.constexpr):
    """Largest ``DT`` value strictly less than ``x``.

    Used to build the exclusive upper bound of the sampling interval in the
    *output* dtype: ``uniform`` samples from ``[from, to)``, so a value that
    rounds up to ``to`` in a low-precision dtype must be stepped back one ulp.

    A zero input needs care: the bit patterns of ``+0.0`` and ``-0.0`` are the
    bottom of the positive block and the top of the negative block, so the
    usual "-1 for x >= 0, +1 otherwise" step walks *out* of the number line and
    wraps around -- ``+0.0`` (0) minus one is all-ones, and ``-0.0``
    (0x8000...0) plus one is all-ones too, both of which are NaN. The true
    predecessor of either zero is ``-smallest_subnormal``, so a zero input is
    answered directly as the negation of the smallest positive subnormal.
    """
    zero = x == 0
    if DT == tl.float32:
        bits = x.to(tl.int32, bitcast=True)
        step = tl.where(x >= 0, -1, 1).to(tl.int32)
        stepped = (bits + step).to(tl.float32, bitcast=True)
        # smallest positive subnormal: the bit pattern with a lone low bit.
        tiny = tl.full((), 1, tl.int32).to(tl.float32, bitcast=True)
        return tl.where(zero, -tiny, stepped).to(tl.float32)
    elif DT == tl.float64:
        bits = x.to(tl.int64, bitcast=True)
        step = tl.where(x >= 0, -1, 1).to(tl.int64)
        stepped = (bits + step).to(tl.float64, bitcast=True)
        tiny = tl.full((), 1, tl.int64).to(tl.float64, bitcast=True)
        return tl.where(zero, -tiny, stepped).to(tl.float64)
    else:
        bits = x.to(tl.int16, bitcast=True)
        step = tl.where(x >= 0, -1, 1).to(tl.int16)
        stepped = (bits + step).to(DT, bitcast=True)
        tiny = tl.full((), 1, tl.int16).to(DT, bitcast=True)
        return tl.where(zero, -tiny, stepped).to(DT)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "from_", "to", "N"])
def uniform_kernel_gcu400(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    scale = to - from_
    r0 = uint_to_uniform_float(r0) * scale + from_
    r1 = uint_to_uniform_float(r1) * scale + from_
    r2 = uint_to_uniform_float(r2) * scale + from_
    r3 = uint_to_uniform_float(r3) * scale + from_
    # 采样值在 float32 中算出，存入低精度 dtype 时的舍入可能正好落到排他上界
    # ``to`` 上（fp16 / bf16 尤其明显），因此要 clamp 到 ``to`` 在输出 dtype 上的
    # 前驱 ``hi``。clamp 统一在 float32 域完成：``hi`` 在 OUT_DT 中可精确表示且
    # 严格小于 ``to``，float32 域取 min 后再做唯一一次 fp32->OUT_DT 转换，既保持
    # ``[from, to)`` 语义，又避免 16 位 dtype 上多一次逐元素 dtype 转换
    # （实测该转换是 fp16/bf16 大 shape 的主要瓶颈）。
    OUT_DT: tl.constexpr = out_ptr.dtype.element_ty
    to_dt = to.to(OUT_DT)
    from_dt = from_.to(OUT_DT)
    hi = _next_below(to_dt, OUT_DT)
    hi = tl.where(to_dt > from_dt, hi, to_dt)
    if OUT_DT == tl.float64:
        hi_wide = hi.to(tl.float64)
        r0 = tl.minimum(r0.to(tl.float64), hi_wide).to(OUT_DT)
        r1 = tl.minimum(r1.to(tl.float64), hi_wide).to(OUT_DT)
        r2 = tl.minimum(r2.to(tl.float64), hi_wide).to(OUT_DT)
        r3 = tl.minimum(r3.to(tl.float64), hi_wide).to(OUT_DT)
    else:
        hi_narrow = hi.to(tl.float32)
        r0 = tl.minimum(r0, hi_narrow).to(OUT_DT)
        r1 = tl.minimum(r1, hi_narrow).to(OUT_DT)
        r2 = tl.minimum(r2, hi_narrow).to(OUT_DT)
        r3 = tl.minimum(r3, hi_narrow).to(OUT_DT)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    # N 能被 BLOCK*4 整除时走无 mask 的直接 store：GCU 上 16 位 dtype 的
    # masked store 会退化为逐元素写，实测比无 mask 慢约 6%。
    if EVEN:
        tl.store(out_ptr + off_0, r0)
        tl.store(out_ptr + off_1, r1)
        tl.store(out_ptr + off_2, r2)
        tl.store(out_ptr + off_3, r3)
    else:
        tl.store(out_ptr + off_0, r0, mask=off_0 < N)
        tl.store(out_ptr + off_1, r1, mask=off_1 < N)
        tl.store(out_ptr + off_2, r2, mask=off_2 < N)
        tl.store(out_ptr + off_3, r3, mask=off_3 < N)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "from_", "to", "N"])
def uniform_kernel_persist_gcu400(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
    BLOCK: tl.constexpr,
    NUM_PROG: tl.constexpr,
):
    """grid-stride 持久化版本：固定 NUM_PROG 个 program 循环覆盖整段。

    每个 program 每次处理 4*BLOCK 个元素（一次 philox 出 4 个 32 bit 随机数），
    计数器区间 [start, start+BLOCK) 在 program 之间、循环之间都不重叠，
    因此随机性口径与朴素版本一致。
    """
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    scale = to - from_
    OUT_DT: tl.constexpr = out_ptr.dtype.element_ty
    to_dt = to.to(OUT_DT)
    hi = _next_below(to_dt, OUT_DT)
    hi = tl.where(to_dt > from_.to(OUT_DT), hi, to_dt)
    hi_narrow = hi.to(tl.float32)
    lane = tl.arange(0, BLOCK)
    _O = lane * 0
    step = NUM_PROG * BLOCK * 4
    start = tl.program_id(0) * BLOCK * 4
    while start < N:
        cur = c0 + (start + lane).to(tl.uint32)
        r0, r1, r2, r3 = tl.philox(philox_seed, cur, c1, _O, _O)
        base = start + lane
        v0 = tl.minimum(uint_to_uniform_float(r0) * scale + from_, hi_narrow).to(OUT_DT)
        v1 = tl.minimum(uint_to_uniform_float(r1) * scale + from_, hi_narrow).to(OUT_DT)
        v2 = tl.minimum(uint_to_uniform_float(r2) * scale + from_, hi_narrow).to(OUT_DT)
        v3 = tl.minimum(uint_to_uniform_float(r3) * scale + from_, hi_narrow).to(OUT_DT)
        off_1 = base + BLOCK
        off_2 = base + BLOCK * 2
        off_3 = base + BLOCK * 3
        tl.store(out_ptr + base, v0, mask=base < N)
        tl.store(out_ptr + off_1, v1, mask=off_1 < N)
        tl.store(out_ptr + off_2, v2, mask=off_2 < N)
        tl.store(out_ptr + off_3, v3, mask=off_3 < N)
        start += step


def uniform_(self, from_=0.0, to=1.0, *, generator=None):
    logger.debug("GEMS_ENFLAME UNIFORM_")
    # aten::uniform_ 拒绝反转区间；原地版本此前缺少同一校验。
    if from_ > to:
        raise RuntimeError(
            "uniform_ expects to return a [from, to) range, but found "
            f"from={from_:g} > to={to:g}"
        )
    # kernel 按扁平连续下标写回；非连续视图先写连续副本再按原布局拷回。
    if not self.is_contiguous():
        contiguous_out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        uniform_(contiguous_out, from_, to, generator=generator)
        self.copy_(contiguous_out)
        return self
    N = volume(self.shape)
    dtype = self.dtype

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(self.device):
        # fp64 输出在持久化核里没有 fp64 域的 clamp 路径，继续走朴素 grid。
        if N >= PERSIST_MIN_NUMEL and dtype != torch.float64:
            # 大 shape 走持久化 grid-stride：固定 program 数，避免 grid 过大。
            uniform_kernel_persist_gcu400[(PERSIST_NUM_PROG,)](
                self,
                N,
                philox_seed,
                philox_offset,
                from_,
                to,
                BLOCK=PERSIST_BLOCK,
                NUM_PROG=PERSIST_NUM_PROG,
                num_warps=PERSIST_NUM_WARPS,
                num_stages=1,
            )
            return self

        if N <= 16384:
            BLOCK = 256
        elif N <= 65536:
            BLOCK = 1024
        elif dtype == torch.float32:
            BLOCK = 32768
        else:
            BLOCK = 16384

        grid = (triton.cdiv(N, BLOCK * UNROLL),)
        nw = 2 if dtype == torch.float32 else 4
        even = N % (BLOCK * UNROLL) == 0
        uniform_kernel_gcu400[grid](
            self,
            N,
            philox_seed,
            philox_offset,
            from_,
            to,
            BLOCK=BLOCK,
            EVEN=even,
            num_warps=nw,
            num_stages=1,
        )
    return self
