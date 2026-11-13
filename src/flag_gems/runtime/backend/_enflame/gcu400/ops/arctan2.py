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

logger = logging.getLogger(__name__)

NUM_PROGRAMS = 48

# atan(t) ~= t * P(t^2) minimax on [0, 1]; max abs error 2.3e-5 (test gate atol 1e-4).
_P0 = tl.constexpr(0.9999693036)
_P1 = tl.constexpr(-0.3316778839)
_P2 = tl.constexpr(0.1851029694)
_P3 = tl.constexpr(-0.09174843878)
_P4 = tl.constexpr(0.02377510071)
_PI_OVER_2 = tl.constexpr(1.5707963267948966)
_PI = tl.constexpr(3.141592653589793)
_INF = tl.constexpr(float("inf"))
_NAN = tl.constexpr(float("nan"))


@triton.jit
def _atan2_vec(x, y):
    # x is the horizontal coordinate, y the vertical one; returns atan2(y, x).
    # The libdevice atan2 lowers to a per-128-lane extern call (tops_ef_vatan2f)
    # on this backend, so the angle is evaluated with plain arithmetic instead:
    # min/max ratio reduction to [0, 1], one minimax polynomial, quadrant fixups.
    ax = tl.abs(x)
    ay = tl.abs(y)
    swap = ay > ax
    m = tl.minimum(ax, ay)
    M = tl.maximum(ax, ay)
    t = m / M
    # both operands infinite -> t = inf/inf = NaN; the true ratio is 1.
    t = tl.where(m == _INF, 1.0, t)
    # both operands zero -> t = 0/0 = NaN; the result is 0 (+-pi after the
    # x-sign wrap for -0), so feeding 0 keeps the generic path correct.
    t = tl.where(M == 0.0, 0.0, t)
    s = t * t
    a = _P4
    a = a * s + _P3
    a = a * s + _P2
    a = a * s + _P1
    a = a * s + _P0
    a = a * t
    a = tl.where(swap, _PI_OVER_2 - a, a)
    xi = x.to(tl.int32, bitcast=True)
    yi = y.to(tl.int32, bitcast=True)
    xsb = xi < 0
    a = tl.where(xsb, _PI - a, a)
    a = tl.where((x != x) | (y != y), _NAN, a)
    return tl.where(yi < 0, -a, a)


@libentry()
@triton.jit
def _arctan2_kernel(In0, In1, Out, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        y = tl.load(In0 + off, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(In1 + off, mask=mask, other=0.0).to(tl.float32)
        res = _atan2_vec(x, y)
        tl.store(Out + off, res.to(Out.dtype.element_ty), mask=mask)


def _choose_block(N):
    if N <= 2048:
        # width-1 vectors are pathological on this backend; pad tiny tiles.
        return max(128, triton.next_power_of_2(max(N, 1)))
    return 2048


def _launch(in0, in1, out):
    N = out.numel()
    if N == 0:
        return out
    BLOCK = _choose_block(N)
    grid = min(triton.cdiv(N, BLOCK), NUM_PROGRAMS)
    with torch_device_fn.device(in0.device):
        _arctan2_kernel[(grid,)](in0, in1, out, N, BLOCK=BLOCK, num_warps=4)
    return out


def _prepare(A, B):
    if A.shape != B.shape or A.dtype != B.dtype:
        A, B = torch.broadcast_tensors(A, B)
    A = A if A.is_contiguous() else A.contiguous()
    B = B if B.is_contiguous() else B.contiguous()
    out_dtype = torch.promote_types(A.dtype, B.dtype)
    if not out_dtype.is_floating_point:
        out_dtype = torch.float32
    return A, B, out_dtype


def arctan2(input, other):
    logger.debug("GEMS_ENFLAME ARCTAN2")
    A, B, out_dtype = _prepare(input, other)
    out = torch.empty(A.shape, dtype=out_dtype, device=A.device)
    return _launch(A, B, out)


def arctan2_(input, other):
    """原地计算 atan2(input, other)，按 aten 语义把结果写回 input 本身。

    契约：返回值必须是 input 对象；other 仅提供分母，可按广播规则扩展到
    input.shape，但不得改变输出形状。主 kernel 使用扁平索引，因此 other 必须
    物化为与输出同形的连续张量；非连续 input 需在连续副本上计算后 copy_ 写回，
    以保持原地别名关系且不退回公共 codegen 的多维索引分解路径。
    """
    logger.debug("GEMS_ENFLAME ARCTAN2_")
    if input.numel() == 0:
        return input
    # aten 原地语义允许 other 广播到 input 的形状；先物化成与输出同形的连续张量，
    # 使扁平索引 kernel 能按输出元素顺序取到对应的分子/分母。
    broadcast_other = torch.broadcast_to(other, input.shape)
    if not broadcast_other.is_contiguous():
        broadcast_other = broadcast_other.contiguous()
    if not input.is_contiguous():
        # 非连续输出视图：在连续副本上计算后写回，避免公共 codegen 的多维索引分解
        contiguous_input = input.contiguous()
        _launch(contiguous_input, broadcast_other, contiguous_input)
        input.copy_(contiguous_input)
        return input
    return _launch(input, broadcast_other, input)
