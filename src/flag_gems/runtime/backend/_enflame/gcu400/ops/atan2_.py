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

"""GCU400 atan2_（aten::atan2_，in-place）flat 快路径。

本 shell 原为公版 ``atan2_kernel(A, B, out0=A)`` 直调——generic
pointwise_dynamic codegen 在本后端把 atan2 落到 ``__ef_atan2f`` extern SIP
调用（48-CTA 钳制 grid 下 ~30× 于访存地板：2^30 fp16 实测 419ms vs native
12.9ms，speedup 0.028×；2026-11-10 j20261110124825 的本地重写曾达
0.89-1.03× 并回写 bench_json 快照，但该修复未提交已丢失，本文件按同族
先例重建）。

kernel = ``atan2.py`` 已交付的象限规约 + K=6 minimax 奇多项式（Estrin
展开，maxerr 1.9e-6）body，叠加 ``arctan2.py`` 的 IEEE 角点守卫
（min==inf 时角度取 π/4（poly 后置——除法链内 select 会串行化软流水，
fp32 4096² A/B 0.68× vs 1.11×）、显式 NaN 掩码——本后端 min/max 丢弃
NaN 不传播；探针实证公版 extern 与 native aten 角点全对，缺守卫即语义
回归）。原地别名 launch（per-element load 后 store 同 offset，别名安
全；arctan2_ 同款契约：结果写回 input 本身并返回同一对象）。

守卫与回退：标量 other 以 fp32 0-dim factory 包装后走快路径广播（公版
pointwise_dynamic 无 scalar 档）；超 int32 安全域（numel >= 2^31，flat
索引越界风险）/ 非 fp16-bf16-fp32 输入 → 公版包装（即改动前路径，语义
不变）；非连续 input 在连续副本上计算后 copy_ 写回（arctan2_ 先例）；
other 按广播规则物化为与 input 同形的连续张量。launch 带 P02 工具链
3 次冷编译 flake 重试（sinh/rad2deg/atan2_ 先例）。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.atan2 import atan2_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# Odd minimax polynomial for atan(t) on t in [0, 1], fitted by least squares:
# atan(t) ~= t * (C0 + C1*t^2 + ... + C6*t^12), max abs error 1.9e-6 (fp32).
_C0 = tl.constexpr(9.999994932e-01)
_C1 = tl.constexpr(-3.332772185e-01)
_C2 = tl.constexpr(1.989734582e-01)
_C3 = tl.constexpr(-1.356217090e-01)
_C4 = tl.constexpr(8.545280521e-02)
_C5 = tl.constexpr(-3.853540616e-02)
_C6 = tl.constexpr(8.408503156e-03)
_PI = tl.constexpr(3.141592653589793)
_PIO2 = tl.constexpr(1.5707963267948966)
_PIO4 = tl.constexpr(0.7853981633974483)
_INF = tl.constexpr(float("inf"))


@libentry()
@triton.jit(do_not_specialize=["N"])
def _atan2_inplace_kernel(X, Y, Out, N, BLOCK: tl.constexpr):
    # ATen: atan2(input, other) = math atan2(y=input, x=other).
    # Quadrant reduction to [0, pi/2] via min/max, one division, odd polynomial
    # in Estrin form, then sign/quadrant fixes with bit-level sign tests so
    # -0.0 is handled like the ATen op.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        ym = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        xm = tl.load(Y + off, mask=mask, other=0.0).to(tl.float32)
        aym = tl.abs(ym)
        axm = tl.abs(xm)
        mn = tl.minimum(aym, axm)
        mx = tl.maximum(aym, axm)
        t = mn / tl.where(axm == 0.0, 1.0, mx)
        t2 = t * t
        t4 = t2 * t2
        t8 = t4 * t4
        q0 = _C0 + t2 * _C1
        q1 = _C2 + t2 * _C3
        q2 = _C4 + t2 * _C5
        r0 = q0 + t4 * q1
        r1 = q2 + t4 * _C6
        p = r0 + t8 * r1
        a = t * p
        # both operands infinite -> t = inf/inf = NaN; the true ratio is 1, so
        # the angle is atan(1) = pi/4 (|y| == |x| keeps the swap fixup off).
        # Guard AFTER the polynomial: a select inside the division->polynomial
        # chain serializes the software pipeline (fp32 4096^2 probe: 0.68x vs
        # 1.11x placed here).
        a = tl.where(mn == _INF, _PIO4, a)
        a = tl.where(aym > axm, _PIO2 - a, a)
        xneg = xm.to(tl.int32, bitcast=True) < 0
        a = tl.where(xneg, _PI - a, a)
        yneg = ym.to(tl.int32, bitcast=True) < 0
        a = tl.where(yneg, -a, a)
        # min/max drop NaN on this backend; propagate it explicitly. The NaN
        # literal stays inline: a module-level tl.constexpr(nan) fails the
        # launch-time global-consistency check (deepcopy makes nan != nan).
        a = tl.where((xm != xm) | (ym != ym), float("nan"), a)
        tl.store(Out + off, a.to(Out.dtype.element_ty), mask=mask)


def _choose_block(N, itemsize):
    # Mirror the delivered atan2.py ladder: half precision stays at 8192,
    # fp32 widens to 65536; tiny shapes pad to at least a 128-wide tile
    # (width-1 vectors are pathological on this backend).
    if N <= 16384:
        return max(128, triton.next_power_of_2(max(N, 1)))
    return 8192 if itemsize <= 2 else 65536


def _launch_inplace(X, Y, out):
    N = X.numel()
    if N == 0:
        return out
    BLOCK = _choose_block(N, X.element_size())
    grid = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic; observed as converterResult=-2 during host-level
    # userland flake windows); retrying the launch recompiles and succeeds.
    # 3 attempts keep deterministic breakage loud while riding out the flake
    # (sinh.py / rad2deg / atan2_.py precedent).
    for attempt in range(3):
        try:
            with torch_device_fn.device(X.device):
                _atan2_inplace_kernel[(grid,)](X, Y, out, N, BLOCK=BLOCK, num_warps=4)
            return out
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "atan2_ flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Flat 1D indexing computes offsets up to ceil(N / BLOCK) * BLOCK - 1; capping
# N at 2**31 - 1 - 65536 keeps the maximum in-tile offset inside int32 for the
# largest BLOCK (65536) used by this kernel (copysign_/relu6_ guard).
_FLAT_MAX_NUMEL = 2**31 - 1 - 65536


def atan2_(input, other):
    logger.debug("GEMS_ENFLAME ATAN2_")
    if not isinstance(other, torch.Tensor):
        # 公版 pointwise_dynamic 无 scalar 档（直调抛 ValueError）；按 aten
        # 弱类型标量语义以 fp32 0-dim factory 包装后走快路径广播。
        other = torch.tensor(other, dtype=torch.float32, device=input.device)
    numel = input.numel()
    if numel == 0:
        return input
    if input.dtype not in _FLAT_DTYPES or numel > _FLAT_MAX_NUMEL:
        return atan2_kernel(input, other, out0=input)
    # aten 原地语义允许 other 广播到 input 的形状；先物化成与输出同形的
    # 连续张量，使扁平索引 kernel 能按输出元素顺序取到对应的分子/分母。
    # 同形连续走免视图快路径（launch-bound 小 shape 的 host 开销敏感）。
    if other.shape == input.shape and other.is_contiguous():
        broadcast_other = other
    else:
        broadcast_other = torch.broadcast_to(other, input.shape)
        if not broadcast_other.is_contiguous():
            broadcast_other = broadcast_other.contiguous()
    if not input.is_contiguous():
        # 非连续输出视图：在连续副本上计算后写回，避免 flat kernel 线性
        # 写入破坏视图布局（arctan2_ 已交付同款契约）。
        contiguous_input = input.contiguous()
        _launch_inplace(contiguous_input, broadcast_other, contiguous_input)
        input.copy_(contiguous_input)
        return input
    return _launch_inplace(input, broadcast_other, input)
