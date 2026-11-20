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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

_NW = 1


@triton.jit
def _rope_half_kernel(
    x_ptr,
    c_ptr,
    s_ptr,
    o_ptr,
    D: tl.constexpr,
    BH: tl.constexpr,
    EXACT: tl.constexpr,
    ONE: tl.constexpr,
):
    if ONE:
        # Host-known single row: the per-program base is a compile-time 0, so
        # every load/store address is a constant and no program-id arithmetic
        # is emitted.
        base = 0
    else:
        base = tl.program_id(0) * D
    H: tl.constexpr = D // 2
    if EXACT:
        # One flat contiguous 128B tile per tensor (single arange, 1D), plus a
        # duplicate load of x at the partner lane (tp = t ^ H, an XOR on the
        # constant lane vector: lane j <-> lane j+H).  The sign comes from a
        # single compare on the same lane vector, so no 2D address arithmetic
        # and no recomputed (1-k)*H+j offset vector are emitted, while the
        # store stays one contiguous 64-lane access.  Round 3 proved that
        # replacing the duplicate load with a register permutation of the same
        # tile costs ~0.69us per op, so re-reading the 128B is the cheap way to
        # obtain the partner lane.
        t = tl.arange(0, D)
        tp = t ^ H
        x2 = tl.load(x_ptr + base + t)
        xp2 = tl.load(x_ptr + base + tp)
        c2 = tl.load(c_ptr + base + t)
        s2 = tl.load(s_ptr + base + t)
        sgn = tl.where(t < H, -1.0, 1.0).to(tl.bfloat16)
        r2 = x2 * c2 + sgn * (xp2 * s2)
        tl.store(o_ptr + base + t, r2)
    else:
        j = tl.arange(0, BH)
        off_a = j
        off_b = H + j
        m = j < H
        xa = tl.load(x_ptr + base + off_a, mask=m, other=0.0).to(tl.float32)
        xb = tl.load(x_ptr + base + off_b, mask=m, other=0.0).to(tl.float32)
        ca = tl.load(c_ptr + base + off_a, mask=m, other=0.0).to(tl.float32)
        cb = tl.load(c_ptr + base + off_b, mask=m, other=0.0).to(tl.float32)
        sa = tl.load(s_ptr + base + off_a, mask=m, other=0.0).to(tl.float32)
        sb = tl.load(s_ptr + base + off_b, mask=m, other=0.0).to(tl.float32)
        ra = xa * ca - xb * sa
        rb = xb * cb + xa * sb
        tl.store(o_ptr + base + off_a, ra.to(tl.bfloat16), mask=m)
        tl.store(o_ptr + base + off_b, rb.to(tl.bfloat16), mask=m)


@triton.jit
def _rope_inter_kernel(
    x_ptr,
    c_ptr,
    s_ptr,
    o_ptr,
    D: tl.constexpr,
    BH: tl.constexpr,
    EXACT: tl.constexpr,
    ONE: tl.constexpr,
):
    if ONE:
        base = 0
    else:
        base = tl.program_id(0) * D
    H: tl.constexpr = D // 2
    j = tl.arange(0, BH)
    off_e = 2 * j
    off_o = 2 * j + 1
    if EXACT:
        # Constant-lane form, same structure that measured 1.340us for the half
        # body: one flat contiguous 128B tile per tensor, a duplicate load of x
        # at the adjacent partner lane (tp = t ^ 1) and a sign from the lane
        # parity.  No reshape/split/join and no 2D address arithmetic is
        # emitted, and the store is one contiguous 64-lane access.
        t = tl.arange(0, D)
        tp = t ^ 1
        x2 = tl.load(x_ptr + base + t)
        xp2 = tl.load(x_ptr + base + tp)
        c2 = tl.load(c_ptr + base + t)
        s2 = tl.load(s_ptr + base + t)
        sgn = tl.where((t & 1) == 0, -1.0, 1.0).to(tl.bfloat16)
        r2 = x2 * c2 + sgn * (xp2 * s2)
        tl.store(o_ptr + base + t, r2)
    else:
        m = j < H
        xe = tl.load(x_ptr + base + off_e, mask=m, other=0.0).to(tl.float32)
        xo = tl.load(x_ptr + base + off_o, mask=m, other=0.0).to(tl.float32)
        ce = tl.load(c_ptr + base + off_e, mask=m, other=0.0).to(tl.float32)
        co = tl.load(c_ptr + base + off_o, mask=m, other=0.0).to(tl.float32)
        se = tl.load(s_ptr + base + off_e, mask=m, other=0.0).to(tl.float32)
        so = tl.load(s_ptr + base + off_o, mask=m, other=0.0).to(tl.float32)
        re = xe * ce - xo * se
        ro = xo * co + xe * so
        tl.store(o_ptr + base + off_e, re.to(tl.bfloat16), mask=m)
        tl.store(o_ptr + base + off_o, ro.to(tl.bfloat16), mask=m)


def _launch(input, r1, r2, rotary_mode="half", rotate=None):
    x = input
    D = int(x.shape[-1])
    out = torch.empty_like(x)

    xf = x.view(-1)
    cf = r1.view(-1)
    sf = r2.view(-1)
    of = out.view(-1)

    n_rows = x.numel() // D
    BH = triton.next_power_of_2(D // 2)
    exact = (D % 2 == 0) and (BH * 2 == D)

    mode = rotary_mode
    if isinstance(mode, str):
        mode = mode.strip().lower()
    grid = (n_rows,)
    one = n_rows == 1
    if mode == "interleave" or mode == "interleaved":
        _rope_inter_kernel[grid](
            xf, cf, sf, of, D=D, BH=BH, EXACT=exact, ONE=one, num_warps=2, num_stages=1
        )
    else:
        _rope_half_kernel[grid](
            xf, cf, sf, of, D=D, BH=BH, EXACT=exact, ONE=one, num_warps=2, num_stages=1
        )
    return out


def npu_rotary_mul(input, r1, r2, rotary_mode="half", rotate=None):
    """Elementwise rotary embedding for equal-shaped BF16 sin, cos and activation tensors."""
    if (
        rotate is not None
        or rotary_mode not in ("half", "interleave")
        or input.shape != r1.shape
        or input.shape != r2.shape
        or input.ndim != 4
        or input.shape[-1] != 64
        or not input.numel()
    ):
        raise ValueError(
            "expected equal 4D operands with 64 channels and half or interleave mode"
        )
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (input, r1, r2)):
        raise ValueError("expected contiguous BF16 operands")
    if (
        input.device.type != "npu"
        or r1.device != input.device
        or r2.device != input.device
    ):
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(input.device):
        return _launch(input, r1, r2, rotary_mode, rotate)
