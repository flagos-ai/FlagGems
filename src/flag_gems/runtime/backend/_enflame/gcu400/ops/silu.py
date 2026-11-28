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
from flag_gems.utils.triton_lang_extension import div_rn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 48

# int32 offset safe domain: the masked tail block evaluates
# block_id * BLOCK + arange for masked lanes as well, reaching up to
# num_blocks * BLOCK - 1; keep n_total <= 2^31 - 2^17 (BLOCK <= 131072)
# so flat int32 offsets cannot wrap negative on the tail block.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)

_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# GCU400 retune of the flat silu family (2026-11-17, interleaved
# do_bench(median, w1000/r100) A/B probes on L600, official bench kernel
# mode).  The old fixed config (BLOCK from _choose_block, nw2, always
# masked, grid cap 96) left the 16-bit streams far off the copy floor
# because the per-element f32 math chain (exp + reciprocal + mul) was
# exposed behind narrow tiles, and put the whole 4096-element core shape
# on a single CTA (grid = cdiv(4096, 8192) = 1).
#
# Division lowering rule discovered by ablation (probe2/3): a plain `/`
# with a *variable* numerator lowers to a slow path on GCU400
# (x - x/(1+exp(x)): 1360us vs 674us; dy*u/(t*t): 2135us vs 1150us),
# while the constant-numerator reciprocal `1.0 / (1.0 + e)` is fast and
# matches torch bit-for-bit.  tl.fdiv(..., ieee_rounding=False) only
# helps when the denominator is a fresh add (silu_fdiv == silu_cur
# within 0.5us) and still hits the slow path for product denominators.
# All bodies below therefore keep the single fast reciprocal form.
#
# Measured (median, interleaved, vs old config):
#   silu_   16-bit 512M: 833 -> 674us (-19%, copy floor 666)
#   silu_   16-bit 16.7M: 31.6 -> 27.8us (-12%)
#   silu_   16-bit 4096: 4.10 -> 3.14us (launch floor 3.11)
#   silu_   fp32 512M: 1333 -> 1328us (copy floor 1330)
#   silu_   fp32 16.7M: 48.7 -> 47.1us
#   silu_   fp32 4096: 4.63 -> 3.27us (-29%)
#   silu    16-bit 512M (oop): 828 -> 742us (-10%)
#   silu_backward 16-bit 512M: 1230 -> 1162us (-5.5%)
#   silu_backward 16-bit 16.7M: 48.9 -> 44.9us (-8%)
#   silu_backward fp32 16.7M: 88.0 -> 75.7us (-14%)
#   silu_backward 16-bit 4096: 4.04 -> 3.27us (-19%)
#
# The backward body uses the algebraically identical factored form
# dy * sigma * (1 + x - x * sigma) (one ALU op fewer than
# 1 + x * (1 - sigma)); verified against aten CPU silu_backward
# (fp32 max_abs 4.8e-7 vs atol 1e-4 gate) and value-by-value equal to
# the old body on the ±inf/NaN/±0 special battery (both saturate
# sigma to 0 for x <= -20 through the same GCU exp path as the native
# aten kernel).


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def silu_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(x_ptr + off).to(tl.float32)
            sigma = 1.0 / (1.0 + tl.exp(-x))
            tl.store(out_ptr + off, x * sigma)
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
            sigma = 1.0 / (1.0 + tl.exp(-x))
            tl.store(out_ptr + off, x * sigma, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def silu_backward_kernel(x, dy):
    dy_fp32 = dy.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    return dx


# Direct-pointer flat kernel, same shape as silu_flat_kernel.  The
# pointwise_dynamic variant is much slower on gcu400: div_rn lowers to a
# slow scalar libdevice call, and the block-pointer form adds overhead.
# A plain division is fast and matches torch bit-for-bit for fp16 here.
@libentry()
@triton.jit(do_not_specialize=["N_total"])
def silu_backward_flat_kernel(
    x_ptr,
    dy_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(x_ptr + off).to(tl.float32)
            dy = tl.load(dy_ptr + off).to(tl.float32)
            sigma = 1.0 / (1.0 + tl.exp(-x))
            dx = dy * sigma * (1.0 + x - x * sigma)
            tl.store(out_ptr + off, dx)
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
            dy = tl.load(dy_ptr + off, mask=mask).to(tl.float32)
            sigma = 1.0 / (1.0 + tl.exp(-x))
            dx = dy * sigma * (1.0 + x - x * sigma)
            tl.store(out_ptr + off, dx, mask=mask)


def _choose_block_silu(n_total, itemsize):
    # Forward/in-place (1R1W) tile ladder, boundary-probed:
    # - tiny (<=8192) rides the ~3.1us launch floor with a <=2048-wide
    #   tile so several CTAs share the work (the old config put all
    #   4096 elements on one CTA);
    # - 32768 -> 65536 tile switch measured at 2^22 (hardswish family
    #   boundary, same 1R1W f32-math shape);
    # - 16-bit 65536 -> 131072 switch measured at 2^25 (48.6 vs 50.2us)
    #   and 2^26 (91.1 vs 90.5us); fp32 never wants 131072 (DSM line,
    #   hardsigmoid/hardswish precedents) and keeps 65536/nw8.
    if n_total <= 8192:
        return min(max(triton.next_power_of_2(n_total), 64), 2048), 1
    if n_total <= 65536:
        return 8192, 8
    if n_total <= (1 << 22):
        return 32768, 8
    if itemsize == 4 or n_total <= (1 << 25):
        return 65536, 8
    return 131072, 4


def _choose_block_silu_backward(n_total, itemsize):
    # Backward (2R1W) tile ladder, boundary-probed.  Wide tiles only
    # hide the math for 16-bit; fp32 streams want 32768 at the same
    # element count (75.7/83.7us at 16.7M fp32 for 32768/65536).
    # In the sub-launch-quantum band a single 65536 tile wins for
    # divisible sizes (4.09us at 65536) while odd sizes prefer more
    # smaller blocks (6.39us at 131073 on 32768 vs 7.99 on 65536).
    if n_total <= 8192:
        return min(max(triton.next_power_of_2(triton.cdiv(n_total, 4)), 256), 1024), 1
    if n_total <= (1 << 18):
        return (65536 if n_total % 65536 == 0 else 32768), 2
    if itemsize == 4:
        return 32768, 2
    return 65536, 2


def _launch_flat(kernel, args, n_total, block, num_warps):
    even = n_total % block == 0
    if not even and num_warps > 2:
        # The tail mask costs nothing at nw2 but is exposed at nw8
        # (36.5 vs 28.4us at 16.7M, hardswish family measurement);
        # odd sizes ride the masked arm on nw2.
        num_warps = 2
    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared
    # nodes); retrying the launch recompiles and succeeds (abs.py /
    # sinh.py / hardswish_.py precedents for the same toolchain family).
    for attempt in range(3):
        try:
            kernel[(grid_size,)](
                *args, n_total, BLOCK=block, EVEN=even, num_warps=num_warps
            )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "silu flat kernel launch failed (attempt %d): %s", attempt + 1, exc
            )


def silu(self):
    logger.debug("GEMS_ENFLAME SILU")
    if not isinstance(self, torch.Tensor):
        raise TypeError("silu(): expected a Tensor")
    if self.dtype not in _FLAT_DTYPES or self.numel() >= _INT32_SAFE_LIMIT:
        from flag_gems.ops.silu import silu as _public

        return _public(self)

    inp = self.contiguous()
    n_total = inp.numel()
    if n_total == 0:
        return torch.empty_like(inp)
    out = torch.empty_like(inp)

    block, num_warps = _choose_block_silu(n_total, inp.element_size())
    with torch_device_fn.device(inp.device):
        _launch_flat(silu_flat_kernel, (inp, out), n_total, block, num_warps)

    return out


def silu_backward(grad_output, self):
    logger.debug("GEMS_ENFLAME SILU_BACKWARD")
    # Mixed dtype/device needs the generic promotion logic; everything
    # else goes through the flat kernel.
    if self.dtype != grad_output.dtype or self.device != grad_output.device:
        return silu_backward_kernel(self, grad_output)
    if self.dtype not in _FLAT_DTYPES or self.numel() >= _INT32_SAFE_LIMIT:
        from flag_gems.ops.silu import silu_backward as _public

        return _public(grad_output, self)

    inp = self.contiguous()
    grad = grad_output.contiguous()
    n_total = inp.numel()
    if n_total == 0:
        return torch.empty_like(inp)
    out = torch.empty_like(inp)

    block, num_warps = _choose_block_silu_backward(n_total, inp.element_size())
    with torch_device_fn.device(inp.device):
        _launch_flat(
            silu_backward_flat_kernel, (inp, grad, out), n_total, block, num_warps
        )

    return out


def silu_(A):
    logger.debug("GEMS_ENFLAME SILU_")
    if not isinstance(A, torch.Tensor):
        raise TypeError("silu_(): expected a Tensor")
    if A.dtype not in _FLAT_DTYPES or A.numel() >= _INT32_SAFE_LIMIT:
        from flag_gems.ops.silu import silu_ as _public

        return _public(A)

    inp = A.contiguous()
    n_total = inp.numel()
    if n_total == 0:
        return A

    block, num_warps = _choose_block_silu(n_total, inp.element_size())
    with torch_device_fn.device(inp.device):
        _launch_flat(silu_flat_kernel, (inp, inp), n_total, block, num_warps)
    # For non-contiguous inputs contiguous() makes a copy, so the result must be copied back into A with A's stride,
    # otherwise writing A by flat offset would land in the wrong memory location.
    if inp is not A:
        A.copy_(inp)

    return A
