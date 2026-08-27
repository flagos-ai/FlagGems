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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def i0_kernel(x_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        ax = tl.abs(x)

        t = x / 3.75
        y = t * t
        p_small = 1.0 + y * (
            3.5156229
            + y
            * (
                3.0899424
                + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))
            )
        )

        yb = 3.75 / ax
        p_big = 0.39894228 + yb * (
            0.01328592
            + yb
            * (
                0.00225319
                + yb
                * (
                    -0.00157565
                    + yb
                    * (
                        0.00916281
                        + yb
                        * (
                            -0.02057706
                            + yb * (0.02635537 + yb * (-0.01647633 + yb * 0.00392377))
                        )
                    )
                )
            )
        )
        res_big = tl.exp(ax) * p_big / tl.sqrt(ax)
        res = tl.where(ax <= 3.75, p_small, res_big)
        tl.store(out_ptr + off, res, mask=mask)


def _launch_i0(out: torch.Tensor, x: torch.Tensor):
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"Input and output must be {flag_gems.device} tensors")

    x_in = x
    out_in = out
    if not x_in.is_floating_point():
        x_in = x_in.to(torch.get_default_dtype())
    if x_in.dtype != out_in.dtype:
        x_in = x_in.to(out_in.dtype)

    x_contig = x_in.contiguous()
    out_was_noncontig = not out_in.is_contiguous()
    out_contig = out_in.contiguous() if out_was_noncontig else out_in

    N_total = out_contig.numel()
    if N_total == 0:
        return out_in

    BLOCK = 8192
    grid_size = min((N_total + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(x_contig.device):
        i0_kernel[(grid_size,)](x_contig, out_contig, N_total, BLOCK=BLOCK, num_warps=4)

    if out_was_noncontig:
        out_in.copy_(out_contig)
    return out_in


def i0(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME I0")
    if x.device.type != flag_gems.device:
        raise ValueError(f"i0: input tensor must be on {flag_gems.device} device")
    out_dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
    out = torch.empty_like(x.to(dtype=out_dtype), dtype=out_dtype, device=x.device)
    _launch_i0(out, x)
    return out


def i0_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME I0_OUT")
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(
            f"i0_out: input and output tensors must be on {flag_gems.device} device"
        )
    if not out.is_floating_point():
        raise TypeError("i0_out: output tensor must be a floating point type")
    if x.numel() != out.numel():
        raise ValueError(
            "i0_out: input and output must have the same number of elements"
        )
    _launch_i0(out, x)
    return out


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def i0_kernel_(x_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        ax = tl.abs(x)

        t = x / 3.75
        y = t * t
        p_small = 1.0 + y * (
            3.5156229
            + y
            * (
                3.0899424
                + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))
            )
        )

        yb = 3.75 / ax
        p_big = 0.39894228 + yb * (
            0.01328592
            + yb
            * (
                0.00225319
                + yb
                * (
                    -0.00157565
                    + yb
                    * (
                        0.00916281
                        + yb
                        * (
                            -0.02057706
                            + yb * (0.02635537 + yb * (-0.01647633 + yb * 0.00392377))
                        )
                    )
                )
            )
        )
        res_big = tl.exp(ax) * p_big / tl.sqrt(ax)
        res = tl.where(ax <= 3.75, p_small, res_big)
        tl.store(x_ptr + off, res, mask=mask)


def i0_(*args, **kwargs):
    logger.debug("GEMS_ENFLAME I0_")
    x = args[0] if args else kwargs.get("self", kwargs.get("input", None))
    if x is None:
        raise ValueError(
            "i0_ expects a tensor as the first positional argument or in keyword 'input'/'self'/'x'."
        )
    if x.device.type != flag_gems.device:
        raise AssertionError(f"Input tensor must be on a {flag_gems.device} device.")
    if not x.is_contiguous():
        raise AssertionError("Input tensor must be contiguous.")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise AssertionError(
            "Unsupported dtype for i0_. Supported: float16, bfloat16, float32, float64."
        )

    N_total = x.numel()
    if N_total == 0:
        return x

    BLOCK = 8192
    grid_size = min((N_total + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(x.device):
        i0_kernel_[(grid_size,)](x, N_total, BLOCK=BLOCK, num_warps=4)
    return x
