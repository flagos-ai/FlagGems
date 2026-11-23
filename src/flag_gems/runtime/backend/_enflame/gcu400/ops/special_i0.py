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

from flag_gems.runtime import device
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Flat grid-stride path: caps CTAs at 48, picks tile and warp count by numel/dtype, and drops
# the mask when numel divides the tile. Same empirical parameters as the relu flat path.
_FLAT_GRID_CAP = 48
_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FLAT_MAX_NUMEL = 2**31 - 1 - 65536


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _special_i0_flat_kernel(
    x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK)
    for block_id in tl.range(pid, num_blocks, num_pids):
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + offsets)
        else:
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        x_f32 = x.to(tl.float32)
        ax = tl.abs(x_f32)

        # |x| <= 3.75 branch (division by a constant turned into multiply by reciprocal)
        t = x_f32 * 0.26666666666666666
        y = t * t
        p_small = 1.0 + y * (
            3.5156229
            + y
            * (
                3.0899424
                + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))
            )
        )

        # |x| > 3.75 branch
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
        # exp(|x|)/sqrt(|x|) becomes exp2 + rsqrt: exp2 is a hardware primitive and 1/sqrt uses
        # the rsqrt approximation (relative error 2^-22, far below the fp32 case tolerance).
        res_big = tl.exp2(ax * 1.4426950408889634) * p_big * tl.math.rsqrt(ax)

        use_small = ax <= 3.75
        res = tl.where(use_small, p_small, res_big)

        res = res.to(x.dtype)
        if EVEN:
            tl.store(out_ptr + offsets, res)
        else:
            tl.store(out_ptr + offsets, res, mask=offsets < n_elements)


def _flat_config(numel, dtype):
    # Tiny sizes: one CTA covers all elements with a single tile, avoiding extra loop iterations.
    if numel <= 4096:
        block = 1 << (numel - 1).bit_length() if numel > 1 else 1
        return (256 if block < 256 else block), 2
    if dtype is torch.float32:
        # f32 needs a large tile (131072 elements = 512KB) to saturate the DTE; on GCU400/L600
        # this measured 80% faster than 32768 at 2^24 and 100% faster at 2^29.
        return (131072, 2) if numel > (1 << 20) else (2048, 2)
        # f16/bf16 must use a small tile: f32 intermediates and the f16 source are live at the
        # same time, and from 8192 on register/dsm pressure costs 3~4x (2^26 measured 1.02ms at
        # 2048, 1.29ms at 8192 and 1.03ms at 16384 with 2.5x jitter), so 2048 is the stable optimum.
    return 2048, 2


def _run_flat(x_ptr, out_ptr, numel, dtype):
    block, num_warps = _flat_config(numel, dtype)
    grid = min(-(-numel // block), _FLAT_GRID_CAP)
    _special_i0_flat_kernel[(grid,)](
        x_ptr,
        out_ptr,
        numel,
        BLOCK=block,
        EVEN=(numel % block == 0),
        num_warps=num_warps,
    )


@triton.jit
def _special_i0_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pids = tl.num_programs(axis=0)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        ax = tl.abs(x_f32)

        # Small region: |x| <= 3.75
        t = x_f32 / 3.75
        y = t * t
        p_small = 1.0 + y * (
            3.5156229
            + y
            * (
                3.0899424
                + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))
            )
        )

        # Large region: |x| > 3.75
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

        use_small = ax <= 3.75
        res = tl.where(use_small, p_small, res_big)

        # Cast back to input dtype for storage
        res = res.to(x.dtype)
        tl.store(out_ptr + offsets, res, mask=mask)


def _launch_special_i0(out: torch.Tensor, x: torch.Tensor):
    assert (
        x.device.type == device.name and out.device.type == device.name
    ), f"Input and output must be on the {device.name} device"
    assert (
        out.numel() == x.numel()
    ), "Input and output must have the same number of elements"
    assert out.device == x.device, "Input and output must be on the same device"

    x_in = x
    out_in = out

    # Ensure floating point compute
    if not x_in.is_floating_point():
        x_in = x_in.to(torch.get_default_dtype())

    # Cast input to match the desired output dtype if needed
    if x_in.dtype != out_in.dtype:
        x_in = x_in.to(out_in.dtype)

    x_contig = x_in.contiguous()
    out_was_noncontig = not out_in.is_contiguous()
    out_contig = out_in.contiguous() if out_was_noncontig else out_in

    n_elements = out_contig.numel()
    if n_elements > 0:
        if x_contig.dtype in _FLAT_DTYPES and n_elements <= _FLAT_MAX_NUMEL:
            _run_flat(x_contig, out_contig, n_elements, x_contig.dtype)
        else:
            # Fallback: not float16/bf16/fp32, or numel outside the int32 offset safety domain
            BLOCK_SIZE = 1024
            grid = (min(triton.cdiv(n_elements, BLOCK_SIZE), 65535),)
            _special_i0_kernel[grid](
                x_contig, out_contig, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )

    if out_was_noncontig:
        out_in.copy_(out_contig)
    return out_in


def special_i0(x: torch.Tensor):
    """
    ATen wrapper: special_i0(Tensor self) -> Tensor
    Computes the zeroth order modified Bessel function of the first kind for each element.
    """
    logger.debug("GEMS_ENFLAME SPECIAL_I0")
    if x.device.type != device.name:
        raise ValueError(
            f"special_i0: input tensor must be on the {device.name} device"
        )
    out_dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
    out = torch.empty_like(x.to(dtype=out_dtype), dtype=out_dtype, device=x.device)
    _launch_special_i0(out, x)
    return out


def special_i0_out(x: torch.Tensor, out: torch.Tensor):
    """
    ATen wrapper: special_i0.out(Tensor self, Tensor out) -> Tensor
    """
    logger.debug("GEMS_ENFLAME SPECIAL_I0_OUT")
    if not (x.device.type == device.name and out.device.type == device.name):
        raise ValueError(
            "special_i0_out: input and output tensors must be on the "
            f"{device.name} device"
        )
    if not out.is_floating_point():
        raise TypeError("special_i0_out: output tensor must be a floating point type")
    if x.numel() != out.numel():
        raise ValueError(
            "special_i0_out: input and output must have the same number of elements"
        )
    _launch_special_i0(out, x)
    return out
