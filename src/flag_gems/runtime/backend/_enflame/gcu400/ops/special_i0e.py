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
def special_i0e_kernel(x_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        ax = tl.abs(x)

        t_small = ax / 3.75
        t2 = t_small * t_small
        p = 1.0 + t2 * (
            3.5156229
            + t2
            * (
                3.0899424
                + t2
                * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))
            )
        )
        small = p * tl.exp(-ax)

        t = 3.75 / ax
        q = 0.39894228 + t * (
            0.01328592
            + t
            * (
                0.00225319
                + t
                * (
                    -0.00157565
                    + t
                    * (
                        0.00916281
                        + t
                        * (
                            -0.02057706
                            + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))
                        )
                    )
                )
            )
        )
        large = q / tl.sqrt(ax)
        y = tl.where(ax > 3.75, large, small)
        tl.store(out_ptr + off, y, mask=mask)


def _run_special_i0e_kernel(x: torch.Tensor, out: torch.Tensor):
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"Tensors must be {flag_gems.device} tensors")
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), "Unsupported dtype"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"

    x_c = x.contiguous()
    out_c = out.contiguous()
    N_total = out_c.numel()
    if N_total == 0:
        return out

    BLOCK = 8192
    grid_size = min((N_total + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(x.device):
        special_i0e_kernel[(grid_size,)](x_c, out_c, N_total, BLOCK=BLOCK, num_warps=4)

    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_i0e(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME SPECIAL_I0E")
    out = torch.empty_like(x)
    return _run_special_i0e_kernel(x, out)


def special_i0e_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME SPECIAL_I0E_OUT")
    if x.shape != out.shape:
        x = x.expand(out.shape)
    return _run_special_i0e_kernel(x, out)
