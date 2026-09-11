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
def special_i1_kernel(x_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        ax = tl.abs(x)

        y = x / 3.75
        y2 = y * y
        p = 0.00032411
        p = 0.00301532 + y2 * p
        p = 0.02658733 + y2 * p
        p = 0.15084934 + y2 * p
        p = 0.51498869 + y2 * p
        p = 0.87890594 + y2 * p
        p = 0.5 + y2 * p
        ans_small = x * p

        t = 3.75 / tl.maximum(ax, 1e-20)
        q = -0.00420059
        q = 0.01787654 + t * q
        q = -0.02895312 + t * q
        q = 0.02282967 + t * q
        q = -0.01031555 + t * q
        q = 0.00163801 + t * q
        q = -0.00362018 + t * q
        q = -0.03988024 + t * q
        q = 0.39894228 + t * q
        pref = tl.exp(ax) / tl.sqrt(tl.maximum(ax, 1e-20))
        ans_large = pref * q
        ans_large = tl.where(x < 0, -ans_large, ans_large)

        ans = tl.where(ax <= 3.75, ans_small, ans_large)
        tl.store(out_ptr + off, ans, mask=mask)


def _launch_special_i1(x: torch.Tensor, out: torch.Tensor):
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"Tensors must be {flag_gems.device} tensors")
    N_total = x.numel()
    if N_total == 0:
        return

    BLOCK = 8192
    grid_size = min((N_total + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(x.device):
        special_i1_kernel[(grid_size,)](
            x.contiguous(), out.contiguous(), N_total, BLOCK=BLOCK, num_warps=4
        )


def special_i1(self: torch.Tensor):
    logger.debug("GEMS_ENFLAME SPECIAL_I1")
    x_c = self.contiguous()
    out = torch.empty_like(x_c)
    _launch_special_i1(x_c, out)
    if self.layout == torch.strided and self.is_contiguous():
        return out
    return out.view_as(self)


def special_i1_out(self: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME SPECIAL_I1_OUT")
    if out.dtype != self.dtype:
        raise TypeError("out dtype must match input dtype")
    if out.device != self.device:
        raise TypeError("out device must match input device")
    x_c = self.contiguous()
    out_c = out.contiguous()
    _launch_special_i1(x_c, out_c)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out
