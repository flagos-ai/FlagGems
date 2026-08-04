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
def _copy_kernel(in_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(in_ptr + off, mask=mask)
        tl.store(out_ptr + off, x, mask=mask)


def _find_input(*args, **kwargs):
    if len(args) > 0 and isinstance(args[0], torch.Tensor):
        return args[0]
    if "self" in kwargs and isinstance(kwargs["self"], torch.Tensor):
        return kwargs["self"]
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, torch.Tensor):
            return v
    raise ValueError("lift_fresh_copy expects a Tensor argument")


def lift_fresh_copy(*args, **kwargs):
    logger.debug("GEMS_ENFLAME LIFT_FRESH_COPY")
    x = _find_input(*args, **kwargs)
    if x.device.type != flag_gems.device:
        raise ValueError(
            f"lift_fresh_copy Triton kernel requires a {flag_gems.device} tensor"
        )

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig, memory_format=torch.contiguous_format)
    N = x_contig.numel()
    if N == 0:
        return out.view_as(x_contig)

    BLOCK = 8192
    grid = min((N + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(x_contig.device):
        _copy_kernel[(grid,)](x_contig, out, N, BLOCK=BLOCK, num_warps=4)
    return out.view_as(x_contig)


def lift_fresh_copy_out(x: torch.Tensor, out: torch.Tensor = None):
    logger.debug("GEMS_ENFLAME LIFT_FRESH_COPY_OUT")
    if x is None or not isinstance(x, torch.Tensor):
        raise ValueError("lift_fresh_copy_out expects 'x' to be a Tensor")
    if x.device.type != flag_gems.device:
        raise ValueError(
            f"lift_fresh_copy_out Triton kernel requires {flag_gems.device} tensors"
        )

    x_contig = x.contiguous()
    if out is None:
        out = torch.empty_like(x_contig, memory_format=torch.contiguous_format)
    else:
        if out.device.type != flag_gems.device:
            raise ValueError(f"Output tensor 'out' must be on {flag_gems.device}")
        if out.dtype != x_contig.dtype:
            raise ValueError("Output tensor 'out' must have the same dtype as input")
        if out.numel() != x_contig.numel() or not out.is_contiguous():
            out.resize_(x_contig.shape)
            if not out.is_contiguous():
                out = out.contiguous()

    N = x_contig.numel()
    if N == 0:
        return out.view_as(x_contig)

    BLOCK = 8192
    grid = min((N + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(x_contig.device):
        _copy_kernel[(grid,)](x_contig, out, N, BLOCK=BLOCK, num_warps=4)
    return out.view_as(x_contig)
