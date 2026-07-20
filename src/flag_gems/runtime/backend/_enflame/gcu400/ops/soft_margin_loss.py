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
BLOCK = 8192


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _soft_margin_loss_elementwise_kernel(
    x_ptr, y_ptr, out_ptr, N_total, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask, other=0.0)
        y = tl.load(y_ptr + off, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        yf = y.to(tl.float32)
        z = -xf * yf
        absz = tl.abs(z)
        vals = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-absz))
        tl.store(out_ptr + off, vals, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _soft_margin_loss_sum_kernel(
    x_ptr, y_ptr, out_ptr, N_total, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask, other=0.0)
        y = tl.load(y_ptr + off, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        yf = y.to(tl.float32)
        z = -xf * yf
        absz = tl.abs(z)
        vals = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-absz))
        vals = tl.where(mask, vals, 0.0)
        acc = tl.sum(vals, axis=0)
        tl.atomic_add(out_ptr, acc)


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == "none":
            return 0
        if r == "mean":
            return 1
        if r == "sum":
            return 2
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def _check_tensors(input: torch.Tensor, target: torch.Tensor):
    if input.device.type != flag_gems.device or target.device.type != flag_gems.device:
        raise AssertionError(
            f"soft_margin_loss: input and target must be {flag_gems.device} tensors for Triton kernel."
        )
    if input.device != target.device:
        raise AssertionError(
            "soft_margin_loss: input and target must be on the same device."
        )
    if input.numel() != target.numel():
        raise AssertionError(
            "soft_margin_loss: input and target must have the same number of elements."
        )
    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    return input, target


def _grid(n_elements):
    return min(triton.cdiv(n_elements, BLOCK), NUM_SIPS * 2)


def soft_margin_loss(input: torch.Tensor, target: torch.Tensor, reduction="mean"):
    logger.debug("GEMS_ENFLAME SOFT_MARGIN_LOSS")
    input, target = _check_tensors(input, target)
    red = _normalize_reduction(reduction)
    n_elements = input.numel()

    if red == 0:
        out = torch.empty_like(input)
        if n_elements == 0:
            return out
        with torch_device_fn.device(input.device):
            _soft_margin_loss_elementwise_kernel[(_grid(n_elements),)](
                input, target, out, n_elements, BLOCK_SIZE=BLOCK, num_warps=4
            )
        return out

    if n_elements == 0:
        if red == 2:
            return torch.zeros((), device=input.device, dtype=input.dtype)
        return torch.full((), float("nan"), device=input.device, dtype=input.dtype)

    tmp_sum = torch.zeros((), device=input.device, dtype=torch.float32)
    with torch_device_fn.device(input.device):
        _soft_margin_loss_sum_kernel[(_grid(n_elements),)](
            input, target, tmp_sum, n_elements, BLOCK_SIZE=BLOCK, num_warps=4
        )
    if red == 2:
        return tmp_sum.to(dtype=input.dtype)
    return (tmp_sum / float(n_elements)).to(dtype=input.dtype)


def soft_margin_loss_out(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction="mean",
    out: torch.Tensor = None,
):
    logger.debug("GEMS_ENFLAME SOFT_MARGIN_LOSS_OUT")
    input, target = _check_tensors(input, target)
    red = _normalize_reduction(reduction)
    n_elements = input.numel()

    if out is None:
        if red == 0:
            out = torch.empty_like(input)
        else:
            out = torch.empty((), device=input.device, dtype=input.dtype)
    else:
        if out.device.type != flag_gems.device:
            raise AssertionError(
                f"soft_margin_loss_out: out must be a {flag_gems.device} tensor."
            )
        if red == 0:
            if out.numel() != n_elements:
                raise AssertionError(
                    "soft_margin_loss_out: for reduction='none', out must match input shape."
                )
        else:
            if out.numel() != 1:
                raise AssertionError(
                    "soft_margin_loss_out: for reduction='sum' or 'mean', out must be a scalar tensor."
                )
        if out.device != input.device:
            raise AssertionError(
                "soft_margin_loss_out: out must be on the same device as input."
            )

    if red == 0:
        if n_elements > 0:
            with torch_device_fn.device(input.device):
                _soft_margin_loss_elementwise_kernel[(_grid(n_elements),)](
                    input, target, out, n_elements, BLOCK_SIZE=BLOCK, num_warps=4
                )
        return out

    if n_elements == 0:
        if red == 2:
            out.fill_(0)
        else:
            out.fill_(float("nan"))
        return out

    tmp_sum = torch.zeros((), device=input.device, dtype=torch.float32)
    with torch_device_fn.device(input.device):
        _soft_margin_loss_sum_kernel[(_grid(n_elements),)](
            input, target, tmp_sum, n_elements, BLOCK_SIZE=BLOCK, num_warps=4
        )
    if red == 2:
        out.fill_(tmp_sum.to(dtype=input.dtype))
    else:
        out.fill_((tmp_sum / float(n_elements)).to(dtype=input.dtype))
    return out
