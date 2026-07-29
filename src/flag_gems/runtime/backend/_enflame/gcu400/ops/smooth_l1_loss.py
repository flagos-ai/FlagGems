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

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

device = device.name
logger = logging.getLogger(__name__)

NUM_SIPS = 24
BLOCK = 32768


@libentry()
@triton.jit(do_not_specialize=["N_total", "beta"])
def _smooth_l1_loss_kernel(
    inp,
    target,
    out,
    N_total,
    beta,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        mask = off < N_total
        inp_val = tl.load(inp + off, mask=mask, other=0.0).to(tl.float32)
        target_val = tl.load(target + off, mask=mask, other=0.0).to(tl.float32)
        diff = tl.abs(inp_val - target_val)
        if BETA_IS_ZERO:
            loss = diff
        else:
            loss = tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        tl.store(out + off, loss, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total", "beta"])
def _smooth_l1_loss_partial_sum_kernel(
    inp,
    target,
    mid,
    N_total,
    beta,
    BETA_IS_ZERO: tl.constexpr,
    REDUCTION_MEAN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        mask = off < N_total
        inp_val = tl.load(inp + off, mask=mask, other=0.0).to(tl.float32)
        target_val = tl.load(target + off, mask=mask, other=0.0).to(tl.float32)
        diff = tl.abs(inp_val - target_val)
        if BETA_IS_ZERO:
            loss = diff
        else:
            loss = tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        loss = tl.where(mask, loss, 0.0)
        acc = tl.sum(loss, axis=0)
        if REDUCTION_MEAN:
            acc = acc / N_total
        tl.store(mid + block_id, acc)


@libentry()
@triton.jit(do_not_specialize=["mid_size"])
def _smooth_l1_loss_sum_kernel(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    acc = tl.zeros([], dtype=tl.float32)
    num_blocks = (mid_size + BLOCK_MID - 1) // BLOCK_MID
    for i in tl.range(0, num_blocks):
        offset = i * BLOCK_MID + tl.arange(0, BLOCK_MID)
        mask = offset < mid_size
        vals = tl.load(mid + offset, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)
    tl.store(out, acc)


@libentry()
@triton.jit(do_not_specialize=["N_total", "reduction_elements", "beta"])
def _smooth_l1_loss_backward_kernel(
    grad_output,
    inp,
    target,
    out,
    N_total,
    reduction_elements,
    beta,
    BETA_IS_ZERO: tl.constexpr,
    REDUCTION_MEAN: tl.constexpr,
    GRAD_OUTPUT_SCALAR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_total
    inp_val = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
    target_val = tl.load(target + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = inp_val - target_val
    if BETA_IS_ZERO:
        grad = tl.where(diff == 0.0, float("nan"), tl.where(diff > 0.0, 1.0, -1.0))
    else:
        grad = tl.where(diff < -beta, -1.0, tl.where(diff > beta, 1.0, diff / beta))
    if GRAD_OUTPUT_SCALAR:
        grad_out = tl.load(grad_output).to(tl.float32)
        if REDUCTION_MEAN:
            grad_out = grad_out * (1.0 / reduction_elements)
        grad_out = tl.full(offsets.shape, grad_out, tl.float32)
    else:
        grad_out = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
        if REDUCTION_MEAN:
            grad_out = grad_out * (1.0 / reduction_elements)
    tl.store(out + offsets, grad * grad_out, mask=mask)


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        if reduction == "none":
            return 0
        if reduction == "mean":
            return 1
        if reduction == "sum":
            return 2
    elif isinstance(reduction, int) and reduction in (0, 1, 2):
        return reduction
    raise ValueError("reduction must be one of 'none', 'mean', or 'sum'")


def _check_input(input, target, beta):
    if beta < 0:
        raise RuntimeError("smooth_l1_loss does not support negative values for beta.")
    if input.device.type != device or target.device.type != device:
        raise AssertionError("smooth_l1_loss: input and target must be CUDA tensors.")
    if input.device != target.device:
        raise AssertionError(
            "smooth_l1_loss: input and target must be on the same device."
        )
    input, target = torch.broadcast_tensors(input, target)
    return input.contiguous(), target.contiguous()


def _check_backward_input(grad_output, input, target, beta):
    input, target = _check_input(input, target, beta)
    reduction_elements = input.numel()
    if grad_output.device.type != device:
        raise AssertionError(
            "smooth_l1_loss_backward: grad_output must be a CUDA tensor."
        )
    if grad_output.device != input.device:
        raise AssertionError(
            "smooth_l1_loss_backward: grad_output must be on the same device."
        )
    if grad_output.numel() != 1:
        grad_output = torch.broadcast_to(grad_output, input.shape)
    return grad_output.contiguous(), input, target, reduction_elements


def _empty_reduction(input, reduction):
    if reduction == 0:
        return torch.empty_like(input)
    if reduction == 1:
        return torch.full((), float("nan"), device=input.device, dtype=input.dtype)
    return torch.zeros((), device=input.device, dtype=input.dtype)


def _grid(n_elements):
    return min(triton.cdiv(n_elements, BLOCK), NUM_SIPS * 2)


def _smooth_l1_loss_none(input, target, beta, out=None):
    n_elements = input.numel()
    if out is None:
        out = torch.empty_like(input)
        out_contiguous = out
    else:
        if out.device != input.device:
            raise AssertionError("smooth_l1_loss.out: out must be on the same device.")
        if tuple(out.shape) != tuple(input.shape):
            out.resize_(input.shape)
        out_contiguous = out if out.is_contiguous() else torch.empty_like(input)

    if n_elements > 0:
        grid = _grid(n_elements)
        beta_f = float(beta)
        with torch_device_fn.device(input.device):
            _smooth_l1_loss_kernel[(grid,)](
                input,
                target,
                out_contiguous,
                n_elements,
                beta_f,
                BETA_IS_ZERO=beta_f == 0.0,
                BLOCK_SIZE=BLOCK,
                num_warps=4,
            )
    if out_contiguous is not out:
        out.copy_(out_contiguous)
    return out


def _smooth_l1_loss_reduce(input, target, beta, reduction, out=None):
    n_elements = input.numel()
    if n_elements == 0:
        result = _empty_reduction(input, reduction)
        if out is None:
            return result
        if out.device != input.device:
            raise AssertionError("smooth_l1_loss.out: out must be on the same device.")
        if out.dim() != 0:
            out.resize_(())
        out.copy_(result)
        return out

    mid_size = triton.cdiv(n_elements, BLOCK)
    block_mid = min(triton.next_power_of_2(mid_size), BLOCK)
    mid = torch.empty((mid_size,), device=input.device, dtype=torch.float32)
    result = out
    if result is None:
        result = torch.empty((), device=input.device, dtype=input.dtype)
    else:
        if result.device != input.device:
            raise AssertionError("smooth_l1_loss.out: out must be on the same device.")
        if result.dim() != 0:
            result.resize_(())

    beta_f = float(beta)
    with torch_device_fn.device(input.device):
        _smooth_l1_loss_partial_sum_kernel[(_grid(n_elements),)](
            input,
            target,
            mid,
            n_elements,
            beta_f,
            BETA_IS_ZERO=beta_f == 0.0,
            REDUCTION_MEAN=reduction == 1,
            BLOCK_SIZE=BLOCK,
            num_warps=4,
        )
        _smooth_l1_loss_sum_kernel[(1,)](
            mid, result, mid_size, BLOCK_MID=block_mid, num_warps=4
        )
    return result


def smooth_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction=1,
    beta: float = 1.0,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME SMOOTH_L1_LOSS")
    reduction = _normalize_reduction(reduction)
    input, target = _check_input(input, target, float(beta))
    if reduction == 0:
        return _smooth_l1_loss_none(input, target, float(beta))
    return _smooth_l1_loss_reduce(input, target, float(beta), reduction)


def smooth_l1_loss_out(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction=1,
    beta: float = 1.0,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME SMOOTH_L1_LOSS_OUT")
    reduction = _normalize_reduction(reduction)
    input, target = _check_input(input, target, float(beta))
    if reduction == 0:
        return _smooth_l1_loss_none(input, target, float(beta), out=out)
    return _smooth_l1_loss_reduce(input, target, float(beta), reduction, out=out)


def smooth_l1_loss_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    reduction,
    beta: float,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME SMOOTH_L1_LOSS_BACKWARD")
    reduction = _normalize_reduction(reduction)
    grad_output, input, target, reduction_elements = _check_backward_input(
        grad_output, input, target, float(beta)
    )
    out = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        return out
    block_size = BLOCK
    if n_elements // block_size > 65535:
        block_size = 65536
    grid = triton.cdiv(n_elements, block_size)
    beta_f = float(beta)
    with torch_device_fn.device(input.device):
        _smooth_l1_loss_backward_kernel[(grid,)](
            grad_output,
            input,
            target,
            out,
            n_elements,
            reduction_elements,
            beta_f,
            BETA_IS_ZERO=beta_f == 0.0,
            REDUCTION_MEAN=reduction == 1,
            GRAD_OUTPUT_SCALAR=grad_output.numel() == 1,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
    return out
