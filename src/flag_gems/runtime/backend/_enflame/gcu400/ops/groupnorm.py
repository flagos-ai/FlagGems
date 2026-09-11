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
from flag_gems.utils import libentry, tl_extra_shim

rsqrt = tl_extra_shim.rsqrt
logger = logging.getLogger(__name__)

MAX_BLOCK_HW = 2048
MAX_BLOCK_HW_BACKWARD = 512
GRID_DIM_X = 24


@libentry()
@triton.jit(do_not_specialize=["eps", "group_size", "C", "HW", "num_groups"])
def group_norm_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    C,
    HW,
    num_groups,
    eps,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW

    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)
    wb_offset = group * group_size + group_offset
    wb_mask = wb_offset < C

    Mean_ptr = Mean + pid
    Rstd_ptr = Rstd + pid

    # Welford pass: compute mean and variance in a single pass
    _sum = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)
    _sq = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)
    for off in tl.range(0, HW, BLOCK_HW_SIZE):
        x_block = tl.make_block_ptr(
            base=X + pid * num_elements,
            shape=(group_size, HW),
            strides=(HW, 1),
            offsets=(0, off),
            block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
            order=(1, 0),
        )
        X_val = tl.load(x_block, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        _sum += X_val
        _sq += X_val * X_val

    total_sum = tl.sum(_sum)
    total_sq = tl.sum(_sq)
    mean = total_sum / num_elements
    var = total_sq / num_elements - mean * mean
    rstd = rsqrt(var + eps)

    if W is None:
        weight = 1
    else:
        weight = tl.load(W + wb_offset, mask=wb_mask, other=0.0)[:, None]
    if B is None:
        bias = 0
    else:
        bias = tl.load(B + wb_offset, mask=wb_mask, other=0.0)[:, None]

    # Normalize pass
    for off in tl.range(0, HW, BLOCK_HW_SIZE):
        x_block = tl.make_block_ptr(
            base=X + pid * num_elements,
            shape=(group_size, HW),
            strides=(HW, 1),
            offsets=(0, off),
            block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
            order=(1, 0),
        )
        y_block = tl.make_block_ptr(
            base=Y + pid * num_elements,
            shape=(group_size, HW),
            strides=(HW, 1),
            offsets=(0, off),
            block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
            order=(1, 0),
        )
        X_val = tl.load(x_block, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        x_hat = (X_val - mean) * rstd
        Y_val = x_hat * weight + bias
        tl.store(y_block, Y_val.to(Y.dtype.element_ty), boundary_check=(0, 1))

    tl.store(Mean_ptr, mean)
    tl.store(Rstd_ptr, rstd)


@libentry()
@triton.jit(do_not_specialize=["group_size", "C", "HW"])
def group_norm_backward_kernel(
    grad_y,
    X,
    W,
    Mean,
    Rstd,
    num_groups,
    group_size,
    grad_x,
    C,
    HW,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW

    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)
    wb_offset = group * group_size + group_offset
    wb_mask = wb_offset < C

    rstd = tl.load(Rstd + pid).to(tl.float32)
    mean = tl.load(Mean + pid).to(tl.float32)
    if W is None:
        weight = 1
    else:
        weight = tl.load(W + wb_offset, mask=wb_mask, other=0.0).to(tl.float32)[:, None]

    dx_part2 = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)
    dx_part3 = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)
    for off in tl.range(0, HW, BLOCK_HW_SIZE):
        dy_block = tl.make_block_ptr(
            base=grad_y + pid * num_elements,
            shape=(group_size, HW),
            strides=(HW, 1),
            offsets=(0, off),
            block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
            order=(1, 0),
        )
        x_block = tl.make_block_ptr(
            base=X + pid * num_elements,
            shape=(group_size, HW),
            strides=(HW, 1),
            offsets=(0, off),
            block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
            order=(1, 0),
        )
        dY_val = tl.load(dy_block, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        X_val = tl.load(x_block, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )

        x_hat = rstd * (X_val - mean)
        dx_hat = weight * dY_val
        dx_part2 += dx_hat
        dx_part3 += dx_hat * x_hat

    dx_2 = tl.sum(dx_part2)
    dx_3 = tl.sum(dx_part3)

    for off in tl.range(0, HW, BLOCK_HW_SIZE):
        dy_block = tl.make_block_ptr(
            base=grad_y + pid * num_elements,
            shape=(group_size, HW),
            strides=(HW, 1),
            offsets=(0, off),
            block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
            order=(1, 0),
        )
        x_block = tl.make_block_ptr(
            base=X + pid * num_elements,
            shape=(group_size, HW),
            strides=(HW, 1),
            offsets=(0, off),
            block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
            order=(1, 0),
        )
        dx_block = tl.make_block_ptr(
            base=grad_x + pid * num_elements,
            shape=(group_size, HW),
            strides=(HW, 1),
            offsets=(0, off),
            block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
            order=(1, 0),
        )
        dY_val = tl.load(dy_block, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        X_val = tl.load(x_block, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )

        x_hat = rstd * (X_val - mean)
        dx_hat = weight * dY_val
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / num_elements)

        tl.store(dx_block, dx.to(grad_x.dtype.element_ty), boundary_check=(0, 1))


@libentry()
@triton.jit(do_not_specialize=["N", "C", "HW", "group_size"])
def weight_bias_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    num_groups,
    group_size,
    N,
    C,
    HW,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid // group_size
    n_offset = tl.arange(0, BLOCK_N)
    mr_mask = n_offset < N

    mean = tl.load(Mean + group + n_offset * num_groups, mask=mr_mask, other=0.0).to(
        tl.float32
    )[:, None]
    rstd = tl.load(Rstd + group + n_offset * num_groups, mask=mr_mask, other=0.0).to(
        tl.float32
    )[:, None]

    dw_acc = tl.zeros([BLOCK_N, BLOCK_HW], dtype=tl.float32)
    db_acc = tl.zeros([BLOCK_N, BLOCK_HW], dtype=tl.float32)

    for off in tl.range(0, HW, BLOCK_HW):
        hw_offset = off + tl.arange(0, BLOCK_HW)
        xy_mask = (n_offset[:, None] < N) & (hw_offset[None, :] < HW)

        dY_ptr = dY + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
        x_ptr = X + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]

        grad_y = tl.load(dY_ptr, mask=xy_mask, other=0.0).to(tl.float32)
        x_f32 = tl.load(x_ptr, mask=xy_mask, other=0.0).to(tl.float32)

        dw_acc += (x_f32 - mean) * rstd * grad_y
        db_acc += grad_y

    if dW is not None:
        dw = tl.sum(dw_acc)
        tl.store(dW + pid, dw)
    if dB is not None:
        db = tl.sum(db_acc)
        tl.store(dB + pid, db)


def group_norm(input, weight, bias, N, C, HxW, group, eps=1e-05):
    logger.debug("GEMS_ENFLAME GROUP_NORM")

    group_size = triton.cdiv(C, group)
    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    y = torch.empty_like(input)
    mean = torch.empty((N, group), dtype=input.dtype, device=input.device)
    rstd = torch.empty((N, group), dtype=input.dtype, device=input.device)

    BLOCK_HW_SIZE = min(triton.next_power_of_2(HxW), MAX_BLOCK_HW)

    grid = (N * group,)
    with torch_device_fn.device(input.device):
        group_norm_kernel[grid](
            input,
            y,
            weight,
            bias,
            mean,
            rstd,
            group_size,
            C,
            HxW,
            group,
            eps,
            BLOCK_GROUP_SIZE=triton.next_power_of_2(group_size),
            BLOCK_HW_SIZE=BLOCK_HW_SIZE,
            num_warps=1,
        )
    return y, mean, rstd


def group_norm_backward(
    grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
):
    logger.debug("GEMS_ENFLAME GROUP_NORM_BACKWARD")

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()
    group_size = triton.cdiv(C, group)

    BLOCK_GROUP_SIZE = triton.next_power_of_2(group_size)
    BLOCK_HW = min(triton.next_power_of_2(HxW), MAX_BLOCK_HW_BACKWARD)
    # DSM constraint: reduce BLOCK_HW if tile is too large for make_block_ptr overhead
    while BLOCK_GROUP_SIZE * BLOCK_HW > 2048 and BLOCK_HW > 128:
        BLOCK_HW //= 2

    if output_mask[0]:
        grad_inp = torch.empty_like(input)
        grid = (N * group,)
        with torch_device_fn.device(input.device):
            group_norm_backward_kernel[grid](
                grad_out,
                input,
                weight,
                mean,
                rstd,
                group,
                group_size,
                grad_inp,
                C,
                HxW,
                BLOCK_GROUP_SIZE=BLOCK_GROUP_SIZE,
                BLOCK_HW_SIZE=BLOCK_HW,
                num_warps=1,
            )
    else:
        grad_inp = None

    if output_mask[1] is False and output_mask[2] is False:
        return grad_inp, None, None

    weight_grad = torch.empty_like(weight) if output_mask[1] else None
    bias_grad = torch.empty_like(weight) if output_mask[2] else None
    BLOCK_HW_WB = min(triton.next_power_of_2(HxW), MAX_BLOCK_HW_BACKWARD)
    BLOCK_N_WB = triton.next_power_of_2(N)
    while BLOCK_N_WB * BLOCK_HW_WB > 2048 and BLOCK_HW_WB > 128:
        BLOCK_HW_WB //= 2
    with torch_device_fn.device(input.device):
        weight_bias_backward_kernel[(C, 1, 1)](
            grad_out,
            input,
            mean,
            rstd,
            weight_grad,
            bias_grad,
            group,
            group_size,
            N,
            C,
            HxW,
            BLOCK_N=BLOCK_N_WB,
            BLOCK_HW=BLOCK_HW_WB,
            num_warps=1,
        )
    return grad_inp, weight_grad, bias_grad
