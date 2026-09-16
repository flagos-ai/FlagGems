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
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


@libentry()
@triton.jit(do_not_specialize=["eps"])
def group_norm_reduce_kernel(
    X,
    Mean,
    Rstd,
    MeanOut,
    RstdOut,
    group_size,
    HW,
    eps,
    BLOCK_HW_SIZE: tl.constexpr,
    WRITE_OUT: tl.constexpr,
):
    pid = ext.program_id(0)
    num_elements = group_size * HW
    base = pid * num_elements

    sum_acc = tl.zeros([BLOCK_HW_SIZE], dtype=tl.float32)
    sumsq_acc = tl.zeros([BLOCK_HW_SIZE], dtype=tl.float32)
    for off in range(0, num_elements, BLOCK_HW_SIZE):
        idx = off + tl.arange(0, BLOCK_HW_SIZE)
        m = idx < num_elements
        x = tl.load(X + base + idx, mask=m, other=0.0).to(tl.float32)
        sum_acc += x
        sumsq_acc += x * x

    mean = tl.sum(sum_acc) / num_elements
    var = tl.sum(sumsq_acc) / num_elements - mean * mean
    rstd = rsqrt(var + eps)
    # Mean/Rstd are the fp32 SCRATCH the normalize kernel reloads at full
    # precision. When the input is fp16/bf16 the returned mean/rstd must be in
    # the input dtype, so we ALSO write MeanOut/RstdOut here (auto-cast on
    # store). That avoids two extra `.to(dtype)` copy kernels after the launch
    # -- on the XPU those dispatch to a gems copy op (~0.085ms launch each),
    # which was the ~0.19ms fp16/bf16 latency floor. For fp32 the scratch IS
    # the output tensor (same dtype), so WRITE_OUT is False and we skip the
    # redundant second store.
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)
    if WRITE_OUT:
        tl.store(MeanOut + pid, mean)
        tl.store(RstdOut + pid, rstd)


@libentry()
@triton.jit
def group_norm_normalize_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    HW,
    num_groups,
    GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    group = pid % num_groups
    base = pid * (group_size * HW)
    ch_base = group * group_size

    mean = tl.load(Mean + pid).to(tl.float32)
    rstd = tl.load(Rstd + pid).to(tl.float32)

    for c in range(0, GROUP_SIZE):
        cbase = base + c * HW
        if W is None:
            weight = 1.0
        else:
            weight = tl.load(W + ch_base + c).to(tl.float32)
        if B is None:
            bias = 0.0
        else:
            bias = tl.load(B + ch_base + c).to(tl.float32)
        for off in range(0, HW, BLOCK_HW_SIZE):
            idx = off + tl.arange(0, BLOCK_HW_SIZE)
            m = idx < HW
            x = tl.load(X + cbase + idx, mask=m, other=0.0).to(tl.float32)
            y = (x - mean) * rstd * weight + bias
            tl.store(Y + cbase + idx, y, mask=m)


@libentry()
@triton.jit
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
    GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    # One program per (n, group); the group is `group_size` channels x HW
    # CONTIGUOUS elements. The reduction is a FLAT 1D per-channel loop over
    # bounded BLOCK_HW_SIZE chunks (GROUP_SIZE is constexpr so the channel
    # loop unrolls statically) with a SCALAR weight per channel -- no 2D
    # [G, HW] tiles. The old 2D form had two XPU pathologies: tiles >= 2x8192
    # hit UNREACHABLE at TritonXPU/Transforms/Legalize.cpp:201 (uni_sram) or
    # silently dropped tl.sum lanes past 8192, and tiles with heavily masked
    # lanes (HxW=4098 -> tail chunk with 2/2048 valid) cost ~90ms of
    # masked-memory-path time. Same structure as native_group_norm_kernel,
    # which is the proven-fast shape on this backend.
    pid = ext.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW
    base = pid * num_elements
    ch_base = group * group_size

    rstd = tl.load(Rstd + pid).to(tl.float32)
    mean = tl.load(Mean + pid).to(tl.float32)

    part2 = tl.zeros([BLOCK_HW_SIZE], dtype=tl.float32)
    part3 = tl.zeros([BLOCK_HW_SIZE], dtype=tl.float32)
    for c in range(0, GROUP_SIZE):
        cbase = base + c * HW
        if W is None:
            weight = 1.0
        else:
            weight = tl.load(W + ch_base + c).to(tl.float32)
        for off in range(0, HW, BLOCK_HW_SIZE):
            idx = off + tl.arange(0, BLOCK_HW_SIZE)
            m = idx < HW
            dy = tl.load(grad_y + cbase + idx, mask=m, other=0.0).to(tl.float32)
            x = tl.load(X + cbase + idx, mask=m, other=0.0).to(tl.float32)
            x_hat = rstd * (x - mean)
            dx_hat = weight * dy
            part2 += dx_hat
            part3 += dx_hat * x_hat

    dx_2 = tl.sum(part2)
    dx_3 = tl.sum(part3)

    for c in range(0, GROUP_SIZE):
        cbase = base + c * HW
        if W is None:
            weight = 1.0
        else:
            weight = tl.load(W + ch_base + c).to(tl.float32)
        for off in range(0, HW, BLOCK_HW_SIZE):
            idx = off + tl.arange(0, BLOCK_HW_SIZE)
            m = idx < HW
            dy = tl.load(grad_y + cbase + idx, mask=m, other=0.0).to(tl.float32)
            x = tl.load(X + cbase + idx, mask=m, other=0.0).to(tl.float32)
            x_hat = rstd * (x - mean)
            dx_hat = weight * dy
            dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / num_elements)
            grad_x_ptr = tl.where(m, cbase + idx, -1)
            tl.store(grad_x + grad_x_ptr, dx, m)


@libentry()
@triton.jit
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
    pid = ext.program_id(0)
    group = pid // group_size
    n_offset = tl.arange(0, BLOCK_N)
    hw_offset = tl.arange(0, BLOCK_HW)
    mr_mask = n_offset < N

    mean_ptr = Mean + group + n_offset * num_groups
    rstd_ptr = Rstd + group + n_offset * num_groups

    mean = tl.load(mean_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]
    rstd = tl.load(rstd_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]

    # Loop over HW in BLOCK_HW chunks: the accumulator tiles stay bounded at
    # BLOCK_N x BLOCK_HW (<= 4096 elements). Without the loop, HxW=4098/16384
    # materialized 16x8192 / 16x16384 tiles -> uni_sram / slow, and the final
    # tl.sum exceeded the 8192-lane safe point (silently dropped lanes).
    dw_acc = tl.zeros([BLOCK_N, BLOCK_HW], dtype=tl.float32)
    db_acc = tl.zeros([BLOCK_N, BLOCK_HW], dtype=tl.float32)
    for off in range(0, HW, BLOCK_HW):
        hw_off = off + hw_offset
        xy_mask = n_offset[:, None] < N and hw_off[None, :] < HW

        dY_ptr = dY + pid * HW + n_offset[:, None] * C * HW + hw_off[None, :]
        x_ptr = X + pid * HW + n_offset[:, None] * C * HW + hw_off[None, :]

        grad_y = tl.load(dY_ptr, mask=xy_mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr, mask=xy_mask, other=0.0)
        x_f32 = x.to(tl.float32)
        dw_acc += (x_f32 - mean) * rstd * grad_y
        db_acc += grad_y

    if dW is not None:
        dw = tl.sum(dw_acc)
        tl.store(dW + pid, dw)
    if dB is not None:
        db = tl.sum(db_acc)
        tl.store(dB + pid, db)


@libentry()
@triton.jit
def weight_bias_backward_kernel_loop(
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
    pid = ext.program_id(0)
    group = pid // group_size

    grad_y_tile = tl.zeros((BLOCK_N, BLOCK_HW), dtype=tl.float32)  # grad_y_tile
    dw_tile = tl.zeros((BLOCK_N, BLOCK_HW), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)

        mean_ptr = Mean + group + n_offset * num_groups
        rstd_ptr = Rstd + group + n_offset * num_groups
        mr_mask = n_offset < N
        mean = tl.load(mean_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]
        rstd = tl.load(rstd_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]

        for start_hw in range(0, HW, BLOCK_HW):
            hw_offset = start_hw + tl.arange(0, BLOCK_HW)
            xy_mask = n_offset[:, None] < N and hw_offset[None, :] < HW
            dY_ptr = dY + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
            grad_y = tl.load(dY_ptr, mask=xy_mask, other=0.0).to(tl.float32)
            grad_y_tile += grad_y

            x_ptr = X + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
            x = tl.load(x_ptr, mask=xy_mask, other=0.0)
            x_f32 = x.to(tl.float32)
            dw_tile += (x_f32 - mean) * rstd * grad_y

    dw = tl.sum(dw_tile)
    db = tl.sum(grad_y_tile)
    tl.store(dW + pid, dw)
    tl.store(dB + pid, db)


def group_norm(input, weight, bias, N, C, HxW, group, eps=1e-05):
    logger.debug("GEMS_KUNLUNXIN GROUP_NORM")

    group_size = triton.cdiv(C, group)
    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    y = torch.empty_like(input)
    # Returned mean/rstd are in the input dtype (matches the generic op).
    # The normalize kernel needs FULL-PRECISION mean/rstd on reload: storing
    # them in a low-precision (fp16/bf16) buffer loses enough bits to fail
    # gems_assert_close. So we keep an fp32 SCRATCH pair for the normalize
    # reload and let the reduce kernel also write the input-dtype outputs
    # directly (see kernel comment). For fp32 input the scratch aliases the
    # output tensors (no extra allocation, the double store is harmless).
    mean = torch.empty((N, group), dtype=input.dtype, device=input.device)
    rstd = torch.empty((N, group), dtype=input.dtype, device=input.device)
    if input.dtype == torch.float32:
        mean_f32, rstd_f32 = mean, rstd
        write_out = False
    else:
        mean_f32 = torch.empty((N, group), dtype=torch.float32, device=input.device)
        rstd_f32 = torch.empty((N, group), dtype=torch.float32, device=input.device)
        write_out = True

    grid = (N * group,)
    block_hw = min(triton.next_power_of_2(HxW), 1024)
    with torch_device_fn.device(input.device):
        if N == 1 and C == 64 and HxW == 1024 and group == 64:
            os.environ["TRITONXPU_OTHER_SIM"] = "1"
            os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
        group_norm_reduce_kernel[grid](
            input,
            mean_f32,
            rstd_f32,
            mean,
            rstd,
            group_size,
            HxW,
            eps,
            BLOCK_HW_SIZE=block_hw,
            WRITE_OUT=write_out,
        )
        group_norm_normalize_kernel[grid](
            input,
            y,
            weight,
            bias,
            mean_f32,
            rstd_f32,
            group_size,
            HxW,
            group,
            GROUP_SIZE=group_size,
            BLOCK_HW_SIZE=block_hw,
        )
        if "TRITONXPU_OTHER_SIM" in os.environ:
            del os.environ["TRITONXPU_OTHER_SIM"]
        if "TRITONXPU_STORE_MASK_SIM" in os.environ:
            del os.environ["TRITONXPU_STORE_MASK_SIM"]

    return y, mean, rstd


def group_norm_backward(
    grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
):
    logger.debug("GEMS_KUNLUNXIN GROUP_NORM_BACKWARD")

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()
    group_size = triton.cdiv(C, group)

    if output_mask[0]:
        grad_inp = torch.empty_like(input)
        grid = (N * group,)
        with torch_device_fn.device(input.device):
            import os

            os.environ["TRITONXPU_OTHER_SIM"] = "1"
            os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
            # Flat 1D per-channel reduction (see kernel docstring): bounded
            # BLOCK_HW_SIZE=1024 chunks, same discipline as the proven
            # native_group_norm_kernel.
            block_hw = min(triton.next_power_of_2(HxW), 1024)
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
                GROUP_SIZE=group_size,
                BLOCK_HW_SIZE=block_hw,
                isCloseUnrollControl=True,
            )
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                del os.environ["TRITONXPU_STORE_MASK_SIM"]

    else:
        grad_inp = None

    if output_mask[1] is False and output_mask[2] is False:
        return grad_inp, None, None

    weight_grad = torch.empty_like(weight) if output_mask[1] else None
    bias_grad = torch.empty_like(weight) if output_mask[2] else None
    with torch_device_fn.device(input.device):
        if N == 32 and C == 32 and HxW == 1024 and group == 8:
            weight_bias_backward_kernel_loop[(C, 1, 1)](
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
                BLOCK_N=1,
                BLOCK_HW=triton.next_power_of_2(HxW),
                isCloseUnrollControl=True,
                isCloseCoreTiling=True,
            )
        else:
            if output_mask[1] is True and output_mask[2] is True:
                isCloseUnrollControl = True
            block_n = triton.next_power_of_2(N)
            # Keep BLOCK_N * BLOCK_HW <= 4096 (same XPU tile-size discipline as
            # the grad-inp kernel above): the kernel loops over HW in chunks.
            # The min 64 pads narrow tiles out of the XPU <=32 BAD interval
            # (2D tiles with inner width <= 32 silently drop stores).
            block_hw = max(
                64,
                min(
                    triton.next_power_of_2(HxW),
                    1 << (max(1, 4096 // block_n).bit_length() - 1),
                ),
            )
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
                BLOCK_N=block_n,
                BLOCK_HW=block_hw,
                isCloseUnrollControl=isCloseUnrollControl,
            )
    return grad_inp, weight_grad, bias_grad
