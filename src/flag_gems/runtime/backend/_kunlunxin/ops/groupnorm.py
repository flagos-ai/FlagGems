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
    N,
    PART_W,
    PART_B,
    GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
    WRITE_W: tl.constexpr,
    WRITE_B: tl.constexpr,
):
    pid = ext.program_id(0)
    n = pid // num_groups
    group = pid % num_groups
    num_elements = group_size * HW
    base = pid * num_elements
    ch_base = group * group_size

    rstd = tl.load(Rstd + pid).to(tl.float32)
    mean = tl.load(Mean + pid).to(tl.float32)
    # Reciprocal once per program: a per-lane divide by a runtime scalar takes
    # the slow full-precision div sequence (~1.9x a multiply on this backend).
    inv_n = 1.0 / num_elements.to(tl.float32)

    dx_2 = 0.0
    dx_3 = 0.0
    # TWO accumulators, not four. The weight is a SCALAR per channel, so
    #   sum(dx_hat)        = weight * sum(dy)
    #   sum(dx_hat * x_hat) = weight * sum(dy * x_hat)
    # which is exactly the pair dW/dB need, and multiplying by the scalar once
    # per channel (after the partial sums) keeps the register footprint at the
    # pre-fusion level. Materialising all four accumulators overflowed the
    # local-memory budget (`out of resource: uni_sram`) on the single-channel
    # group shapes.
    partw = tl.zeros([BLOCK_HW_SIZE], dtype=tl.float32)
    partb = tl.zeros([BLOCK_HW_SIZE], dtype=tl.float32)
    if EVEN:
        for c in range(0, GROUP_SIZE):
            cbase = base + c * HW
            if W is None:
                weight = 1.0
            else:
                weight = tl.load(W + ch_base + c).to(tl.float32)
            partw = tl.zeros([BLOCK_HW_SIZE], dtype=tl.float32)
            partb = tl.zeros([BLOCK_HW_SIZE], dtype=tl.float32)
            for off in range(0, HW, BLOCK_HW_SIZE):
                idx = off + tl.arange(0, BLOCK_HW_SIZE)
                dy = tl.load(grad_y + cbase + idx).to(tl.float32)
                x = tl.load(X + cbase + idx).to(tl.float32)
                x_hat = rstd * (x - mean)
                partw += dy * x_hat
                partb += dy
            dw = tl.sum(partw)
            db = tl.sum(partb)
            dx_2 += weight * db
            dx_3 += weight * dw
            if WRITE_W:
                tl.store(PART_W + n * C + ch_base + c, dw)
            if WRITE_B:
                tl.store(PART_B + n * C + ch_base + c, db)
    else:
        for c in range(0, GROUP_SIZE):
            cbase = base + c * HW
            if W is None:
                weight = 1.0
            else:
                weight = tl.load(W + ch_base + c).to(tl.float32)
            partw = tl.zeros([BLOCK_HW_SIZE], dtype=tl.float32)
            partb = tl.zeros([BLOCK_HW_SIZE], dtype=tl.float32)
            for off in range(0, HW, BLOCK_HW_SIZE):
                idx = off + tl.arange(0, BLOCK_HW_SIZE)
                m = idx < HW
                dy = tl.load(grad_y + cbase + idx, mask=m, other=0.0).to(tl.float32)
                x = tl.load(X + cbase + idx, mask=m, other=0.0).to(tl.float32)
                x_hat = rstd * (x - mean)
                partw += dy * x_hat
                partb += dy
            dw = tl.sum(partw)
            db = tl.sum(partb)
            dx_2 += weight * db
            dx_3 += weight * dw
            if WRITE_W:
                tl.store(PART_W + n * C + ch_base + c, dw)
            if WRITE_B:
                tl.store(PART_B + n * C + ch_base + c, db)

    if EVEN:
        for c in range(0, GROUP_SIZE):
            cbase = base + c * HW
            if W is None:
                weight = 1.0
            else:
                weight = tl.load(W + ch_base + c).to(tl.float32)
            for off in range(0, HW, BLOCK_HW_SIZE):
                idx = off + tl.arange(0, BLOCK_HW_SIZE)
                dy = tl.load(grad_y + cbase + idx).to(tl.float32)
                x = tl.load(X + cbase + idx).to(tl.float32)
                x_hat = rstd * (x - mean)
                dx_hat = weight * dy
                dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) * inv_n)
                tl.store(grad_x + cbase + idx, dx)
    else:
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
                dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) * inv_n)
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
    EVEN: tl.constexpr,
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
    #
    # EVEN (N == BLOCK_N and HW % BLOCK_HW == 0) drops the per-tile mask from
    # BOTH loads. On this backend a masked 2D load lowers to a per-lane select
    # even when the mask is statically all-true for the launched tile, so the
    # 2D masked version measured ~104 GB/s on the (2,64,128,128) case.
    dw_acc = tl.zeros([BLOCK_N, BLOCK_HW], dtype=tl.float32)
    db_acc = tl.zeros([BLOCK_N, BLOCK_HW], dtype=tl.float32)
    if EVEN:
        for off in range(0, HW, BLOCK_HW):
            hw_off = off + hw_offset
            dY_ptr = dY + pid * HW + n_offset[:, None] * C * HW + hw_off[None, :]
            x_ptr = X + pid * HW + n_offset[:, None] * C * HW + hw_off[None, :]

            grad_y = tl.load(dY_ptr).to(tl.float32)
            x = tl.load(x_ptr)
            x_f32 = x.to(tl.float32)
            dw_acc += (x_f32 - mean) * rstd * grad_y
            db_acc += grad_y
    else:
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


@libentry()
@triton.jit
def weight_bias_fold_kernel(
    PART_W,
    PART_B,
    dW,
    dB,
    N,
    C,
    BLOCK_N: tl.constexpr,
    WRITE_W: tl.constexpr,
    WRITE_B: tl.constexpr,
):
    """Fold the per-(batch, channel) partials left by the grad-input kernel.

    `group_norm_backward_kernel` already walks every byte dW/dB need, so it
    leaves one partial per (n, channel) instead of a second kernel re-reading
    dY and X. Here one program per channel sums those N partials -- a handful
    of floats, i.e. the only cross-batch reduction left. When N == 1 the
    partials ARE the outputs and this kernel is not launched at all.
    """
    pid = ext.program_id(0)
    off = tl.arange(0, BLOCK_N)
    mask = off < N
    if WRITE_W:
        vw = tl.load(PART_W + off * C + pid, mask=mask, other=0.0).to(tl.float32)
        tl.store(dW + pid, tl.sum(vw))
    if WRITE_B:
        vb = tl.load(PART_B + off * C + pid, mask=mask, other=0.0).to(tl.float32)
        tl.store(dB + pid, tl.sum(vb))


_BLOCK_MISS = object()
# (dtype, HxW, GROUP_SIZE, write_w, write_b) -> lane width, or None when the
# fused form did not compile at ANY width (caller then takes the unfused path).
_GRAD_BLOCK = {}


def _block_ladder(HxW):
    """Candidate 1D lane widths, widest first (1024 is the backend's widest
    legal online-reduction tile; 64 is the smallest the accumulator form
    accepts).

    The 64 floor is a CORRECTNESS requirement, not tuning: the backend widens
    `tl.arange(0, n)` to `round_up(n, 64)` lanes (extending the iota), so a
    nominal 1-lane tile really issues 64 lanes. With the mask-free EVEN store
    that turns a 1-element row into a 64-element write past the tensor, and with
    a mask it still writes the -1 fallback address 63 times. Keeping the tile
    at >= 64 leaves a sub-64 row on the masked path exactly like the original
    kernel (EVEN then never applies, because HW % 64 != 0 for HW < 64).
    """
    top = min(max(triton.next_power_of_2(HxW), 64), 1024)
    widths = [top]
    while widths[-1] > 64:
        widths.append(max(64, widths[-1] // 2))
    return widths


def _launch_grad_kernel(
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
    N,
    part_w,
    part_b,
    block_hw,
    write_w,
    write_b,
):
    group_norm_backward_kernel[(N * group,)](
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
        N,
        part_w,
        part_b,
        GROUP_SIZE=group_size,
        BLOCK_HW_SIZE=block_hw,
        EVEN=(HxW % block_hw == 0),
        WRITE_W=write_w,
        WRITE_B=write_b,
        isCloseUnrollControl=True,
        buffer_size_limit=2048,
    )


def _launch_grad_ladder(
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
    N,
    part_w,
    part_b,
    write_w,
    write_b,
):
    """Launch the fused grad-input kernel at the widest width that compiles.

    The safe 1D-tile width on this backend is NOT monotonic and it changes with
    the element size and with how many tiles are live: adding the two dW/dB
    scalar stores pushes fp32 (16,3,16,16) over the local-memory budget at 256
    lanes while fp16 compiles at 1024. So the width is probed once per
    (dtype, shape, write mask) and memoised; `False` means every width failed
    and the caller must run the unfused form.
    """
    key = (input.dtype, HxW, group_size, bool(write_w), bool(write_b))
    picked = _GRAD_BLOCK.get(key, _BLOCK_MISS)
    if picked is not _BLOCK_MISS:
        if picked is None:
            return False
        _launch_grad_kernel(
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
            N,
            part_w,
            part_b,
            picked,
            write_w,
            write_b,
        )
        return True
    for width in _block_ladder(HxW):
        try:
            _launch_grad_kernel(
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
                N,
                part_w,
                part_b,
                width,
                write_w,
                write_b,
            )
        except Exception as exc:  # noqa: BLE001 - only the sram budget is retried
            text = "%s %s" % (type(exc).__name__, exc)
            if "uni_sram" not in text and "OutOfResources" not in text:
                raise
            logger.debug(
                "GEMS_KUNLUNXIN GROUP_NORM_BACKWARD fused width %d rejected (%s)",
                width,
                text.split("\n")[0],
            )
            continue
        _GRAD_BLOCK[key] = width
        logger.debug(
            "GEMS_KUNLUNXIN GROUP_NORM_BACKWARD fused width %d (HxW=%d)",
            width,
            HxW,
        )
        return True
    _GRAD_BLOCK[key] = None
    return False


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

    if N == 0 or HxW == 0:
        # aten answers this degenerate shape with an empty dX and ZERO dW/dB;
        # there is no element to reduce, and any launch would only pay fixed
        # cost (HxW == 0 additionally divides by zero when sizing the tile).
        return (
            torch.empty_like(input) if output_mask[0] else None,
            (
                torch.zeros(weight.shape, dtype=weight.dtype, device=weight.device)
                if output_mask[1]
                else None
            ),
            (
                torch.zeros(weight.shape, dtype=weight.dtype, device=weight.device)
                if output_mask[2]
                else None
            ),
        )

    want_w = bool(output_mask[1])
    want_b = bool(output_mask[2])
    # The N==32/C==32/HxW==1024/group==8 shape keeps its dedicated kernel (that
    # was the fix for a miscompile on this shape), so it opts out of the fused
    # dW/dB path below.
    special_wb = N == 32 and C == 32 and HxW == 1024 and group == 8
    # Fusing dW/dB into the grad-input kernel needs that kernel to run (it is
    # the pass that reads the bytes) and needs the shape to stay off the
    # dedicated special-case kernel.
    fuse_wb = (want_w or want_b) and bool(output_mask[0]) and not special_wb

    weight_grad = torch.empty_like(weight) if want_w else None
    bias_grad = torch.empty_like(weight) if want_b else None
    if fuse_wb and N > 1:
        # fp32 partials: the per-batch values are summed here, before the final
        # cast, so a bf16/fp16 input keeps fp32 reduction precision.
        part_w = (
            torch.empty((N * C,), dtype=torch.float32, device=input.device)
            if want_w
            else None
        )
        part_b = (
            torch.empty((N * C,), dtype=torch.float32, device=input.device)
            if want_b
            else None
        )
    else:
        # N == 1: one partial per channel IS the output (the store casts).
        part_w = weight_grad
        part_b = bias_grad

    if output_mask[0]:
        grad_inp = torch.empty_like(input)
        with torch_device_fn.device(input.device):
            import os

            os.environ["TRITONXPU_OTHER_SIM"] = "1"
            os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
            # Flat 1D per-channel reduction (see kernel docstring): bounded
            # BLOCK_HW_SIZE=1024 chunks, same discipline as the proven
            # native_group_norm_kernel.
            # 1D lane width for the reduce/store loops: 1024 is the widest legal
            # online-reduction tile on this backend. The FUSED form keeps four
            # tiles live (dY, X, the store and the two accumulators), and the
            # width that fits that budget is dtype/shape dependent and NOT
            # monotonic, so it is probed once (see _launch_grad_ladder).
            fused_ok = False
            if fuse_wb:
                fused_ok = _launch_grad_ladder(
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
                    N,
                    part_w,
                    part_b,
                    want_w,
                    want_b,
                )
            if not fused_ok:
                # Either dW/dB are not needed (nothing to fuse) or every fused
                # width overflowed the local-memory budget: run the plain
                # grad-input kernel and let the weight/bias kernel below do the
                # per-channel reduction.
                fuse_wb = False
                # >=64 floor for the same reason as _block_ladder.
                block_hw = min(max(triton.next_power_of_2(HxW), 64), 1024)
                _launch_grad_kernel(
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
                    N,
                    None,
                    None,
                    block_hw,
                    False,
                    False,
                )
            if fuse_wb and N > 1:
                weight_bias_fold_kernel[(C, 1, 1)](
                    part_w,
                    part_b,
                    weight_grad,
                    bias_grad,
                    N,
                    C,
                    BLOCK_N=triton.next_power_of_2(N),
                    WRITE_W=want_w,
                    WRITE_B=want_b,
                )
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                del os.environ["TRITONXPU_STORE_MASK_SIM"]

    else:
        grad_inp = None

    if not want_w and not want_b:
        return grad_inp, None, None

    with torch_device_fn.device(input.device):
        if special_wb:
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
        elif not fuse_wb:
            # HEAD only assigned this in the both-outputs branch and then read it
            # unconditionally, so `output_mask = [True, True, False]` (or
            # [True, False, True]) raised UnboundLocalError. Default it.
            isCloseUnrollControl = output_mask[1] is True and output_mask[2] is True
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
                EVEN=(N == block_n and HxW % block_hw == 0),
                isCloseUnrollControl=isCloseUnrollControl,
            )
    return grad_inp, weight_grad, bias_grad
