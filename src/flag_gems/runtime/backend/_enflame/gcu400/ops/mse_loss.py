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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_NP2 = triton.next_power_of_2
_CDIV = triton.cdiv
_SINGLE_THRESHOLD = 4096
_NONE_BS = 32768
_NONE_PD_THRESHOLD = 2097152
_MAX_PID = 48
_SMALL_MAX = 131072
_LARGE_THRESHOLD = 4194304
_SMALL_BS = 4096
_MID_BS = 8192
_LARGE_BS = 32768


@triton.jit
def mse_single_kernel(
    inp, target, out, M, BLOCK_SIZE: tl.constexpr, reduction: tl.constexpr
):
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    inp_val = tl.load(inp + offset, mask=mask, other=0.0).to(tl.float32)
    tgt_val = tl.load(target + offset, mask=mask, other=0.0).to(tl.float32)
    d = inp_val - tgt_val
    result = tl.sum(d * d)
    if reduction == 1:
        result = result / M
    tl.store(out, result)


@triton.jit
def mse_gs_kernel(
    inp,
    target,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    N_FULL: tl.constexpr,
    NUM_PID: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    NATIVE_ACC: tl.constexpr,
    reduction: tl.constexpr,
):
    pid = tl.program_id(0)
    s = 0.0
    for blk in range(pid, N_FULL, NUM_PID):
        offset = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(inp + offset)
        y = tl.load(target + offset)
        if NATIVE_ACC:
            d = x - y
            s += tl.sum(d * d).to(tl.float32)
        else:
            d = (x - y).to(tl.float32)
            s += tl.sum(d * d)
    if HAS_TAIL:
        if pid == N_FULL % NUM_PID:
            offset = N_FULL * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offset < M
            x = tl.load(inp + offset, mask=mask, other=0.0)
            y = tl.load(target + offset, mask=mask, other=0.0)
            if NATIVE_ACC:
                d = x - y
                s += tl.sum(d * d).to(tl.float32)
            else:
                d = (x - y).to(tl.float32)
                s += tl.sum(d * d)
    if reduction == 1:
        s = s / M
    tl.store(mid + pid, s)


@triton.jit
def mse_vec_kernel(
    inp,
    target,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    N_FULL: tl.constexpr,
    NUM_PID: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    NATIVE_ACC: tl.constexpr,
    reduction: tl.constexpr,
):
    # Vector-accumulator variant: elementwise running accumulator across the
    # grid-stride loop, a single reduction tree at the end. This amortizes the
    # per-iteration cross-warp tree cost of mse_gs_kernel (measured -14~-19%
    # on 16-bit dtypes at large shapes, -15~-24% on mid shapes on GCU400).
    pid = tl.program_id(0)
    if NATIVE_ACC:
        acc = tl.zeros([BLOCK_SIZE], dtype=inp.dtype.element_ty)
    else:
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for blk in range(pid, N_FULL, NUM_PID):
        offset = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(inp + offset)
        y = tl.load(target + offset)
        if NATIVE_ACC:
            d = x - y
            acc += d * d
        else:
            d = (x - y).to(tl.float32)
            acc += d * d
    if HAS_TAIL:
        if pid == N_FULL % NUM_PID:
            offset = N_FULL * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offset < M
            x = tl.load(inp + offset, mask=mask, other=0.0)
            y = tl.load(target + offset, mask=mask, other=0.0)
            if NATIVE_ACC:
                d = x - y
                acc += d * d
            else:
                d = (x - y).to(tl.float32)
                acc += d * d
    s = tl.sum(acc.to(tl.float32))
    if reduction == 1:
        s = s / M
    tl.store(mid + pid, s)


@triton.jit
def mse_reduce_k2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    mid_val = tl.load(mid + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(out, tl.sum(mid_val))


@triton.jit
def mse_none_kernel(inp, target, out, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    x = tl.load(inp + offset, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(target + offset, mask=mask, other=0.0).to(tl.float32)
    d = x - y
    tl.store(out + offset, d * d, mask=mask)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def mse_none_pd(x, y):
    return (x - y) * (x - y)


def mse_loss(inp, target, reduction=1):
    logger.debug("GEMS_ENFLAME MSE_LOSS")
    M = inp.numel()
    dtype = inp.dtype

    if reduction == 0:
        if M >= _NONE_PD_THRESHOLD:
            return mse_none_pd(inp, target)
        inp_c = inp if inp.is_contiguous() else inp.contiguous()
        tgt_c = target if target.is_contiguous() else target.contiguous()
        out = torch.empty_like(inp)
        bs = min(_NONE_BS, _NP2(M))
        mse_none_kernel[(_CDIV(M, bs), 1, 1)](inp_c, tgt_c, out, M, bs)
        return out

    inp_c = inp if inp.is_contiguous() else inp.contiguous()
    tgt_c = target if target.is_contiguous() else target.contiguous()

    out = torch.empty([], dtype=dtype, device=inp.device)

    if M <= _SINGLE_THRESHOLD:
        mse_single_kernel[(1, 1, 1)](inp_c, tgt_c, out, M, _NP2(M), reduction)
        return out

    if dtype == torch.float32 and M > _LARGE_THRESHOLD:
        # fp32 at large shapes: per-iteration tree structure is register-cheap
        # and reaches the fp32 read wall (2.79TB/s), outperforming the vector
        # accumulator whose 32768-wide fp32 acc vector adds pressure.
        block_size = _LARGE_BS
        n_full = M // block_size
        has_tail = (M % block_size) != 0
        num_pid = _MAX_PID
        mid = torch.empty(num_pid, dtype=torch.float32, device=inp.device)
        mse_gs_kernel[(num_pid, 1, 1)](
            inp_c,
            tgt_c,
            mid,
            M,
            block_size,
            n_full,
            num_pid,
            has_tail,
            False,
            reduction,
            num_warps=2,
        )
    else:
        native_acc = dtype == torch.bfloat16
        if M <= _SMALL_MAX:
            block_size = _SMALL_BS
            num_warps = 2
        elif M <= _LARGE_THRESHOLD:
            block_size = _MID_BS
            num_warps = 4
        else:
            block_size = _LARGE_BS
            num_warps = 2
        n_full = M // block_size
        has_tail = (M % block_size) != 0
        num_pid = min(_MAX_PID, n_full)
        mid = torch.empty(num_pid, dtype=torch.float32, device=inp.device)
        mse_vec_kernel[(num_pid, 1, 1)](
            inp_c,
            tgt_c,
            mid,
            M,
            block_size,
            n_full,
            num_pid,
            has_tail,
            native_acc,
            reduction,
            num_warps=num_warps,
        )
    mse_reduce_k2[(1, 1, 1)](mid, out, num_pid, _NP2(num_pid))
    return out
