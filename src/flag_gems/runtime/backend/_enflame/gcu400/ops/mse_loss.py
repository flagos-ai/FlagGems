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

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

_NP2 = triton.next_power_of_2
_CDIV = triton.cdiv
_REDUCE_BS = 65536
_NONE_BS = 32768
_NONE_PD_THRESHOLD = 2097152


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
def mse_reduce_k1(
    inp, target, mid, M, BLOCK_SIZE: tl.constexpr, reduction: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    inp_val = tl.load(inp + offset, mask=mask, other=0.0).to(tl.float32)
    tgt_val = tl.load(target + offset, mask=mask, other=0.0).to(tl.float32)
    d = inp_val - tgt_val
    s = tl.sum(d * d)
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

    if M <= 1024:
        mse_single_kernel[(1, 1, 1)](inp_c, tgt_c, out, M, _NP2(M), reduction)
    else:
        mid_size = _CDIV(M, _REDUCE_BS)
        mid = torch.empty(mid_size, dtype=torch.float32, device=inp.device)
        mse_reduce_k1[(mid_size, 1, 1)](inp_c, tgt_c, mid, M, _REDUCE_BS, reduction)
        mse_reduce_k2[(1, 1, 1)](mid, out, mid_size, _NP2(mid_size))

    return out
