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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .copy import copy_

logger = logging.getLogger(__name__)


def heur_tile_m(args):
    M = args["M"]
    if M <= 512:
        return 128
    return 256


def heur_tile_n(args):
    N = args["N"]
    if N <= 512:
        return 128
    return 256


def heur_tile_k(args):
    # addmm closure probe: on this backend fp16 prefers BK=256 while
    # bf16/fp32 prefer BK=128 (fp32 BK=256 collapses on 4096^3).
    if args.get("TILE_K_CHOICE", 128) == 256:
        return 256
    return 128


def heur_num_warps(args):
    if args["M"] <= 512 and args["N"] <= 512:
        return 4
    return 8


@libentry()
@triton.heuristics(
    {
        "TILE_M": heur_tile_m,
        "TILE_N": heur_tile_n,
        "TILE_K": heur_tile_k,
        "num_warps": heur_num_warps,
    }
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addbmm_kernel(
    A,
    B,
    bias,
    O,
    alpha,
    beta,
    M,
    N,
    K,
    batch,
    batch_stride_a,
    batch_stride_b,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TILE_K_CHOICE,
):
    # 2D [M, N] output; the batch dimension is accumulated inside this program.
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, TILE_M)
    grid_n = tl.cdiv(N, TILE_N)
    # re-order program ID for better L2 reuse along the N dimension
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_am = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_bn = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for bz in range(0, batch):
        a_ptrs = A + bz * batch_stride_a + offs_am[:, None] * K + offs_k[None, :]
        b_ptrs = B + bz * batch_stride_b + offs_k[:, None] * N + offs_bn[None, :]
        for k in range(0, tl.cdiv(K, TILE_K)):
            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * TILE_K),
                other=0.0,
            )
            b_tile = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < K - k * TILE_K) & (offs_bn[None, :] < N),
                other=0.0,
            )
            accumulator += tl.dot(a, b_tile, allow_tf32=False)
            a_ptrs += TILE_K
            b_ptrs += TILE_K * N

    c_ptrs = O + offs_am[:, None] * N + offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    bias_value = tl.load(
        bias + offs_am[:, None] * N + offs_bn[None, :], mask=c_mask, other=0.0
    )

    out = accumulator * alpha + bias_value * beta
    # tl.store converts to the output pointer dtype.
    tl.store(c_ptrs, out, mask=c_mask)


def _direct_addbmm(batch1, batch2, bias, alpha, beta, out=None):
    B, M, K = batch1.shape
    _, _, N = batch2.shape
    batch1 = batch1.contiguous()
    batch2 = batch2.contiguous()
    bias = bias.contiguous()
    if out is None:
        out = torch.empty((M, N), dtype=bias.dtype, device=bias.device)

    tile_k_choice = 256 if batch1.dtype == torch.float16 else 128
    grid_fn = lambda meta: (
        triton.cdiv(M, meta["TILE_M"]) * triton.cdiv(N, meta["TILE_N"]),
    )
    with torch_device_fn.device(batch1.device):
        addbmm_kernel[grid_fn](
            batch1,
            batch2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            B,
            batch1.stride(0),
            batch2.stride(0),
            GROUP_M=8,
            TILE_K_CHOICE=tile_k_choice,
            num_stages=3,
        )
    return out


def addbmm(bias, batch1, batch2, beta=1.0, alpha=1.0):
    logger.debug("GEMS_KUNLUNXIN ADDBMM")
    return _direct_addbmm(batch1, batch2, bias, alpha, beta)


def addbmm_(self, batch1, batch2, *, beta=1.0, alpha=1.0):
    logger.debug("GEMS_KUNLUNXIN ADDBMM_")
    if self.is_contiguous():
        _direct_addbmm(batch1, batch2, self, alpha, beta, out=self)
    else:
        tmp = _direct_addbmm(batch1, batch2, self, alpha, beta, out=None)
        copy_(self, tmp)
    return self
