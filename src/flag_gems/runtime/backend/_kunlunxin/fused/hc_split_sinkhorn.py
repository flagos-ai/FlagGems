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

import sys

import torch
import triton
import triton.language as tl

from flag_gems.fused.mhc.hc_split_sinkhorn import (
    hc_split_sinkhorn as _general_hc_split_sinkhorn,
)

_SUPPORTED_HC = (2, 4)
_BLOCK_N = 64


@triton.jit(do_not_specialize=["num_tokens"])
def _hc_split_sinkhorn_kernel_hc4(
    mixes_ptr,  # (N, 24) f32, N = 真实行数（无需 BLOCK 对齐）
    hc_scale_ptr,  # (3,) f32
    hc_base_ptr,  # (24,) f32
    pre_ptr,  # (N_pad, 4) f32, N_pad % BLOCK_N == 0
    post_ptr,  # (N_pad, 4) f32
    comb_ptr,  # (N_pad, 16) f32
    num_tokens,  # 真实行数；[num_tokens, N_pad) 为冗余行（输入读被 clamp）
    BLOCK_N: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    """Vectorized split + 4x4 Sinkhorn, exact tiles, no masks."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    # Pad 行 (offs >= num_tokens) 的输入读被 clamp 到最后一行，输出落在
    # pre/post/comb 尾部、由主机侧截断丢弃。免去对 mixes 做全量 pad 拷贝
    # (ATen constant_pad_nd)，同时消除输出缓冲越界写。
    offs_r = tl.minimum(offs, num_tokens - 1)
    base = offs_r * 24

    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)

    m0 = tl.load(mixes_ptr + base + 0)
    m1 = tl.load(mixes_ptr + base + 1)
    m2 = tl.load(mixes_ptr + base + 2)
    m3 = tl.load(mixes_ptr + base + 3)
    m4 = tl.load(mixes_ptr + base + 4)
    m5 = tl.load(mixes_ptr + base + 5)
    m6 = tl.load(mixes_ptr + base + 6)
    m7 = tl.load(mixes_ptr + base + 7)

    b0 = tl.load(hc_base_ptr + 0)
    b1 = tl.load(hc_base_ptr + 1)
    b2 = tl.load(hc_base_ptr + 2)
    b3 = tl.load(hc_base_ptr + 3)
    b4 = tl.load(hc_base_ptr + 4)
    b5 = tl.load(hc_base_ptr + 5)
    b6 = tl.load(hc_base_ptr + 6)
    b7 = tl.load(hc_base_ptr + 7)

    tl.store(pre_ptr + offs * 4 + 0, tl.sigmoid(m0 * scale_0 + b0) + EPS)
    tl.store(pre_ptr + offs * 4 + 1, tl.sigmoid(m1 * scale_0 + b1) + EPS)
    tl.store(pre_ptr + offs * 4 + 2, tl.sigmoid(m2 * scale_0 + b2) + EPS)
    tl.store(pre_ptr + offs * 4 + 3, tl.sigmoid(m3 * scale_0 + b3) + EPS)

    tl.store(post_ptr + offs * 4 + 0, 2.0 * tl.sigmoid(m4 * scale_1 + b4))
    tl.store(post_ptr + offs * 4 + 1, 2.0 * tl.sigmoid(m5 * scale_1 + b5))
    tl.store(post_ptr + offs * 4 + 2, 2.0 * tl.sigmoid(m6 * scale_1 + b6))
    tl.store(post_ptr + offs * 4 + 3, 2.0 * tl.sigmoid(m7 * scale_1 + b7))

    cb = 8
    b8 = tl.load(hc_base_ptr + cb + 0)
    b9 = tl.load(hc_base_ptr + cb + 1)
    b10 = tl.load(hc_base_ptr + cb + 2)
    b11 = tl.load(hc_base_ptr + cb + 3)
    b12 = tl.load(hc_base_ptr + cb + 4)
    b13 = tl.load(hc_base_ptr + cb + 5)
    b14 = tl.load(hc_base_ptr + cb + 6)
    b15 = tl.load(hc_base_ptr + cb + 7)
    b16 = tl.load(hc_base_ptr + cb + 8)
    b17 = tl.load(hc_base_ptr + cb + 9)
    b18 = tl.load(hc_base_ptr + cb + 10)
    b19 = tl.load(hc_base_ptr + cb + 11)
    b20 = tl.load(hc_base_ptr + cb + 12)
    b21 = tl.load(hc_base_ptr + cb + 13)
    b22 = tl.load(hc_base_ptr + cb + 14)
    b23 = tl.load(hc_base_ptr + cb + 15)

    cm_00 = tl.load(mixes_ptr + base + cb + 0) * scale_2 + b8
    cm_01 = tl.load(mixes_ptr + base + cb + 1) * scale_2 + b9
    cm_02 = tl.load(mixes_ptr + base + cb + 2) * scale_2 + b10
    cm_03 = tl.load(mixes_ptr + base + cb + 3) * scale_2 + b11
    cm_10 = tl.load(mixes_ptr + base + cb + 4) * scale_2 + b12
    cm_11 = tl.load(mixes_ptr + base + cb + 5) * scale_2 + b13
    cm_12 = tl.load(mixes_ptr + base + cb + 6) * scale_2 + b14
    cm_13 = tl.load(mixes_ptr + base + cb + 7) * scale_2 + b15
    cm_20 = tl.load(mixes_ptr + base + cb + 8) * scale_2 + b16
    cm_21 = tl.load(mixes_ptr + base + cb + 9) * scale_2 + b17
    cm_22 = tl.load(mixes_ptr + base + cb + 10) * scale_2 + b18
    cm_23 = tl.load(mixes_ptr + base + cb + 11) * scale_2 + b19
    cm_30 = tl.load(mixes_ptr + base + cb + 12) * scale_2 + b20
    cm_31 = tl.load(mixes_ptr + base + cb + 13) * scale_2 + b21
    cm_32 = tl.load(mixes_ptr + base + cb + 14) * scale_2 + b22
    cm_33 = tl.load(mixes_ptr + base + cb + 15) * scale_2 + b23

    rm = tl.maximum(tl.maximum(cm_00, cm_01), tl.maximum(cm_02, cm_03))
    cm_00 = tl.exp(cm_00 - rm)
    cm_01 = tl.exp(cm_01 - rm)
    cm_02 = tl.exp(cm_02 - rm)
    cm_03 = tl.exp(cm_03 - rm)
    inv_rs = 1.0 / (cm_00 + cm_01 + cm_02 + cm_03)
    cm_00 = cm_00 * inv_rs + EPS
    cm_01 = cm_01 * inv_rs + EPS
    cm_02 = cm_02 * inv_rs + EPS
    cm_03 = cm_03 * inv_rs + EPS

    rm = tl.maximum(tl.maximum(cm_10, cm_11), tl.maximum(cm_12, cm_13))
    cm_10 = tl.exp(cm_10 - rm)
    cm_11 = tl.exp(cm_11 - rm)
    cm_12 = tl.exp(cm_12 - rm)
    cm_13 = tl.exp(cm_13 - rm)
    inv_rs = 1.0 / (cm_10 + cm_11 + cm_12 + cm_13)
    cm_10 = cm_10 * inv_rs + EPS
    cm_11 = cm_11 * inv_rs + EPS
    cm_12 = cm_12 * inv_rs + EPS
    cm_13 = cm_13 * inv_rs + EPS

    rm = tl.maximum(tl.maximum(cm_20, cm_21), tl.maximum(cm_22, cm_23))
    cm_20 = tl.exp(cm_20 - rm)
    cm_21 = tl.exp(cm_21 - rm)
    cm_22 = tl.exp(cm_22 - rm)
    cm_23 = tl.exp(cm_23 - rm)
    inv_rs = 1.0 / (cm_20 + cm_21 + cm_22 + cm_23)
    cm_20 = cm_20 * inv_rs + EPS
    cm_21 = cm_21 * inv_rs + EPS
    cm_22 = cm_22 * inv_rs + EPS
    cm_23 = cm_23 * inv_rs + EPS

    rm = tl.maximum(tl.maximum(cm_30, cm_31), tl.maximum(cm_32, cm_33))
    cm_30 = tl.exp(cm_30 - rm)
    cm_31 = tl.exp(cm_31 - rm)
    cm_32 = tl.exp(cm_32 - rm)
    cm_33 = tl.exp(cm_33 - rm)
    inv_rs = 1.0 / (cm_30 + cm_31 + cm_32 + cm_33)
    cm_30 = cm_30 * inv_rs + EPS
    cm_31 = cm_31 * inv_rs + EPS
    cm_32 = cm_32 * inv_rs + EPS
    cm_33 = cm_33 * inv_rs + EPS

    inv_cs0 = 1.0 / (cm_00 + cm_10 + cm_20 + cm_30 + EPS)
    inv_cs1 = 1.0 / (cm_01 + cm_11 + cm_21 + cm_31 + EPS)
    inv_cs2 = 1.0 / (cm_02 + cm_12 + cm_22 + cm_32 + EPS)
    inv_cs3 = 1.0 / (cm_03 + cm_13 + cm_23 + cm_33 + EPS)
    cm_00 *= inv_cs0
    cm_10 *= inv_cs0
    cm_20 *= inv_cs0
    cm_30 *= inv_cs0
    cm_01 *= inv_cs1
    cm_11 *= inv_cs1
    cm_21 *= inv_cs1
    cm_31 *= inv_cs1
    cm_02 *= inv_cs2
    cm_12 *= inv_cs2
    cm_22 *= inv_cs2
    cm_32 *= inv_cs2
    cm_03 *= inv_cs3
    cm_13 *= inv_cs3
    cm_23 *= inv_cs3
    cm_33 *= inv_cs3

    for _ in range(SINKHORN_ITERS - 1):
        inv_rs0 = 1.0 / (cm_00 + cm_01 + cm_02 + cm_03 + EPS)
        inv_rs1 = 1.0 / (cm_10 + cm_11 + cm_12 + cm_13 + EPS)
        inv_rs2 = 1.0 / (cm_20 + cm_21 + cm_22 + cm_23 + EPS)
        inv_rs3 = 1.0 / (cm_30 + cm_31 + cm_32 + cm_33 + EPS)
        cm_00 *= inv_rs0
        cm_01 *= inv_rs0
        cm_02 *= inv_rs0
        cm_03 *= inv_rs0
        cm_10 *= inv_rs1
        cm_11 *= inv_rs1
        cm_12 *= inv_rs1
        cm_13 *= inv_rs1
        cm_20 *= inv_rs2
        cm_21 *= inv_rs2
        cm_22 *= inv_rs2
        cm_23 *= inv_rs2
        cm_30 *= inv_rs3
        cm_31 *= inv_rs3
        cm_32 *= inv_rs3
        cm_33 *= inv_rs3

        inv_cs0 = 1.0 / (cm_00 + cm_10 + cm_20 + cm_30 + EPS)
        inv_cs1 = 1.0 / (cm_01 + cm_11 + cm_21 + cm_31 + EPS)
        inv_cs2 = 1.0 / (cm_02 + cm_12 + cm_22 + cm_32 + EPS)
        inv_cs3 = 1.0 / (cm_03 + cm_13 + cm_23 + cm_33 + EPS)
        cm_00 *= inv_cs0
        cm_01 *= inv_cs1
        cm_02 *= inv_cs2
        cm_03 *= inv_cs3
        cm_10 *= inv_cs0
        cm_11 *= inv_cs1
        cm_12 *= inv_cs2
        cm_13 *= inv_cs3
        cm_20 *= inv_cs0
        cm_21 *= inv_cs1
        cm_22 *= inv_cs2
        cm_23 *= inv_cs3
        cm_30 *= inv_cs0
        cm_31 *= inv_cs1
        cm_32 *= inv_cs2
        cm_33 *= inv_cs3

    co = offs * 16
    tl.store(comb_ptr + co + 0, cm_00)
    tl.store(comb_ptr + co + 1, cm_01)
    tl.store(comb_ptr + co + 2, cm_02)
    tl.store(comb_ptr + co + 3, cm_03)
    tl.store(comb_ptr + co + 4, cm_10)
    tl.store(comb_ptr + co + 5, cm_11)
    tl.store(comb_ptr + co + 6, cm_12)
    tl.store(comb_ptr + co + 7, cm_13)
    tl.store(comb_ptr + co + 8, cm_20)
    tl.store(comb_ptr + co + 9, cm_21)
    tl.store(comb_ptr + co + 10, cm_22)
    tl.store(comb_ptr + co + 11, cm_23)
    tl.store(comb_ptr + co + 12, cm_30)
    tl.store(comb_ptr + co + 13, cm_31)
    tl.store(comb_ptr + co + 14, cm_32)
    tl.store(comb_ptr + co + 15, cm_33)


@triton.jit(do_not_specialize=["num_tokens"])
def _hc_split_sinkhorn_kernel_hc2(
    mixes_ptr,  # (N, 8) f32, N = 真实行数（无需 BLOCK 对齐）
    hc_scale_ptr,  # (3,) f32
    hc_base_ptr,  # (8,) f32
    pre_ptr,  # (N_pad, 2) f32, N_pad % BLOCK_N == 0
    post_ptr,  # (N_pad, 2) f32
    comb_ptr,  # (N_pad, 4) f32
    num_tokens,  # 真实行数；[num_tokens, N_pad) 为冗余行（输入读被 clamp）
    BLOCK_N: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    """Vectorized split + 2x2 Sinkhorn, exact tiles, no masks."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.minimum(offs, num_tokens - 1)
    base = offs_r * 8

    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)

    m0 = tl.load(mixes_ptr + base + 0)
    m1 = tl.load(mixes_ptr + base + 1)
    m2 = tl.load(mixes_ptr + base + 2)
    m3 = tl.load(mixes_ptr + base + 3)

    b0 = tl.load(hc_base_ptr + 0)
    b1 = tl.load(hc_base_ptr + 1)
    b2 = tl.load(hc_base_ptr + 2)
    b3 = tl.load(hc_base_ptr + 3)
    b4 = tl.load(hc_base_ptr + 4)
    b5 = tl.load(hc_base_ptr + 5)
    b6 = tl.load(hc_base_ptr + 6)
    b7 = tl.load(hc_base_ptr + 7)

    tl.store(pre_ptr + offs * 2 + 0, tl.sigmoid(m0 * scale_0 + b0) + EPS)
    tl.store(pre_ptr + offs * 2 + 1, tl.sigmoid(m1 * scale_0 + b1) + EPS)
    tl.store(post_ptr + offs * 2 + 0, 2.0 * tl.sigmoid(m2 * scale_1 + b2))
    tl.store(post_ptr + offs * 2 + 1, 2.0 * tl.sigmoid(m3 * scale_1 + b3))

    cm_00 = tl.load(mixes_ptr + base + 4) * scale_2 + b4
    cm_01 = tl.load(mixes_ptr + base + 5) * scale_2 + b5
    cm_10 = tl.load(mixes_ptr + base + 6) * scale_2 + b6
    cm_11 = tl.load(mixes_ptr + base + 7) * scale_2 + b7

    rm = tl.maximum(cm_00, cm_01)
    cm_00 = tl.exp(cm_00 - rm)
    cm_01 = tl.exp(cm_01 - rm)
    inv_rs = 1.0 / (cm_00 + cm_01)
    cm_00 = cm_00 * inv_rs + EPS
    cm_01 = cm_01 * inv_rs + EPS

    rm = tl.maximum(cm_10, cm_11)
    cm_10 = tl.exp(cm_10 - rm)
    cm_11 = tl.exp(cm_11 - rm)
    inv_rs = 1.0 / (cm_10 + cm_11)
    cm_10 = cm_10 * inv_rs + EPS
    cm_11 = cm_11 * inv_rs + EPS

    inv_cs0 = 1.0 / (cm_00 + cm_10 + EPS)
    inv_cs1 = 1.0 / (cm_01 + cm_11 + EPS)
    cm_00 *= inv_cs0
    cm_10 *= inv_cs0
    cm_01 *= inv_cs1
    cm_11 *= inv_cs1

    for _ in range(SINKHORN_ITERS - 1):
        inv_rs0 = 1.0 / (cm_00 + cm_01 + EPS)
        inv_rs1 = 1.0 / (cm_10 + cm_11 + EPS)
        cm_00 *= inv_rs0
        cm_01 *= inv_rs0
        cm_10 *= inv_rs1
        cm_11 *= inv_rs1
        inv_cs0 = 1.0 / (cm_00 + cm_10 + EPS)
        inv_cs1 = 1.0 / (cm_01 + cm_11 + EPS)
        cm_00 *= inv_cs0
        cm_10 *= inv_cs0
        cm_01 *= inv_cs1
        cm_11 *= inv_cs1

    co = offs * 4
    tl.store(comb_ptr + co + 0, cm_00)
    tl.store(comb_ptr + co + 1, cm_01)
    tl.store(comb_ptr + co + 2, cm_10)
    tl.store(comb_ptr + co + 3, cm_11)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split + Sinkhorn (kunlunxin / XPU specialized).

    Same interface and semantics as
    ``flag_gems.fused.mhc.hc_split_sinkhorn.hc_split_sinkhorn``; uses the
    XPU exact-tile kernels above for hc_mult in {2, 4} on cuda-type devices,
    any other shape / device goes to the general implementation.
    """
    if mixes.device.type != "cuda" or hc_mult not in _SUPPORTED_HC:
        return _general_hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
        )

    outer_shape = mixes.shape[:-1]
    mix_hc = (2 + hc_mult) * hc_mult
    mixes_flat = mixes.reshape(-1, mix_hc).contiguous()
    num_tokens = mixes_flat.shape[0]
    device = mixes.device

    # 输出缓冲按 _BLOCK_N 对齐的完整行数分配：exact-tile 内核按整个 grid
    # 无 mask 写满，[num_tokens, n_pad) 的冗余行在返回前被截断丢弃。
    # 输入 mixes_flat 不做任何 pad 拷贝（F.pad 在 XPU 上 fallback 到 ATen
    # constant_pad_nd，且旧实现输出缓冲只按 num_tokens 分配会越界写）。
    pad = (-num_tokens) % _BLOCK_N
    n_pad = num_tokens + pad

    pre = torch.empty((n_pad, hc_mult), dtype=torch.float32, device=device)
    post = torch.empty((n_pad, hc_mult), dtype=torch.float32, device=device)
    comb = torch.empty((n_pad, hc_mult * hc_mult), dtype=torch.float32, device=device)

    if num_tokens == 0:
        return (
            pre.view(*outer_shape, hc_mult),
            post.view(*outer_shape, hc_mult),
            comb.view(*outer_shape, hc_mult, hc_mult),
        )

    grid = n_pad // _BLOCK_N

    common = dict(
        mixes_ptr=mixes_flat,
        hc_scale_ptr=hc_scale,
        hc_base_ptr=hc_base,
        pre_ptr=pre,
        post_ptr=post,
        comb_ptr=comb,
        num_tokens=num_tokens,
        BLOCK_N=_BLOCK_N,
        SINKHORN_ITERS=sinkhorn_iters,
        EPS=eps,
        num_warps=4,
        num_stages=1,
    )
    if hc_mult == 4:
        _hc_split_sinkhorn_kernel_hc4[(grid,)](**common)
    else:
        _hc_split_sinkhorn_kernel_hc2[(grid,)](**common)

    pre = pre[:num_tokens]
    post = post[:num_tokens]
    comb = comb[:num_tokens]

    return (
        pre.view(*outer_shape, hc_mult),
        post.view(*outer_shape, hc_mult),
        comb.view(*outer_shape, hc_mult, hc_mult),
    )


def _install():
    """Wire the XPU implementation into the direct-import entrypoint.

    The mhc fused family is called via direct module import
    (``from flag_gems.fused.mhc.hc_split_sinkhorn import hc_split_sinkhorn``)
    in both tests/test_mhc_ops.py and benchmark/test_mhc.py, so the normal
    SpecOpRegistrar namespace swap cannot reach it. Replace the attribute on
    the already-imported module (loaded during ``import flag_gems``).
    """
    mod = sys.modules.get("flag_gems.fused.mhc.hc_split_sinkhorn")
    if mod is not None:
        cur = getattr(mod, "hc_split_sinkhorn", None)
        if cur is _general_hc_split_sinkhorn:
            mod.hc_split_sinkhorn = hc_split_sinkhorn


_install()
