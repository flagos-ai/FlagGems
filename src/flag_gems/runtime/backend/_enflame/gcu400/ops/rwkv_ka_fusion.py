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

logger = logging.getLogger(__name__)


@triton.jit
def rwkv_ka_fusion_kernel(
    k_ptr,
    kk_ptr,
    a_ptr,
    ka_ptr,
    o_k_ptr,
    o_kk_ptr,
    o_kka_ptr,
    T,
    C,
    H,
    N,
    BLOCK_T: tl.constexpr,
    H_size: tl.constexpr,
    N_size: tl.constexpr,
    EVEN_T: tl.constexpr,
    EVEN_HN: tl.constexpr,
    SQUARE_F32: tl.constexpr,
    OK_FIRST: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    t_offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    h_offs = tl.arange(0, H_size)
    n_offs = tl.arange(0, N_size)
    offs = (
        (t_offs * C)[:, None, None]
        + (h_offs * N)[None, :, None]
        + n_offs[None, None, :]
    )
    c_offs = (h_offs * N)[:, None] + n_offs[None, :]

    if EVEN_T and EVEN_HN:
        k = tl.load(k_ptr + offs)
        a = tl.load(a_ptr + offs)
        kk = tl.load(kk_ptr + c_offs)
        ka = tl.load(ka_ptr + c_offs)
        mask = None
    else:
        m3 = (
            (t_offs < T)[:, None, None]
            & (h_offs < H)[None, :, None]
            & (n_offs < N)[None, None, :]
        )
        m2 = (h_offs < H)[:, None] & (n_offs < N)[None, :]
        k = tl.load(k_ptr + offs, mask=m3, other=0.0)
        a = tl.load(a_ptr + offs, mask=m3, other=0.0)
        kk = tl.load(kk_ptr + c_offs, mask=m2, other=0.0)
        ka = tl.load(ka_ptr + c_offs, mask=m2, other=0.0)
        mask = m3

    # One fp32 copy of `a` feeds both o_k and o_kka (single convert).
    a32 = a.to(tl.float32)

    if SQUARE_F32:
        # fp16: exact-product chain kt32 = f32(k) * f32(kk) matches the fp32
        # reference step bit-for-bit (margin 0.69 vs 0.80 for the fp16-rounded
        # product) and removes the fp16-overflow domain quirk of k*kk.
        kt32 = k.to(tl.float32) * kk.to(tl.float32)[None, :, :]
    else:
        kt = k * kk[None, :, :]

    # o_k is independent of the norm chain; fp32 stores it first so k/a
    # liveness ends before the reduction. The kt product above is issued
    # before the store so the reduction chain does not serialize behind
    # the store issue.
    if OK_FIRST:
        ok = k * (1.0 + (a32 - 1.0) * ka[None, :, :])
        tl.store(o_k_ptr + offs, ok, mask=mask)

    if SQUARE_F32:
        s = tl.sum(kt32 * kt32, axis=2)
    elif kt.dtype == tl.float32:
        s = tl.sum(kt * kt, axis=2)
        kt32 = kt
    else:
        s = tl.sum((kt * kt).to(tl.float32), axis=2)
        kt32 = kt.to(tl.float32)

    # Div-by-norm broadcast costs BLOCK_T*H*N divides per program; a
    # reciprocal on the small [BT, H] tile cuts it to BLOCK_T*H divides.
    inv = 1.0 / tl.sqrt(s + 1e-12)
    okk = kt32 * inv[:, :, None]
    tl.store(o_kk_ptr + offs, okk, mask=mask)
    okka = okk * a32
    tl.store(o_kka_ptr + offs, okka, mask=mask)

    if not OK_FIRST:
        ok = k * (1.0 + (a32 - 1.0) * ka[None, :, :])
        tl.store(o_k_ptr + offs, ok, mask=mask)


def rwkv_ka_fusion(
    k: torch.Tensor, kk: torch.Tensor, a: torch.Tensor, ka: torch.Tensor, H: int, N: int
):
    logger.debug("GEMS_ENFLAMERWKV_KA_FUSION")
    if k.dim() == 1:
        T = 1
        C = k.shape[0]
    else:
        T, C = k.shape

    o_k = torch.empty_like(k)
    o_kk = torch.empty_like(k)
    o_kka = torch.empty_like(k)

    dtype = k.dtype
    if dtype == torch.float16:
        # fp32 exact-product square chain with reciprocal norm: margin 0.69
        # of tolerance; BLOCK_T=128 would need >917504B DSM on every probed
        # chain variant, so it stays at the DSM-fitting 64.
        # num_warps=1 plus the early o_k store measured 300us against 355us for
        # the num_warps=4 late-store order at T=4096, so this wins on 16-bit.
        bt_cap, square_f32, ok_first, num_warps = 64, True, True, 1
    elif dtype == torch.bfloat16:
        # elem square + fp32 reciprocal chain: margins <= 0.55 of tolerance.
        # BLOCK_T=128 needs 1187840B DSM > 917504B limit on every probed
        # chain variant, so it stays at 64.
        bt_cap, square_f32, ok_first, num_warps = 64, False, True, 1
    elif dtype == torch.float32:
        # OK_FIRST store order trims liveness enough that BT=64 compiles,
        # but measured 327us vs 301us at BT=32 (T=32768); the late-store
        # order at BT=32 was 406us, so 32 + OK_FIRST is the sweet spot.
        # fp32 keeps num_warps=4: the 32-bit tiles are the DSM-critical ones
        # and the single-warp variant measured 0.0908ms vs 0.0905ms.
        bt_cap, square_f32, ok_first, num_warps = 32, False, True, 4
    else:
        raise NotImplementedError(f"rwkv_ka_fusion: unsupported dtype {dtype}")

    BLOCK_T = min(triton.next_power_of_2(max(1, T // 16)), bt_cap)
    H_size = triton.next_power_of_2(max(1, H))
    N_size = triton.next_power_of_2(max(1, N))
    EVEN_T = (T % BLOCK_T == 0) and (T > 0)
    EVEN_HN = (H == H_size) and (N == N_size)

    grid = (triton.cdiv(T, BLOCK_T),)
    rwkv_ka_fusion_kernel[grid](
        k,
        kk,
        a,
        ka,
        o_k,
        o_kk,
        o_kka,
        T,
        C,
        H,
        N,
        BLOCK_T,
        H_size,
        N_size,
        EVEN_T,
        EVEN_HN,
        square_f32,
        ok_first,
        num_warps=num_warps,
    )

    return o_k, o_kk, o_kka
