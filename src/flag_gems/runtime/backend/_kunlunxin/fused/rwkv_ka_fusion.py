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
    M,
    H: tl.constexpr,
    N: tl.constexpr,
    TILE_R: tl.constexpr,
):
    # The tensors k/a/outputs are [T, C] contiguous with C = H*N, so viewed as
    # [M, N] (M = T*H) they stay contiguous: element (row, n) is at row*N + n.
    # Each row is one head's N-vector that needs an L2 normalization over N.
    # We process TILE_R rows per program as a [TILE_R, N] tile and reduce over
    # axis=1 (the only reduction axis XPU supports for 2D tiles), turning the
    # original grid=T / serial-H-loop / 64-wide launch-bound kernel into a wide
    # vectorized one.
    pid = tl.program_id(axis=0)
    row = pid * TILE_R + tl.arange(0, TILE_R)
    n = tl.arange(0, N)
    row_ok = row < M
    mask = row_ok[:, None]

    offs = row[:, None] * N + n[None, :]
    k = tl.load(k_ptr + offs, mask=mask, other=0.0)
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)

    # kk/ka are per-channel [C] = [H*N]; row's head is (row % H), so the channel
    # index is (row % H)*N + n.
    h = row % H
    c_idx = h[:, None] * N + n[None, :]
    kk = tl.load(kk_ptr + c_idx, mask=mask, other=0.0)
    ka = tl.load(ka_ptr + c_idx, mask=mask, other=0.0)

    kt = k * kk
    kt2 = (kt * kt).to(tl.float32)
    norm_kt2 = tl.sum(kt2, axis=1)
    norm_kt = tl.sqrt(norm_kt2 + 1e-12)
    okk = kt / norm_kt[:, None]
    tl.store(o_kk_ptr + offs, okk, mask=mask)

    ok = k * (1 + (a.to(tl.float32) - 1) * ka)
    okka = okk * a
    tl.store(o_k_ptr + offs, ok, mask=mask)
    tl.store(o_kka_ptr + offs, okka, mask=mask)


def _choose_tile_r(M, N):
    # Bounded tile (avoids the giant-tile anti-pattern); shrink for small M so
    # we still expose enough parallel programs. Kept a power of 2.
    tile = max(1, 8192 // N)  # 128 for N=64
    while tile > 1 and triton.cdiv(M, tile) < 64:
        tile //= 2
    return tile


def rwkv_ka_fusion(
    k: torch.Tensor, kk: torch.Tensor, a: torch.Tensor, ka: torch.Tensor, H: int, N: int
):
    logger.debug("GEMS_KUNLUNXIN RWKV KA FUSION")

    if k.dim() == 1:
        T = 1
        C = k.shape[0]
    else:
        T, C = k.shape

    o_k = torch.empty_like(k)
    o_kk = torch.empty_like(k)
    o_kka = torch.empty_like(k)

    M = T * H  # rows in the [M, N] view (C == H * N)
    tile_r = _choose_tile_r(M, N)
    grid = (triton.cdiv(M, tile_r),)
    # isCloseVectorization=True: the default XPU store vectorization miscompiles
    # this 2D [TILE_R, N] tile store at some sizes (fp16 [32,64], fp32 [16,64]),
    # dropping lanes n=0,32. Disabling it fixes correctness with ~0 perf cost
    # (this kernel is not store-bandwidth bound).
    rwkv_ka_fusion_kernel[grid](
        k,
        kk,
        a,
        ka,
        o_k,
        o_kk,
        o_kka,
        M,
        H,
        N,
        tile_r,
        isCloseVectorization=True,
    )

    return o_k, o_kk, o_kka
