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
    NEED_MASK: tl.constexpr,
):
    # The entire tensors k/a/outputs are [T, C] contiguous with C = H*N, so
    # viewed as [M, N] (M = T*H) they stay contiguous: element (row, n) is at
    # row*N + n.  Each row is one head's N-vector that needs an L2
    # normalization over N.  We process TILE_R rows per program as a
    # [TILE_R, N] tile and reduce over axis=1 (the only reduction axis XPU
    # supports for 2D tiles), turning the original grid=T / serial-H-loop /
    # 64-wide launch-bound kernel into a wide vectorized one.
    pid = tl.program_id(axis=0)
    row = pid * TILE_R + tl.arange(0, TILE_R)
    n = tl.arange(0, N)
    offs = row[:, None] * N + n[None, :]

    if NEED_MASK:
        mask = (row < M)[:, None]
        k = tl.load(k_ptr + offs, mask=mask, other=0.0)
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    else:
        # When M % TILE_R == 0 the row mask is all-true.  Keeping that
        # constant mask on XPU degrades the whole kernel ~50-80x (masked 2D
        # load/store path, measured 11.3ms -> 0.16ms at T=4096/N=64), so
        # specialize the divisible case to unmasked memory ops.  This is the
        # same NEED_MASK pattern used by the rms_norm family on this backend.
        k = tl.load(k_ptr + offs)
        a = tl.load(a_ptr + offs)

    # kk/ka are per-channel [C] = [H*N]; the lane's head is (row % H), so the
    # channel index is (row % H)*N + n.
    h = row % H
    c_idx = h[:, None] * N + n[None, :]
    if NEED_MASK:
        kk = tl.load(kk_ptr + c_idx, mask=mask, other=0.0)
        ka = tl.load(ka_ptr + c_idx, mask=mask, other=0.0)
    else:
        kk = tl.load(kk_ptr + c_idx)
        ka = tl.load(ka_ptr + c_idx)

    kt = k * kk
    kt2 = (kt * kt).to(tl.float32)
    norm_kt2 = tl.sum(kt2, axis=1)
    norm_kt = tl.sqrt(norm_kt2 + 1e-12)
    okk = kt / norm_kt[:, None]

    ok = k * (1 + (a.to(tl.float32) - 1) * ka)
    okka = okk * a
    if NEED_MASK:
        tl.store(o_kk_ptr + offs, okk, mask=mask)
        tl.store(o_k_ptr + offs, ok, mask=mask)
        tl.store(o_kka_ptr + offs, okka, mask=mask)
    else:
        tl.store(o_kk_ptr + offs, okk)
        tl.store(o_k_ptr + offs, ok)
        tl.store(o_kka_ptr + offs, okka)


def _choose_tile_r(M, N):
    # Fixed [128, N] tiles: 128 rows at N=64 stays in SRAM (8192-element
    # budget), and 128 rows is the XPU sweet spot.  The previous policy shrank
    # the tile until grid >= 64, producing many tiny programs that ran 4-6x
    # slower on this backend (T=64: 0.066ms at tile=8 vs 0.010ms at tile=128;
    # T=16: 0.19ms at tile=2 vs 0.010ms at tile=128).  The tile must stay a
    # power of two (tl.arange) and is capped at M for small row counts; the
    # tail (M % tile != 0) is handled by the NEED_MASK path in the kernel.
    tile = min(128, 8192 // N)
    if tile > M:
        tile = 1 << (max(1, M).bit_length() - 1) if M > 0 else 1
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
    need_mask = (M % tile_r) != 0  # tail block: rows not covered by TILE_R
    grid = (triton.cdiv(M, tile_r),)
    # isCloseVectorization is deliberately left at default: with the fixed
    # [128, N] tile every size compiles to the same unmasked 2D store shape, so
    # the sizes that used to miscompile (fp16 [32,64], fp32 [16,64]) no longer
    # occur, and the default (unvectorized-store) path is ~1.7x faster than
    # forcing it on.
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
        need_mask,
    )

    return o_k, o_kk, o_kka
