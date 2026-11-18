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

"""Ascend MX-FP8 quantized batch matrix multiplication (QuantBatchMatmulV3).

Replaces ``torch_npu.npu_quant_matmul`` in the MXFP8 call form used by
vllm-ascend ``AscendW8A8MXFP8LinearMethod.apply``::

    torch_npu.npu_quant_matmul(
        quantized_x, layer.weight, layer.weight_scale,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale=pertoken_scale,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        bias=bias, output_dtype=output_dtype,
        group_sizes=[1, 1, 32])

Numerical definition::

    out[m, n] = sum_k  x[m,k] * 2^(sa[m,k//32]-127) * w[k,n] * 2^(sb[k//32,n]-127)

where ``sa`` / ``sb`` are e8m0 scales (uint8), one per 32 K-elements (MX spec).

Key implementation notes inherited from GroupedMatmul and refined here:

1. ``tl.dot_scaled`` lowers to a single hardware MX matmul instruction.
2. Mask cost depends on whether the boundary is a compile-time constant;
   static masks are free, dynamic masks cause 13x slowdown on scale loads.
3. No runtime arithmetic on dynamic index values (clamp is as bad as div/mod).
4. Weight scale physical layout is interleaved ``[K/64, N, 2]``.
5. NZ physical order is ``[N/32, K/16, 16, 32]``.

Supports both ND and FRACTAL_NZ weight layouts (auto-detected).
"""

from __future__ import annotations

import logging

import torch
import torch_npu  # noqa: F401  register NPU backend
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

__all__ = ["quant_batch_matmul_mxfp8"]

MX_BLOCK = 32  # e8m0 scale covers 32 K-elements
ACL_FORMAT_FRACTAL_NZ = 29

# UB budget: 950 raw UB is 256 KB, backend graph budget is 80%.
_UB_BUDGET_BYTES = 256 * 1024 * 80 // 100  # 204800


def _cube_core_num(device=None) -> int:
    """Return the cube core count (950DT: 32)."""
    try:
        idx = torch.npu.current_device() if device is None else device
        return int(torch_npu.npu.get_device_properties(idx).cube_core_num)
    except Exception:
        return 32


# ---------------------------------------------------------------------- kernel


@triton.jit
def _e8m0_to_f32(v):
    """e8m0 biased exponent -> fp32: bit-shift into fp32 exponent field."""
    return (v.to(tl.uint32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _qbmm_kernel(
    A,       # fp8_e4m3fn [M_pad, K]
    SA,      # uint8      [M_pad, K/32]
    B,       # fp8_e4m3fn [K, N]  (ND or NZ)
    SB,      # uint8      physical [K/32, N] interleaved
    C,       # output or partials
    BIAS,    # fp32 [N] or dummy
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    PSTRIDE: tl.constexpr,
    MX: tl.constexpr,
    WEIGHT_NZ: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    A_BCAST: tl.constexpr,
    SB_NK: tl.constexpr,
    MASK_M: tl.constexpr,
    M_CE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_tiles: tl.constexpr = N // BLOCK_N
    n_kblk: tl.constexpr = K // MX
    chunk: tl.constexpr = n_kblk // SPLIT_K

    num_m_tiles: tl.constexpr = NUM_M_TILES
    tile_n = pid % num_n_tiles
    M_SINGLE: tl.constexpr = num_m_tiles == 1
    if M_SINGLE:
        tile_m = 0
        sk = pid // num_n_tiles
    else:
        tile_m = (pid // num_n_tiles) % num_m_tiles
        sk = pid // (num_n_tiles * num_m_tiles)

    offs_m0 = tl.arange(0, BLOCK_M).to(tl.int64)
    offs_m = tile_m * BLOCK_M + offs_m0
    offs_n0 = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_n = tile_n.to(tl.int64) * BLOCK_N + offs_n0
    offs_kk = tl.arange(0, BLOCK_K).to(tl.int64)
    offs_sk = tl.arange(0, BLOCK_K // MX).to(tl.int64)

    ROW_MASK: tl.constexpr = M_CE > 0 and not A_BCAST
    row0 = offs_m0 * 0 if A_BCAST else offs_m0
    if A_BCAST or M_SINGLE:
        m_off_a = 0
        m_off_sa = 0
    else:
        m_off_a = tile_m.to(tl.int64) * (BLOCK_M * K)
        m_off_sa = tile_m.to(tl.int64) * (BLOCK_M * n_kblk)
    a_base = A + m_off_a + row0[:, None] * K
    sa_base = SA + m_off_sa + row0[:, None] * n_kblk

    if WEIGHT_NZ:
        nz_g: tl.constexpr = BLOCK_N // 32
        nvec = (offs_n0 // 32) * (K * 32) + (offs_n0 % 32)
        b_base = B + tile_n.to(tl.int64) * (nz_g * K * 32) + nvec[None, :]
    else:
        b_base = B + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    kb_start = sk.to(tl.int64) * chunk
    for k0 in tl.range(0, chunk, BLOCK_K // MX):
        kb0 = kb_start + k0.to(tl.int64)
        koff = kb0 * MX

        if ROW_MASK:
            rm = offs_m0 < M_CE
            a = tl.load(a_base + koff + offs_kk[None, :], mask=rm[:, None],
                        other=tl.zeros((1, 1), dtype=A.dtype.element_ty))
            sa = tl.load(sa_base + kb0 + offs_sk[None, :], mask=rm[:, None],
                         other=tl.zeros((1, 1), dtype=SA.dtype.element_ty))
        else:
            a = tl.load(a_base + koff + offs_kk[None, :])
            sa = tl.load(sa_base + kb0 + offs_sk[None, :])
        if WEIGHT_NZ:
            b = tl.load(b_base + (koff + offs_kk)[:, None] * 32)
        else:
            b = tl.load(b_base + (koff + offs_kk)[:, None] * N)
        if SB_NK:
            sb = tl.load(SB + offs_n[:, None] * n_kblk + kb0 + offs_sk[None, :])
        else:
            sb_off = ((kb0 // 2) * (N * 2)
                      + offs_n[:, None] * 2
                      + (offs_sk[None, :] // 2) * (N * 2)
                      + (offs_sk[None, :] % 2))
            sb = tl.load(SB + sb_off)
        acc = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)

    if SPLIT_K == 1:
        if HAS_BIAS:
            acc += tl.load(BIAS + offs_n)[None, :]
        if M_CE > 0:
            tl.store(C + offs_m[:, None] * N + offs_n[None, :],
                     acc.to(C.dtype.element_ty), mask=(offs_m < M_CE)[:, None])
        elif MASK_M:
            tl.store(C + offs_m[:, None] * N + offs_n[None, :],
                     acc.to(C.dtype.element_ty), mask=(offs_m < M)[:, None])
        else:
            tl.store(C + offs_m[:, None] * N + offs_n[None, :],
                     acc.to(C.dtype.element_ty))
    else:
        pbase = C + sk.to(tl.int64) * PSTRIDE + offs_m[:, None] * N + offs_n[None, :]
        if M_CE > 0:
            tl.store(pbase, acc, mask=(offs_m < M_CE)[:, None])
        elif MASK_M:
            tl.store(pbase, acc, mask=(offs_m < M)[:, None])
        else:
            tl.store(pbase, acc)


@triton.jit
def _combine_kernel(
    P,       # fp32 [SPLIT_K, M, N]
    C,       # bf16/fp16 [M, N]
    BIAS,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PSTRIDE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Reduce split-K partials with a single 2D tile load."""
    m = tl.program_id(0)
    tile_n = tl.program_id(1)
    offs_n = (tile_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_sk = tl.arange(0, SPLIT_K).to(tl.int64)
    base = m.to(tl.int64) * N + offs_n
    part = tl.load(P + offs_sk[:, None] * PSTRIDE + base[None, :])
    acc = tl.sum(part, axis=0)
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n)
    tl.store(C + base, acc.to(C.dtype.element_ty))


# --------------------------------------------------------- validation / tiling


def _is_nz(t: torch.Tensor) -> bool:
    """Check whether a weight tensor uses FRACTAL_NZ physical layout."""
    try:
        return torch_npu.get_npu_format(t) == ACL_FORMAT_FRACTAL_NZ
    except Exception:
        return False


def _divisors_pow2(n: int, lo: int, hi: int):
    """Powers of two in [lo, hi] that divide *n*, largest first."""
    out = []
    c = hi
    while c >= lo:
        if n % c == 0:
            out.append(c)
        c //= 2
    return out


def _pick_tiling(M: int, K: int, N: int, cores: int,
                 block_n: int | None, split_k: int | None, block_k: int | None):
    """Select (BLOCK_N, SPLIT_K, BLOCK_K).

    Rules: BLOCK_N as large as possible (<=1024), parallelism via split-K.
    """
    num_m_tiles = (M + 15) // 16
    cands_n = _divisors_pow2(N, 32, 1024)
    if not cands_n:
        raise ValueError(f"N={N}: no power-of-two in [32,1024] divides it for BLOCK_N")

    n_kblk = K // MX_BLOCK
    max_sk = max(1, n_kblk // 8)
    sk_ok = num_m_tiles == 1

    def pick_sk(tiles: int) -> int:
        if not sk_ok:
            return 1
        sk = 1
        while sk * 2 <= max_sk and tiles * sk * 2 <= cores \
                and n_kblk % (sk * 2) == 0:
            sk *= 2
        return sk

    if block_n is not None:
        bn = block_n
    else:
        bn = cands_n[0]
        while num_m_tiles * (N // bn) * pick_sk(num_m_tiles * (N // bn)) \
                < max(1, cores // 4):
            nxt = [c for c in cands_n if c < bn]
            if not nxt:
                break
            bn = nxt[0]

    if split_k is not None:
        sk = split_k
        if sk > 1 and not sk_ok:
            raise ValueError(
                f"M={M} has {num_m_tiles} m-tiles, cannot use split_k={sk}: "
                f"runtime row mask + split-K partials store triggers compiler abort"
            )
    else:
        sk = pick_sk(num_m_tiles * (N // bn))

    if n_kblk % sk != 0:
        raise ValueError(f"split_k={sk} must divide K/32={n_kblk}")
    chunk = n_kblk // sk

    if block_k is not None:
        bk = block_k
    else:
        bk = 64 if chunk % 2 == 0 else None
        if bk is None:
            raise ValueError(
                f"chunk={chunk} is odd, cannot pick BLOCK_K as multiple of 64; "
                f"reduce split_k (current {sk})"
            )
    if bk % (2 * MX_BLOCK) != 0:
        raise ValueError(
            f"block_k={bk} must be a multiple of {2 * MX_BLOCK}"
        )
    if chunk % (bk // MX_BLOCK) != 0:
        raise ValueError(f"block_k={bk} must divide per-split K length {chunk * MX_BLOCK}")
    return bn, sk, bk


# --------------------------------------------------------------- public API


def quant_batch_matmul_mxfp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    pertoken_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    block_m: int = 16,
    block_n: int | None = None,
    block_k: int | None = None,
    split_k: int | None = None,
    num_cores: int | None = None,
    weight_nz: bool | None = None,
    a_bcast: bool | None = None,
    sb_nk: bool = False,
    m_const: int | None = None,
) -> torch.Tensor:
    """MX-FP8 quantized matmul replacing ``torch_npu.npu_quant_matmul`` MXFP8 form.

    Parameters
    ----------
    x : fp8_e4m3fn [M, K]
    weight : fp8_e4m3fn [K, N], ND or FRACTAL_NZ (auto-detected)
    weight_scale : uint8, (K/32)*N elements, physical [K/64, N, 2] interleaved
    pertoken_scale : uint8, M*(K/32) elements
    bias : fp32 [N], optional
    output_dtype : bf16 or fp16
    """
    logger.debug("GEMS_ASCEND QUANT_BATCH_MATMUL_MXFP8")

    if x.ndim != 2:
        raise ValueError(f"x must be 2-D [M, K], got {tuple(x.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D [K, N], got {tuple(weight.shape)}")
    M, K = x.shape
    Kw, N = weight.shape
    if K != Kw:
        raise ValueError(f"K mismatch: x {K} vs weight {Kw}")
    if K % MX_BLOCK != 0:
        raise ValueError(f"K={K} must be a multiple of {MX_BLOCK}")
    if x.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"x/weight must be float8_e4m3fn, got {x.dtype}/{weight.dtype}"
        )
    n_kblk = K // MX_BLOCK
    if weight_scale.numel() != n_kblk * N:
        raise ValueError(
            f"weight_scale elements should be (K/32)*N = {n_kblk * N}, "
            f"got {weight_scale.numel()}"
        )
    if pertoken_scale.numel() < M * n_kblk:
        raise ValueError(
            f"pertoken_scale elements should be M*(K/32) = {M * n_kblk}, "
            f"got {pertoken_scale.numel()}"
        )
    for name, t in (("x", x), ("weight_scale", weight_scale),
                    ("pertoken_scale", pertoken_scale)):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if weight_nz is None:
        weight_nz = _is_nz(weight)
    if weight_nz:
        if K % 16 != 0 or N % 32 != 0:
            raise ValueError(
                f"NZ layout requires K%16==0 and N%32==0, got K={K} N={N}"
            )
    elif not weight.is_contiguous():
        raise ValueError("weight must be contiguous for ND layout")

    if block_m != 16:
        raise ValueError("block_m is fixed at 16 (cube minimum M granularity)")
    cores = num_cores if num_cores is not None else _cube_core_num(x.device.index)
    bn, sk, bk = _pick_tiling(M, K, N, cores, block_n, split_k, block_k)
    if N % bn != 0:
        raise ValueError(f"block_n={bn} must divide N={N}")

    num_m_tiles = (M + block_m - 1) // block_m
    m_pad = num_m_tiles * block_m

    # B double-buffer decision
    backend_opts: dict = {}
    b_tile = bk * bn
    acc_size = block_m * bn * 4
    a_tile = block_m * bk
    scales = (block_m + bn) * (bk // MX_BLOCK)
    need = b_tile * 2 + acc_size + a_tile + scales
    if need <= _UB_BUDGET_BYTES:
        backend_opts["multibuffer"] = True

    if a_bcast is None:
        a_bcast = M == 1
    elif a_bcast and M != 1:
        raise ValueError(f"a_bcast only valid when M==1, got M={M}")

    m_ce = M if (m_const is None and M < block_m) else (m_const or 0)
    if m_ce and not (0 < m_ce < block_m):
        raise ValueError(f"m_const only supports 0 < M < {block_m}, got {m_ce}")
    if m_ce and m_ce != M:
        raise ValueError(f"m_const={m_ce} must equal actual M={M}")
    p_rows = M if m_ce else m_pad

    if a_bcast or m_pad == M or m_ce:
        ain, sa = x, pertoken_scale
    else:
        ain = torch.empty((m_pad, K), device=x.device, dtype=x.dtype)
        ain[:M] = x
        sa = torch.zeros(m_pad * n_kblk, device=x.device, dtype=torch.uint8)
        sa[: M * n_kblk] = pertoken_scale.reshape(-1)[: M * n_kblk]

    has_bias = bias is not None
    if has_bias:
        if bias.numel() != N:
            raise ValueError(f"bias elements should be N={N}, got {bias.numel()}")
        bias_t = bias.reshape(-1).to(torch.float32).contiguous()
    else:
        bias_t = pertoken_scale  # dummy, kernel won't read it

    grid = (num_m_tiles * (N // bn) * sk,)

    with torch_device_fn.device(x.device):
        if sk == 1:
            out = torch.empty((M, N), device=x.device, dtype=output_dtype)
            _qbmm_kernel[grid](
                ain, sa, weight, weight_scale, out, bias_t, M,
                N=N, K=K, BLOCK_M=block_m, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=1,
                NUM_M_TILES=num_m_tiles,
                PSTRIDE=p_rows * N, MX=MX_BLOCK, WEIGHT_NZ=weight_nz,
                HAS_BIAS=has_bias,
                A_BCAST=a_bcast, SB_NK=sb_nk, MASK_M=m_pad != M, M_CE=m_ce,
                **backend_opts,
            )
            return out

        part = torch.empty((sk, p_rows, N), device=x.device, dtype=torch.float32)
        _qbmm_kernel[grid](
            ain, sa, weight, weight_scale, part, bias_t, M,
            N=N, K=K, BLOCK_M=block_m, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk,
            NUM_M_TILES=num_m_tiles,
            PSTRIDE=p_rows * N, MX=MX_BLOCK, WEIGHT_NZ=weight_nz,
            HAS_BIAS=has_bias,
            A_BCAST=a_bcast, SB_NK=sb_nk, MASK_M=False, M_CE=m_ce,
            **backend_opts,
        )
        out = torch.empty((M, N), device=x.device, dtype=output_dtype)
        cbn = next(
            (c for c in _divisors_pow2(N, 1, 2048) if sk * c * 4 <= 64 * 1024),
            _divisors_pow2(N, 1, 2048)[-1],
        )
        _combine_kernel[(M, N // cbn)](
            part, out, bias_t,
            N=N, SPLIT_K=sk, PSTRIDE=p_rows * N, BLOCK_N=cbn, HAS_BIAS=has_bias,
        )
        return out
