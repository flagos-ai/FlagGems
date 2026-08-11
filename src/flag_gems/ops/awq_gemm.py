# SPDX-License-Identifier: Apache-2.0
"""
AWQ quantized GEMM (awq-gemm) forward operator for FlagGems.

Implements the fused-dequant INT4 GEMM used by AWQ (Activation-aware Weight
Quantization). The packed weight layout follows the classic AWQ kernel format
(see llm-awq / AutoAWQ `awq_gemm`):

    qweight: (K // 8, N) int32
        Each int32 packs 8 consecutive K-codes as nibbles: nibble (k % 8) of
        qweight[k // 8, n] is the quantized code of w[k, n].
    qzeros:  (K // group_size, N // 8) int32   (optional, None for symmetric)
        Each int32 packs 8 consecutive N zero-point codes as nibbles: nibble
        (n % 8) of qzeros[g, n // 8] is the zero point of group g, column n.
    scales:  (K // group_size, N) fp16 / bf16

Semantics (per group g = k // group_size):

    out[m, n] = sum_k input[m, k] * (qweight_code(k, n) - qzeros_code(g, n))
                                        * scales[g, n]

i.e. the dequantized weight matrix is (K, N) and `out = input @ w`.

The Triton kernel dequantizes the INT4 weights on the fly inside the K-loop
(bit-plane extraction + register-level stacking) so no intermediate fp16
weight matrix is ever materialized, and feeds the fp16 dequantized tile
directly into `tl.dot`.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAX_BLOCK_SIZE_K = 128
_MIN_BLOCK_SIZE_K = 16


def _pick_block_size_k(group_size: int) -> int:
    """Largest power-of-two tile (<= 128, >= 16) that divides group_size.

    A K-tile must never straddle two quant groups, so BLOCK_SIZE_K has to
    divide `group_size` (and `tl.arange` requires a power of two).
    """
    for bk in (_MAX_BLOCK_SIZE_K, 64, 32, _MIN_BLOCK_SIZE_K):
        if group_size % bk == 0:
            return bk
    raise ValueError(
        f"group_size={group_size} is not divisible by any power-of-two in "
        f"[{_MIN_BLOCK_SIZE_K}, {_MAX_BLOCK_SIZE_K}]; AWQ GEMM cannot tile it."
    )


def _select_config(M: int, N: int, group_size: int, device: str) -> dict:
    """Heuristic launch configuration (no autotune, so BLOCK_SIZE_K can be
    derived from `group_size` at launch time)."""
    block_size_k = _pick_block_size_k(group_size)
    if device != "cuda":
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": block_size_k,
            "GROUP_SIZE_M": 4,
            "num_warps": 4,
            "num_stages": 3,
        }
    # Decode-shaped batches (M <= 16) keep a small M-tile for occupancy.
    if M <= 16:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": block_size_k,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 4,
        }
    block_size_n = 64 if N < 128 else 128
    block_size_m = 64 if M < 64 else 128
    num_warps = 4 if block_size_n <= 128 else 8
    return {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": 8,
        "num_warps": num_warps,
        "num_stages": 4,
    }


@triton.jit
def _unpack_k8(b, X: tl.constexpr, Y: tl.constexpr):
    """Split the 8 nibbles of an (X, Y) int32 tile and reassemble them so that
    output row r = 8 * packed_row + i holds nibble i of packed_row
    (matches the AWQ K-packing: w[k, n] = nibble (k % 8) of qweight[k // 8, n]).

    The planes are stacked along a middle axis to form an (X, 8, Y) tensor
    [plane0; plane1; ...; plane7], then reshaped to (8X, Y).
    """
    b0 = b & 0xF
    b1 = (b >> 4) & 0xF
    b2 = (b >> 8) & 0xF
    b3 = (b >> 12) & 0xF
    b4 = (b >> 16) & 0xF
    b5 = (b >> 20) & 0xF
    b6 = (b >> 24) & 0xF
    b7 = (b >> 28) & 0xF
    # (X, 2, Y) with nibble dim in the middle
    p01 = tl.permute(tl.join(b0, b1), (0, 2, 1))
    p23 = tl.permute(tl.join(b2, b3), (0, 2, 1))
    p45 = tl.permute(tl.join(b4, b5), (0, 2, 1))
    p67 = tl.permute(tl.join(b6, b7), (0, 2, 1))
    # (X, 4, Y)
    p0123 = tl.reshape(tl.permute(tl.join(p01, p23), (0, 3, 1, 2)), (X, 4, Y))
    p4567 = tl.reshape(tl.permute(tl.join(p45, p67), (0, 3, 1, 2)), (X, 4, Y))
    # (X, 8, Y) then (8X, Y)
    p01234567 = tl.reshape(tl.permute(tl.join(p0123, p4567), (0, 3, 1, 2)), (X, 8, Y))
    return tl.reshape(p01234567, (8 * X, Y))


@triton.jit
def awq_gemm_kernel(
    A,
    Qweight,
    Qzeros,
    Scales,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_qk,
    stride_qn,
    stride_zg,
    stride_zn,
    stride_sg,
    stride_sn,
    stride_cm,
    stride_cn,
    group_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_ZEROS: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    BLOCK_SIZE_K_PACK: tl.constexpr = BLOCK_SIZE_K // 8

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Modulo offsets keep out-of-range loads in-bounds; stores are masked.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_kp = tl.arange(0, BLOCK_SIZE_K_PACK)

    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = Qweight + offs_kp[:, None] * stride_qk + offs_bn[None, :] * stride_qn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k0 * BLOCK_SIZE_K
        a = tl.load(a_ptrs).to(COMPUTE_DTYPE)
        b_packed = tl.load(b_ptrs)  # (BLOCK_SIZE_K // 8, BLOCK_SIZE_N) int32
        b_codes = _unpack_k8(b_packed, BLOCK_SIZE_K_PACK, BLOCK_SIZE_N)
        b = b_codes.to(COMPUTE_DTYPE)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)

        g = k_start // group_size
        if HAS_ZEROS:
            # Per-column zero codes: nibble (offs_bn % 8) of the packed word
            # at column offs_bn // 8. `offs_bn` is wrapped so loads stay
            # in-bounds even when BLOCK_SIZE_N > N.
            z_packed = tl.load(Qzeros + g * stride_zg + (offs_bn // 8) * stride_zn)
            z_codes = (z_packed >> (4 * (offs_bn % 8))) & 0xF  # (BLOCK_SIZE_N,)
            z = tl.broadcast_to(
                tl.reshape(z_codes.to(COMPUTE_DTYPE), (1, BLOCK_SIZE_N)),
                (BLOCK_SIZE_K, BLOCK_SIZE_N),
            )
        else:
            z = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=COMPUTE_DTYPE)

        scale = tl.load(Scales + g * stride_sg + offs_bn * stride_sn)  # (N,)
        s = tl.broadcast_to(
            tl.reshape(scale, (1, BLOCK_SIZE_N)), (BLOCK_SIZE_K, BLOCK_SIZE_N)
        )

        w = (b - z) * s  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        accumulator = tl.dot(a, w, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K_PACK * stride_qk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def awq_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor | None,
    scales: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """AWQ quantized GEMM forward: `out = input @ dequant(qweight, qzeros, scales)`.

    Args:
        input: (M, K) activations in fp16 / bf16 / fp32 (contiguous).
        qweight: (K // 8, N) int32 packed INT4 weight codes (nibble k % 8).
        qzeros: (K // group_size, N // 8) int32 packed zero points (nibble n % 8),
            or None for symmetric quantization (no zero point).
        scales: (K // group_size, N) per-group scale in fp16 / bf16.
        group_size: quantization group size (must divide K, and be divisible
            by a power-of-two tile in [16, 128]).
        out_dtype: output dtype; defaults to `input.dtype`.

    Returns:
        (M, N) tensor with `out[m, n] = sum_k input[m, k] * w[k, n]` where
        `w[k, n] = (code(k, n) - zero(g, n)) * scales[g, n]`, g = k // group_size.
    """
    assert input.ndim == 2, "input must be 2D (M, K)"
    assert input.is_contiguous(), "input must be contiguous"
    M, K = input.shape
    assert K % group_size == 0, "group_size must divide K"
    assert qweight.dtype == torch.int32, "qweight must be int32"
    N = qweight.shape[1]
    assert qweight.shape == (
        K // 8,
        N,
    ), f"qweight must be (K//8, N) = {(K // 8, N)}, got {qweight.shape}"
    G = K // group_size
    assert scales.shape == (
        G,
        N,
    ), f"scales must be (K//group_size, N) = {(G, N)}, got {scales.shape}"
    has_zeros = qzeros is not None
    if has_zeros:
        assert qzeros.dtype == torch.int32, "qzeros must be int32"
        assert N % 8 == 0, "qzeros packing requires N % 8 == 0"
        assert qzeros.shape == (G, N // 8), (
            f"qzeros must be (K//group_size, N//8) = {(G, N // 8)}, "
            f"got {qzeros.shape}"
        )

    compute_dtype = (
        input.dtype if input.dtype in (torch.float16, torch.bfloat16) else torch.float16
    )
    # Cast scales into the kernel compute dtype (no-op when already matching).
    scales = scales.to(compute_dtype)

    if out_dtype is None:
        out_dtype = input.dtype
    C = torch.empty((M, N), device=input.device, dtype=out_dtype)

    config = _select_config(M, N, group_size, str(input.device.type))
    if input.dtype == torch.float32:
        # fp32 A tiles occupy 2x shared memory; shrink the pipeline stages.
        config["num_stages"] = min(config["num_stages"], 2)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    awq_gemm_kernel[grid](
        input,
        qweight,
        qzeros,
        scales,
        C,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        qzeros.stride(0) if has_zeros else 0,
        qzeros.stride(1) if has_zeros else 0,
        scales.stride(0),
        scales.stride(1),
        C.stride(0),
        C.stride(1),
        group_size,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        HAS_ZEROS=has_zeros,
        COMPUTE_DTYPE=tl.float16 if compute_dtype == torch.float16 else tl.bfloat16,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return C


def pack_awq_weight(
    w: torch.Tensor,
    group_size: int,
    dtype: torch.dtype = torch.float16,
) -> tuple:
    """Quantize a full-precision weight into the AWQ INT4 packed format.

    Args:
        w: (K, N) weight (in_features, out_features) in fp16 / bf16 / fp32.
        group_size: quantization group size (must divide K).
        dtype: dtype of the returned `scales` (typically fp16 / bf16).

    Returns:
        (qweight, scales, qzeros) matching the layout consumed by `awq_gemm`.
        Asymmetric per-group quantization:
            scale[g] = (max - min) / 15, zero[g] = round(-min / scale[g]),
            code = clamp(round(w / scale + zero), 0, 15)
            w ~= (code - zero) * scale
    """
    assert w.ndim == 2, "w must be 2D (K, N)"
    K, N = w.shape
    assert K % group_size == 0, "group_size must divide K"
    assert N % 8 == 0, "N must be a multiple of 8 for zero-point packing"

    wf = w.to(torch.float32)
    G = K // group_size
    wg = wf.reshape(G, group_size, N)

    wmin = wg.amin(dim=1, keepdim=True)
    wmax = wg.amax(dim=1, keepdim=True)
    scale = (wmax - wmin).clamp(min=1e-8) / 15.0
    z_code = torch.clamp(torch.round(-wmin / scale), 0, 15)
    w_code = torch.clamp(torch.round(wg / scale + z_code), 0, 15)

    w_code = w_code.reshape(K, N).to(torch.int32)
    z_code = z_code.reshape(G, N).to(torch.int32)

    # Pack 8 consecutive K codes into one int32 (nibble k % 8).
    w_code8 = w_code.view(K // 8, 8, N)
    qweight = torch.zeros(K // 8, N, dtype=torch.int32, device=w.device)
    for i in range(8):
        qweight |= w_code8[:, i, :] << (4 * i)

    # Pack 8 consecutive N zero codes into one int32 (nibble n % 8).
    z_code8 = z_code.view(G, N // 8, 8)
    qzeros = torch.zeros(G, N // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        qzeros |= z_code8[:, :, j] << (4 * j)

    scales = scale.squeeze(1).to(dtype)
    return qweight, scales, qzeros
