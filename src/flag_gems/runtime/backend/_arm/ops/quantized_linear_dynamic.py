"""
FlagGems ARM backend: Triton-CPU INT8 GEMM for quantized::linear_dynamic.

Replaces the OneDNN/ACL implementation of torch.ops.quantized.linear_dynamic
with a Triton-CPU i8mm kernel on ARM64 (SVE2 + i8mm).

Kernel configs (validated on CIX P1 CD8180):
  M=1          → BM=1,  BN=64, BK=4  (ConvertDotGeneric, 63 GOPS decode)
  M=2          → BM=2,  BN=64, BK=4  (ConvertDotGeneric, LLVM unrolls K=4)
  M%64==0      → BM=64, BN=64, BK=32 (SVE2 i8mm dynamic ForOp, 411 GOPS)
  M%8==0       → BM=8,  BN=64, BK=32 (SVE2 i8mm dynamic ForOp, 100-128 GOPS)
  otherwise    → pad M to next %8==0, BM=8 (zero-pad extra rows, then slice output)

Fusion optimisation (2026-03-06):
  _i8mm_fused_kernel takes FP32 activation input directly and outputs FP32.
  Quantisation (FP32→INT8) and dequantisation (INT32→FP32) are fused inside
  the kernel, eliminating 7 separate PyTorch operator calls per linear layer:
    BEFORE: abs, max, div, round_, clamp_, to(int8), empty(int32),
            dot-kernel, to(float32), mul_
    AFTER:  abs, max,  fused-kernel  (saves ~17 ms/tok on Qwen3-1.7B)

Weight tiling optimisation (2026-03-06):
  _i8mm_fused_tiled_kernel uses pre-tiled weights [K//BK, N//BN, BK, BN].
  Each B tile is contiguous in memory, eliminating strided cache-miss pattern
  of the row-major [K,N] layout (stride_bk = N = 18944 causes L2 misses).
  Applied to all prefill paths (M≥4); decode (M=1,2) keeps row-major layout.
  Extra memory: ~1x weight size (e.g. +1.7 GB for Qwen3-1.7B). One-time cost
  at first inference per weight.

Weight cache: keyed on w.data_ptr() (stable physical address).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Tile dimensions for prefill weight layout (must match kernel constexprs)
_TILE_BK = 32
_TILE_BN = 64

# Runtime flag: enable M-padding for non-M%8 prefill shapes (Phase 4).
# Set to False to revert to Phase 3 BM=4 static path (for benchmarking).
_ENABLE_PADDING = True


# ---------------------------------------------------------------------------
# Fused + tiled kernel: FP32 input → INT8 quant → tiled INT8 GEMM → FP32 out
# Used for prefill paths (M≥4, BK=32) where B tile is contiguous in memory.
# ---------------------------------------------------------------------------

@triton.jit
def _i8mm_fused_tiled_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    N_TILES,              # int32: N // BLOCK_N  (number of N-tiles)
    inv_x_scale,          # float32 scalar: 127.0 / x_abs_max
    out_scale,            # float32 scalar: (x_abs_max / 127.0) * w_scale
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,   # must equal _TILE_BN (64)
    BLOCK_K: tl.constexpr,   # must equal _TILE_BK (32)
):
    """
    Fused INT8 GEMM with tiled weight layout.

    A[M,K]   fp32   (activation, row-major)
    B tiled  int8   layout [K//BK, N//BN, BK, BN] — each tile contiguous
    C[M,N]   fp32   output

    The tiled layout ensures each B tile load is a contiguous BK*BN-byte
    block, eliminating the stride-bk=N cache-miss pattern of row-major [K,N].
    SVE2 i8mm (smmla) path fires as before: both operands are int8.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load FP32 activation tile; quantise to INT8 in-kernel
        a_fp32 = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
        a_scaled = a_fp32 * inv_x_scale
        a_clamped = tl.minimum(tl.maximum(a_scaled, -128.0), 127.0)
        a_int8 = a_clamped.to(tl.int8)

        # Load tiled B: tile (k, pid_n) is contiguous BK*BN bytes
        # Layout: b_ptr[k * N_TILES + pid_n][BK][BN]
        b_base = b_ptr + (k * N_TILES + pid_n) * BLOCK_K * BLOCK_N
        b = tl.load(
            b_base
            + tl.arange(0, BLOCK_K)[:, None] * BLOCK_N
            + tl.arange(0, BLOCK_N)[None, :]
        )
        acc += tl.dot(a_int8, b)

    # Dequantise: int32 → float32, scale and store
    c_fp32 = acc.to(tl.float32) * out_scale
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c_fp32,
    )


# ---------------------------------------------------------------------------
# Fused kernel: FP32 input → INT8 quant → row-major INT8 GEMM → FP32 out
# Used for decode paths (M=1,2, BK=4) where tile is tiny (4×64 bytes).
# ---------------------------------------------------------------------------

@triton.jit
def _i8mm_fused_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    inv_x_scale,          # float32 scalar: 127.0 / x_abs_max
    out_scale,            # float32 scalar: (x_abs_max / 127.0) * w_scale
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused INT8 GEMM with row-major weight layout [K, N].
    Used for decode (M=1,2, BK=4): LLVM fully unrolls K=4 loop.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        a_fp32 = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
        a_scaled = a_fp32 * inv_x_scale
        a_clamped = tl.minimum(tl.maximum(a_scaled, -128.0), 127.0)
        a_int8 = a_clamped.to(tl.int8)

        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )
        acc += tl.dot(a_int8, b)

    c_fp32 = acc.to(tl.float32) * out_scale
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c_fp32,
    )


# ---------------------------------------------------------------------------
# Legacy unfused kernel (kept for reference / debugging)
# ---------------------------------------------------------------------------

@triton.jit
def _i8mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Unfused INT8 GEMM: A int8, B int8 → C int32."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.int32),
    )


# ---------------------------------------------------------------------------
# Weight cache
# ---------------------------------------------------------------------------

# w_raw.data_ptr() → (weight_kn [K,N], weight_tiled [K//BK,N//BN,BK,BN] or None,
#                     weight_scale float, bias or None)
_weight_cache: dict = {}


def _get_weight(W_prepack):
    w, bias = W_prepack.unpack()                       # w: qint8 [N, K]
    key = w.data_ptr()                                 # stable physical address
    if key in _weight_cache:
        return _weight_cache[key]

    # Row-major [K, N] for decode (M=1,2, BK=4)
    weight_kn = w.int_repr().T.contiguous()            # int8 [K, N]
    K, N = weight_kn.shape

    # Tiled [K//BK, N//BN, BK, BN] for prefill (M≥4, BK=32, BN=64)
    # Each tile is BK*BN contiguous bytes → eliminates strided cache misses.
    BK, BN = _TILE_BK, _TILE_BN
    if K % BK == 0 and N % BN == 0:
        weight_tiled = (
            weight_kn
            .reshape(K // BK, BK, N // BN, BN)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # int8 [K//BK, N//BN, BK, BN]
    else:
        weight_tiled = None
        logger.debug(
            "FlagGems ARM: K=%d N=%d not divisible by BK=%d BN=%d; "
            "tiled layout disabled for this layer", K, N, BK, BN
        )

    weight_scale = float(w.q_scale())
    entry = (weight_kn, weight_tiled, weight_scale, bias)
    _weight_cache[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def _triton_quantized_linear_dynamic(X, W_prepack, reduce_range=False):
    """
    Triton-CPU replacement for torch.ops.quantized.linear_dynamic (CPU).

    X        : float32 tensor, shape [..., K]
    W_prepack: torch.ScriptObject (LinearPackedParamsBase), qint8 [N, K]
    Returns  : float32 tensor, shape [..., N]

    Decode (M=1,2): _i8mm_fused_kernel, row-major weight [K,N], BK=4.
      LLVM fully unrolls K=4 loop → fastest for tiny GEMV.

    Prefill (M≥3): _i8mm_fused_tiled_kernel, tiled weight [K//32,N//64,32,64], BK=32.
      BM=64 for M%64==0; BM=8 for all other M (with zero-padding if M%8≠0).
      Padding: M=84 → M_kernel=88 (+4 zero rows), unlocks Dynamic ForOp path
      (100-128 GOPS) vs old BM=4 static path (57-73 GOPS).
    """
    weight_kn, weight_tiled, weight_scale, bias = _get_weight(W_prepack)

    K = X.shape[-1]
    N = weight_kn.shape[1]
    orig_shape = X.shape

    x2d = X.view(-1, K)
    M = x2d.shape[0]

    # Compute activation scale (one reduction, unavoidable for per-tensor quant)
    x_abs_max = x2d.abs().max().item()
    if x_abs_max == 0.0:
        out2d = torch.zeros(M, N, dtype=torch.float32)
        if bias is not None:
            out2d = out2d + bias
        return out2d.view(*orig_shape[:-1], N)

    inv_x_scale = 127.0 / x_abs_max
    out_scale   = (x_abs_max / 127.0) * weight_scale

    # ------------------------------------------------------------------
    # Decode paths (M=1,2): row-major weight, BK=4, ConvertDotGeneric.
    # ------------------------------------------------------------------
    if M == 1:
        BM, BN, BK = 1, 64, 4
        out2d = torch.empty(M, N, dtype=torch.float32)
        _i8mm_fused_kernel[(1, N // BN)](
            x2d, weight_kn, out2d,
            M, N, K,
            x2d.stride(0), x2d.stride(1),
            weight_kn.stride(0), weight_kn.stride(1),
            out2d.stride(0), out2d.stride(1),
            inv_x_scale=inv_x_scale, out_scale=out_scale,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )

    elif M == 2:
        BM, BN, BK = 2, 64, 4
        out2d = torch.empty(M, N, dtype=torch.float32)
        _i8mm_fused_kernel[(1, N // BN)](
            x2d, weight_kn, out2d,
            M, N, K,
            x2d.stride(0), x2d.stride(1),
            weight_kn.stride(0), weight_kn.stride(1),
            out2d.stride(0), out2d.stride(1),
            inv_x_scale=inv_x_scale, out_scale=out_scale,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )

    # ------------------------------------------------------------------
    # Prefill path (M≥3).
    #
    # Routing observed empirically via A/B vs commit 80be6a2e^:
    #   M%64==0   → legacy _i8mm_kernel (fused kernel regresses ~15-20%
    #                here due to BM=64 BK=32 epilog register pressure).
    #   M=4       → legacy BM=1 BK=4 (BM=4 BK=32 SVE2 static path is slower
    #                than BM=1 BK=4 ConvertDotGeneric for this tiny shape).
    #   M%8==0    → fused kernel BM=8 BK=32 (SVE2 i8mm Dynamic ForOp, ~1.4x).
    #   otherwise → pad to %8, fused BM=8 BK=32.
    # ------------------------------------------------------------------
    elif M % 64 == 0:
        # Legacy path: external quant → _i8mm_kernel (int8×int8→int32) → external dequant.
        # Fused kernel's BM=64 epilog hurts LLVM register allocation here.
        BM, BN, BK = 64, 64, 32
        # NOTE: no .round_() — match fused kernel's .to(int8) truncate behavior.
        # Rounding here (when fused kernel truncates) creates argmax drift at
        # long generations because this M's rounding mode differs from other M's.
        x_q = (x2d * inv_x_scale).clamp_(-128, 127).to(torch.int8)
        c_i32 = torch.empty(M, N, dtype=torch.int32)
        _i8mm_kernel[(M // BM, N // BN)](
            x_q, weight_kn, c_i32,
            M, N, K,
            x_q.stride(0), x_q.stride(1),
            weight_kn.stride(0), weight_kn.stride(1),
            c_i32.stride(0), c_i32.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
        out2d = c_i32.to(torch.float32).mul_(out_scale)

    elif M == 4:
        # Legacy BM=1 BK=4 path: faster than BM=4 BK=32 static i8mm here.
        BM, BN, BK = 1, 64, 4
        # NOTE: no .round_() — match fused kernel's .to(int8) truncate behavior.
        # Rounding here (when fused kernel truncates) creates argmax drift at
        # long generations because this M's rounding mode differs from other M's.
        x_q = (x2d * inv_x_scale).clamp_(-128, 127).to(torch.int8)
        c_i32 = torch.empty(M, N, dtype=torch.int32)
        _i8mm_kernel[(M, N // BN)](
            x_q, weight_kn, c_i32,
            M, N, K,
            x_q.stride(0), x_q.stride(1),
            weight_kn.stride(0), weight_kn.stride(1),
            c_i32.stride(0), c_i32.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
        out2d = c_i32.to(torch.float32).mul_(out_scale)

    else:
        # Fused kernel path: BM=8 BK=32 (Dynamic ForOp SVE2 i8mm, wins here).
        use_tiled = weight_tiled is not None
        BN, BK = 64, 32

        if M % 8 == 0:
            BM = 8
            x_kernel, M_kernel = x2d, M
        elif _ENABLE_PADDING:
            # Pad to next multiple of 8 → Dynamic ForOp path
            # e.g. M=84 → M_kernel=88 (4 extra zero rows)
            M_kernel = ((M + 7) // 8) * 8
            BM = 8
            x_kernel = torch.zeros(M_kernel, K, dtype=x2d.dtype)
            x_kernel[:M].copy_(x2d)
        else:
            # Phase 3 fallback: no padding, BM=4 if aligned else BM=1
            BM = 4 if M % 4 == 0 else 1
            x_kernel, M_kernel = x2d, M

        out_kernel = torch.empty(M_kernel, N, dtype=torch.float32)
        grid = (M_kernel // BM, N // BN)

        if use_tiled:
            _i8mm_fused_tiled_kernel[grid](
                x_kernel, weight_tiled, out_kernel,
                M_kernel, N, K,
                x_kernel.stride(0), x_kernel.stride(1),
                out_kernel.stride(0), out_kernel.stride(1),
                N // BN,
                inv_x_scale=inv_x_scale, out_scale=out_scale,
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            )
        else:
            _i8mm_fused_kernel[grid](
                x_kernel, weight_kn, out_kernel,
                M_kernel, N, K,
                x_kernel.stride(0), x_kernel.stride(1),
                weight_kn.stride(0), weight_kn.stride(1),
                out_kernel.stride(0), out_kernel.stride(1),
                inv_x_scale=inv_x_scale, out_scale=out_scale,
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            )

        # Slice off the padding rows (out_kernel[:M] is a view, no copy)
        out2d = out_kernel[:M] if M_kernel != M else out_kernel

    if bias is not None:
        out2d = out2d + bias
    return out2d.view(*orig_shape[:-1], N)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_quantized_lib = None   # keep reference alive to prevent GC


def register():
    """
    Register Triton implementation for quantized::linear_dynamic on CPU.
    Idempotent: safe to call multiple times.
    """
    global _quantized_lib
    if _quantized_lib is not None:
        return

    try:
        _quantized_lib = torch.library.Library("quantized", "IMPL")
        _quantized_lib.impl(
            "linear_dynamic",
            _triton_quantized_linear_dynamic,
            "CPU",
            allow_override=True,
        )
        logger.debug(
            "FlagGems ARM: registered Triton-CPU i8mm (fused+tiled) for quantized::linear_dynamic"
        )
    except Exception as e:
        logger.warning(
            f"FlagGems ARM: failed to register quantized::linear_dynamic override: {e}"
        )
