"""
FlagGems ARM backend: Triton-CPU INT8 matmul for aten::_int_mm.

Replaces the scalar fallback of aten::_int_mm on CPU with a Triton-CPU
SVE2 i8mm kernel on ARM64 (CIX P1 CD8180, SVE2 + i8mm).

Interface:
    aten::_int_mm(Tensor self: int8, Tensor mat2: int8) -> Tensor: int32
    self  : [M, K]  int8 — already-quantised activation
    mat2  : [K, N]  int8 — weight (column-major, i.e. row-major [K,N])
    output: [M, N] int32

Use cases covered:
    - torchao Int8DynamicActivationInt8WeightConfig
    - Any code that calls torch._int_mm / torch.ops.aten._int_mm on CPU

Routing (same M-branch + padding strategy as quantized_linear_dynamic.py):
    M==1       → BM=1,  BK=4  (ConvertDotGeneric, LLVM unrolls K loop)
    M==2       → BM=2,  BK=4  (2-row ConvertDotGeneric)
    M%64==0    → BM=64, BK=32 (SVE2 i8mm Dynamic ForOp, ~411 GOPS)
    M%8==0     → BM=8,  BK=32 (SVE2 i8mm Dynamic ForOp, ~100-170 GOPS)
    otherwise  → pad to M%8==0, BM=8, BK=32 (e.g. M=84→88)

Unlike quantized_linear_dynamic, no weight cache or quant/dequant fusion
is needed: inputs are already int8, output is int32.

Scalar baseline: 1.9 GOPS (OMP=8 has no effect).
Triton target:   M=1 → 63 GOPS, M=64 → 411 GOPS, M=84→88 → 170 GOPS.
"""

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triton kernel: int8 @ int8 → int32 (row-major weights, BK-loop)
# Reuses same pattern as _i8mm_kernel in quantized_linear_dynamic.
# ---------------------------------------------------------------------------

@triton.jit
def _int8mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """int8 GEMM: A[M,K] int8 @ B[K,N] int8 → C[M,N] int32."""
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
        acc,
    )


# ---------------------------------------------------------------------------
# Weight cache: torchao int8dq provides col-major weights that need
# .contiguous() to become row-major for the Triton kernel.  Without caching,
# this copy (3-11ms per call) dominates every token.  Cache by data_ptr()
# so each weight is made contiguous only once (first call per layer).
# ---------------------------------------------------------------------------
_INT_MM_B_CACHE: dict = {}

# ---------------------------------------------------------------------------
# NEON SDOT kernel for M=1 INT8 GEMV.
# Uses pre-packed weights in SDOT-friendly format: B_packed[K//4, N//4, 4, 4]
# where B_packed[kb, nb, ni, ki] = B_original[kb*4+ki, nb*4+ni].
# K-outer loop maximizes L1 cache reuse for activation vector.
# 3.1-3.6x faster than Triton SMLAL for compute-dominated shapes.
# ---------------------------------------------------------------------------
_SDOT_LIB = None
_SDOT_WEIGHT_CACHE: dict = {}  # (data_ptr, K, N) -> (B_packed, b_ref)


def _get_sdot_lib():
    """Load SDOT functions from TLE NEON .o (auto-compiled from triton.language.extra.cpu.neon).

    The C source is defined in neon.py, compiled by gcc on first use, and cached.
    When Triton kernels are compiled, the same .o gets linked into kernel .so.
    """
    global _SDOT_LIB
    if _SDOT_LIB is not None:
        return _SDOT_LIB
    import ctypes
    import pathlib
    try:
        from triton.language.extra.cpu.neon import get_all_object_files
        import subprocess, shutil, platform
        obj_files = get_all_object_files()
        if not obj_files:
            return None
        # Link .o into a shared lib for ctypes loading
        obj_path = obj_files[0]
        cache_dir = os.path.dirname(obj_path)
        so_path = os.path.join(cache_dir, "tle_neon_sdot.so")
        if not os.path.exists(so_path) or os.path.getmtime(so_path) < os.path.getmtime(obj_path):
            cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
            cc_cmd = [cc, obj_path, "-shared", "-o", so_path,
                      "-fopenmp", "-lgomp", "-lm"]
            if platform.machine() in ("aarch64", "arm64"):
                cc_cmd += ["-march=armv8.2-a+dotprod"]
            subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        lib = ctypes.CDLL(so_path)
        lib.tle_sdot_pack_weights.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64
        ]
        lib.tle_sdot_gemv_m1.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64
        ]
        lib.tle_sdot_gemv_m1_fused_bf16.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64
        ]
        _SDOT_LIB = lib
        return lib
    except Exception:
        return None


def _sdot_enabled():
    return os.getenv("FLAGGEMS_ARM_SDOT", "1").lower() in ("1", "true", "on")


def _get_sdot_packed_weight(b_rowmajor, K, N):
    """Get or create SDOT pre-packed weight. Cached by (data_ptr, K, N).

    Holds a reference to the original tensor to prevent GC from reusing
    the data_ptr address, which would cause stale cache hits.
    """
    key = (b_rowmajor.data_ptr(), K, N)
    if key in _SDOT_WEIGHT_CACHE:
        return _SDOT_WEIGHT_CACHE[key][0]
    lib = _get_sdot_lib()
    if lib is None:
        return None
    packed = torch.empty(K // 4, N // 4, 4, 4, dtype=torch.int8)
    lib.tle_sdot_pack_weights(
        b_rowmajor.data_ptr(), packed.data_ptr(), K, N
    )
    _SDOT_WEIGHT_CACHE[key] = (packed, b_rowmajor)  # hold ref to prevent GC
    return packed


def launch_sdot_fused_bf16(x_bf16, b_rowmajor, w_scale, K, N):
    """Fused BF16→INT8 quant + SDOT GEMV + dequant→BF16 via TLE NEON (neon.py).

    Args:
        x_bf16: [K] bfloat16 activation (1D, contiguous)
        b_rowmajor: [K, N] int8 weight (row-major, will be pre-packed and cached)
        w_scale: [N] float32 per-channel weight scale
        K, N: dimensions

    Returns:
        [N] bfloat16 output, or None if not applicable.
    """
    if not _sdot_enabled():
        return None
    if K % 4 != 0 or N % 4 != 0:
        return None
    lib = _get_sdot_lib()
    if lib is None:
        return None
    packed = _get_sdot_packed_weight(b_rowmajor, K, N)
    if packed is None:
        return None
    out = torch.empty(N, dtype=torch.bfloat16)
    lib.tle_sdot_gemv_m1_fused_bf16(
        x_bf16.data_ptr(), packed.data_ptr(), w_scale.data_ptr(),
        out.data_ptr(), K, N
    )
    return out


def _launch_sdot_m1(a, b_rowmajor, K, N):
    """Launch NEON SDOT M=1 GEMV via TLE NEON (neon.py).
    Returns [1, N] int32 or None if not applicable."""
    if not _sdot_enabled():
        return None
    if K % 4 != 0 or N % 4 != 0:
        return None
    lib = _get_sdot_lib()
    if lib is None:
        return None
    packed = _get_sdot_packed_weight(b_rowmajor, K, N)
    if packed is None:
        return None
    out = torch.empty(1, N, dtype=torch.int32)
    lib.tle_sdot_gemv_m1(
        a.data_ptr(), packed.data_ptr(), out.data_ptr(),
        K, N
    )
    return out


def _triton_int_mm(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """
    Triton-CPU replacement for aten::_int_mm on ARM64.

    self : [M, K] int8  — activation (changes every token, not cached)
    mat2 : [K, N] int8  — weight (fixed after quantization, cached by data_ptr)
    Returns [M, N] int32
    """
    assert self.dtype == torch.int8 and mat2.dtype == torch.int8, (
        f"_int_mm expects int8 inputs, got {self.dtype}, {mat2.dtype}"
    )
    M, K = self.shape
    K2, N = mat2.shape
    assert K == K2, f"_int_mm shape mismatch: [{M},{K}] @ [{K2},{N}]"

    # Activation: always contiguous (per-token, no cache)
    a = self.contiguous()

    # Weight: cache row-major copy — first call per layer pays the copy cost;
    # all subsequent token decodes are ~free (dict lookup only).
    b_key = mat2.data_ptr()
    if b_key not in _INT_MM_B_CACHE:
        _INT_MM_B_CACHE[b_key] = mat2.contiguous()
    b = _INT_MM_B_CACHE[b_key]

    BN = 64
    BK_prefill = 32

    # Fallback for non-BN-aligned N (uncommon in practice)
    if N % BN != 0:
        logger.debug("FlagGems _int_mm: N=%d not %%64, using int32 fallback", N)
        return a.to(torch.int32) @ b.to(torch.int32)

    # ------------------------------------------------------------------
    # Decode M=1: NEON SDOT with pre-packed weights via torch.ops custom op.
    # Pre-packs B[K,N] → B_packed[K//4, N//4, 4, 4] SDOT lane format.
    # Uses K-outer loop for L1 cache reuse. 2.5x faster than Triton SMLAL.
    # Falls back to Triton SMLAL if SDOT not available.
    # ------------------------------------------------------------------
    if M == 1:
        sdot_result = _launch_sdot_m1(a, b, K, N)
        if sdot_result is not None:
            return sdot_result

        # Fallback: Triton SMLAL (BM=1, BK=4)
        BM, BK = 1, 4
        out = torch.empty(M, N, dtype=torch.int32)
        _int8mm_kernel[(1, N // BN)](
            a, b, out,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
        return out

    if M == 2:
        BM, BK = 2, 4
        out = torch.empty(M, N, dtype=torch.int32)
        _int8mm_kernel[(1, N // BN)](
            a, b, out,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
        return out

    # ------------------------------------------------------------------
    # Prefill path (M ≥ 3): BK=32, target SVE2 i8mm Dynamic ForOp.
    # Pad M to next multiple of 8 to unlock Dynamic ForOp for all shapes.
    # ------------------------------------------------------------------
    BK = BK_prefill if K % BK_prefill == 0 else 4

    if M % 64 == 0:
        BM = 64
        a_kernel, M_kernel = a, M
    elif M % 8 == 0:
        BM = 8
        a_kernel, M_kernel = a, M
    else:
        # Zero-pad to next multiple of 8
        M_kernel = ((M + 7) // 8) * 8
        BM = 8
        a_kernel = torch.zeros(M_kernel, K, dtype=torch.int8)
        a_kernel[:M].copy_(a)

    out_kernel = torch.empty(M_kernel, N, dtype=torch.int32)
    grid = (M_kernel // BM, N // BN)

    _int8mm_kernel[grid](
        a_kernel, b, out_kernel,
        M_kernel, N, K,
        a_kernel.stride(0), a_kernel.stride(1),
        b.stride(0), b.stride(1),
        out_kernel.stride(0), out_kernel.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )

    return out_kernel[:M] if M_kernel != M else out_kernel


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_int_mm_lib = None   # keep reference alive to prevent GC


def register():
    """
    Register Triton implementation for aten::_int_mm on CPU.
    Idempotent: safe to call multiple times.
    """
    global _int_mm_lib
    if _int_mm_lib is not None:
        return

    try:
        _int_mm_lib = torch.library.Library("aten", "IMPL")
        _int_mm_lib.impl(
            "_int_mm",
            _triton_int_mm,
            "CPU",
            allow_override=True,
        )
        logger.debug(
            "FlagGems ARM: registered Triton-CPU i8mm for aten::_int_mm"
        )
    except Exception as e:
        logger.warning(
            "FlagGems ARM: failed to register aten::_int_mm override: %s", e
        )
