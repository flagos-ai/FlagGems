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
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry, libtuner, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

tanh = tl_extra_shim.tanh
pow = tl_extra_shim.pow

logger = logging.getLogger(__name__)

# Reused empty context: replaces the create/destroy cost of torch_device_fn.device(...)
# when the device already matches (same as addmm.py); tile dispatch cache:
# _select_activation_out_config is pure, saving ~2us per host call.
_NULL_CONTEXT = nullcontext()
_CONFIG_CACHE = {}


# --- TUNING PARAMS START ---
# Compile-time / launch-time config for the out variant, tuned one variable at a time; semantics unchanged.
# Whether the default entry reuses the deterministic kernel of the out variant; @libtuner autotune
# triggers a device-level abort in do_bench on GCU400, and enabling this bypasses that path entirely.
USE_DETERMINISTIC_FOR_DEFAULT = 1
FALLBACK_TO_ORIGINAL = 0
GROUP_M = 8
USE_EVEN_FASTPATH = 1
USE_FAST_GELU = 1
USE_FAST_RELU = 0
USE_FAST_SCALE = 1
USE_FAST_TANH = 1
BLOCK_M_OVERRIDE = 0
BLOCK_N_OVERRIDE = 0
BLOCK_K_OVERRIDE = 0
NUM_WARPS_OVERRIDE = 0
NUM_STAGES_OVERRIDE = 0
# --- TUNING PARAMS END ---


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
    flagtune_op_name="addmm",
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_activation_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_GELU: tl.constexpr = False,
    IS_FP64: tl.constexpr = False,
):
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if IS_FP64:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        if IS_FP64:
            accumulator += tl.dot(a, b, allow_tf32=False)
        else:
            accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
    bias = tl.load(i_ptrs, mask=c_mask, other=0.0)

    accumulator = accumulator * alpha + bias * beta

    if USE_GELU:
        # GELU with tanh approximation (matches aten::_addmm_activation use_gelu=True)
        # Route by accumulator dtype: float32 truncation cannot meet float64's
        # tolerance, so fp64 evaluates the activation in double precision too.
        # fp16/bf16 and fp32 keep their existing fp32 evaluation.
        if IS_FP64:
            acc_gelu = accumulator
        else:
            acc_gelu = accumulator.to(tl.float32)
        accumulator = (
            0.5
            * acc_gelu
            * (1 + tanh(acc_gelu * 0.79788456 * (1 + 0.044715 * pow(acc_gelu, 2))))
        )
    else:
        # ReLU (default activation)
        accumulator = tl.where(accumulator > 0, accumulator, 0.0)

    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


def _addmm_activation_original(
    bias, mat1, mat2, *, beta=1, alpha=1, use_gelu=False, out=None
):
    """Autotuned implementation kept only as the FALLBACK_TO_ORIGINAL escape
    hatch: its libtuner ``do_bench`` calls ``torch_gcu.synchronize`` and aborts
    the device on gcu400 (exit 134 on the second correctness case)."""
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    if USE_DETERMINISTIC_FOR_DEFAULT:
        # Reuse the deterministic Triton kernel of the out variant to avoid the device-level abort caused by
        # @libtuner autotune running do_bench on GCU400; semantics and tile accuracy are identical.
        if out is None:
            out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
        return _addmm_activation_out(
            bias, mat1, mat2, beta=beta, alpha=alpha, use_gelu=use_gelu, out=out
        )

    logger.debug(
        "GEMS _ADDMM_ACTIVATION, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "use_gelu: %s",
        M,
        N,
        K,
        use_gelu,
    )
    mat1 = mat1.contiguous()
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    bias = bias.broadcast_to((M, N))

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(mat1.device):
        addmm_activation_kernel[grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            bias.stride(0),
            bias.stride(1),
            out.stride(0),
            out.stride(1),
            USE_GELU=use_gelu,
            IS_FP64=mat1.dtype == torch.float64,
        )
    return out


def _addmm_activation(bias, mat1, mat2, *, beta=1, alpha=1, use_gelu=False, out=None):
    """Default path: the deterministic out implementation.

    The autotuned kernel above aborts the device through libtuner's
    ``do_bench`` -> ``torch_gcu.synchronize`` on its first launch, so the
    non-out entry point reuses the deterministic dispatcher.  The old kernel
    stays reachable through ``FALLBACK_TO_ORIGINAL``.
    """
    logger.debug("GEMS_ENFLAME _ADDMM_ACTIVATION")
    if FALLBACK_TO_ORIGINAL:
        return _addmm_activation_original(
            bias, mat1, mat2, beta=beta, alpha=alpha, use_gelu=use_gelu, out=out
        )
    if out is None:
        # new_empty reuses the existing tensor's device/dtype, saving one device/dtype
        # resolution versus torch.empty(device=...) (addmm A/B: 85 -> 81us)
        out = mat1.new_empty((mat1.shape[0], mat2.shape[1]))
    return _addmm_activation_out(
        bias, mat1, mat2, beta=beta, alpha=alpha, use_gelu=use_gelu, out=out
    )


GCU_MAX_GRID_Y = 255


def _select_activation_out_config(M, N, K, dtype, matrix_bias=True):
    """Deterministic GEMM tile dispatch for the fused addmm+activation kernel.

    2026-11-19 perf round (fresh core re-anchor + two sweep rounds, official
    do_bench kernel mode median, GCU400/L600):
      - 16-bit: the K<=512 band takes a single BK=512 masked segment on a
        64-row tile (384^3: 18.0 -> 8.4us; deep-K single segment avoids the
        k-loop pipeline overhead, addmm campaign precedent). Larger K uses
        (256,128,256) tiles with a genuine 2-D matrix bias (1024^3 24.1us /
        2048^3 108.4us / 4096^3 699.9us), beating the addmm base tile
        (128,256,256) by 8-9% for this fused kernel (1024^3 26.6us, 2048^3
        113.2us, 4096^3 759.6us). The 1-D/scalar/colvec bias arms broadcast
        into the tile and only fit the register budget at BM,BN <= 128, so
        non-matrix bias keeps the 128-row tiles (1024^3 vec 26.0us).
      - fp32: small divisible shapes keep addmm's (64,128,128) winner
        (384^3 12.0us); saturated fp32 (M*N*K >= 2^28) switches to
        (128,128,256) (1024^3 57.0 vs 74.0us, 2048^3 311.7 vs 478.1us,
        4096^3 2223 vs 3788us) — a segment the addmm campaign never tuned
        because it delegates to native there.
      - num_warps/num_stages vary within ±2% on the winners (w2 s2 kept).
    """
    if K <= 64 or M <= 64 or N <= 64:
        # Small matrices use shallow tiles to avoid large amounts of masked-out work
        # (small-arm dispatch mirrors addmm._select_addmm_config).
        if K <= 64 or N <= 16:
            block_m, block_n, block_k = 32, 64, 32 if K <= 32 else 64
            num_warps = 4
            num_stages = 3
        else:
            block_m, block_n = 64, 64
            block_k = 128 if K <= 128 else 256
            num_warps = 4
            num_stages = 2
    else:
        if dtype != torch.float32:
            if K <= 512:
                block_m, block_n, block_k = 64, 128, 512
            elif matrix_bias:
                block_m, block_n = 256, 128
                block_k = 256 if K % 256 == 0 else 128
            else:
                block_m, block_n = 128, 128
                block_k = 512 if K % 512 == 0 else (256 if K % 256 == 0 else 128)
            num_warps = 2
            num_stages = 2
        else:
            if M % 64 == 0 and N % 128 == 0 and K % 128 == 0 and M * N * K < (1 << 28):
                block_m, block_n, block_k = 64, 128, 128
                num_warps = 2
                num_stages = 3
            elif K % 256 == 0:
                block_m, block_n, block_k = 128, 128, 256
                num_warps = 2
                num_stages = 2
            else:
                block_m, block_n, block_k = 128, 128, 128
                num_warps = 2
                num_stages = 2

    grid_n = triton.cdiv(N, block_n)
    if grid_n > GCU_MAX_GRID_Y:
        block_n = max(triton.next_power_of_2(triton.cdiv(N, GCU_MAX_GRID_Y)), 16)

    if BLOCK_M_OVERRIDE:
        block_m = BLOCK_M_OVERRIDE
    if BLOCK_N_OVERRIDE:
        block_n = BLOCK_N_OVERRIDE
    if BLOCK_K_OVERRIDE:
        block_k = BLOCK_K_OVERRIDE
    num_warps = NUM_WARPS_OVERRIDE or num_warps
    num_stages = NUM_STAGES_OVERRIDE or num_stages
    return block_m, block_n, block_k, num_warps, num_stages


def _classify_activation_bias(bias, m, n):
    """Route the bias to a 1-D or scalar addressing arm whenever a broadcast
    stride collapses to zero; the matrix arm never receives a zero stride.

    Same routing contract as addmm._classify_bias: a 0-stride scalar used in
    2-D bias-load addressing lowers to an out-of-bounds access that trips a
    device RAS interrupt and kills the process (compiler_issues/
    flaggems_enflame_addmm_zero_stride_bias_load.md). The 1-D arms and any
    genuinely strided 2-D matrix bias are immune. Returns
    (bias, stride_im, stride_in, is_vector, is_scalar, is_colvec).
    """
    if bias.numel() == 1:
        return bias, 0, 0, False, True, False
    if bias.ndim == 1:
        return bias, 0, bias.stride(0), True, False, False
    if bias.shape[0] == m and bias.shape[1] == n:
        bias_stride_m, bias_stride_n = bias.stride()
        if bias_stride_m != 0 and bias_stride_n != 0:
            return bias, bias_stride_m, bias_stride_n, False, False, False
    bias_view = bias.broadcast_to((m, n))
    stride_im = bias_view.stride(0)
    stride_in = bias_view.stride(1)
    if stride_im == 0:
        return bias, 0, stride_in, True, False, False
    if stride_in == 0:
        return bias, stride_im, 0, False, False, True
    return bias_view, stride_im, stride_in, False, False, False


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_activation_out_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_M_VALUE: tl.constexpr,
    EVEN: tl.constexpr,
    USE_GELU: tl.constexpr,
    IS_FP64: tl.constexpr,
    USE_FAST_GELU_VALUE: tl.constexpr,
    USE_FAST_RELU_VALUE: tl.constexpr,
    USE_FAST_SCALE_VALUE: tl.constexpr,
    USE_FAST_TANH_VALUE: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr = False,
    BIAS_IS_SCALAR: tl.constexpr = False,
    BIAS_IS_COLVEC: tl.constexpr = False,
):
    """Deterministic-tile addmm + bias + activation fused kernel with a 1D grouped grid.

    2026-11-19 rework: every address expression stays affine in tl.arange so
    the backend strength-reduces the pointer grids instead of materializing
    i64 pointer tensors in DSM (the old safe-clamped addressing inflated DSM
    and cut SM-side concurrency: 1024^3 fp16 66.3 -> 26.6us at identical
    tiles). The masked path mirrors addmm_kernel (scalar k_remaining masks +
    affine pointer advance); bias loads go through dedicated 1-D/scalar arms
    whenever a broadcast stride collapses to zero, so the 2-D matrix arm
    never sees stride 0 (RAS-kill trap, see _classify_activation_bias).
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    width = GROUP_M_VALUE * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M_VALUE, GROUP_M_VALUE)
    pid_m = group_id * GROUP_M_VALUE + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if IS_FP64:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN:
            # When all three dimensions divide evenly, drop every bounds mask at compile time and keep only the dot
            # pipeline.
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_SIZE_K
            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N),
                other=0.0,
            )
        if IS_FP64:
            accumulator += tl.dot(a, b, allow_tf32=False)
        else:
            accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    if not EVEN:
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if BIAS_IS_VECTOR:
        if EVEN:
            bias = tl.load(i_ptr + offs_cn * stride_in)[None, :]
        else:
            bias = tl.load(i_ptr + offs_cn * stride_in, mask=offs_cn < N, other=0.0)[
                None, :
            ]
    elif BIAS_IS_COLVEC:
        if EVEN:
            bias = tl.load(i_ptr + offs_cm * stride_im)[:, None]
        else:
            bias = tl.load(i_ptr + offs_cm * stride_im, mask=offs_cm < M, other=0.0)[
                :, None
            ]
    elif BIAS_IS_SCALAR:
        bias = tl.load(i_ptr)
    else:
        i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
        if EVEN:
            bias = tl.load(i_ptrs)
        else:
            bias = tl.load(i_ptrs, mask=c_mask, other=0.0)

    if USE_FAST_SCALE_VALUE:
        # When alpha and beta are both 1, two scalar multiplies safely collapse into one add.
        accumulator = accumulator + bias
    else:
        accumulator = accumulator * alpha + bias * beta

    if USE_GELU:
        if IS_FP64:
            acc_gelu = accumulator
        else:
            acc_gelu = accumulator.to(tl.float32)
        if USE_FAST_GELU_VALUE:
            # pow(x, 2) matches x*x point-by-point under IEEE semantics, avoiding a libdevice powf call.
            square = acc_gelu * acc_gelu
        else:
            square = pow(acc_gelu, 2)
        z = acc_gelu * 0.79788456 * (1 + 0.044715 * square)
        if USE_FAST_TANH_VALUE:
            tanh_z = 2.0 * tl.sigmoid(2.0 * z) - 1.0
        else:
            tanh_z = tanh(z)
        accumulator = 0.5 * acc_gelu * (1 + tanh_z)
    else:
        if USE_FAST_RELU_VALUE:
            accumulator = tl.maximum(accumulator, 0.0)
        else:
            accumulator = tl.where(accumulator > 0, accumulator, 0.0)

    c = accumulator.to(bias.dtype)
    if EVEN:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=c_mask)


def _addmm_activation_out(bias, mat1, mat2, *, beta=1, alpha=1, use_gelu=False, out):
    """The out variant uses deterministic dispatch; the original autotune path is used only with
    FALLBACK_TO_ORIGINAL."""
    logger.debug("GEMS_ADDMM_ACTIVATION_OUT")
    if FALLBACK_TO_ORIGINAL:
        return _addmm_activation_original(
            bias, mat1, mat2, beta=beta, alpha=alpha, use_gelu=use_gelu, out=out
        )

    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape
    assert out.shape == (M, N), "Incompatible output shape"

    mat1 = mat1.contiguous()
    if mat2.stride(0) > 1 and mat2.stride(1) > 1:
        mat2 = mat2.contiguous()
    bias_arg, stride_im, stride_in, is_vector, is_scalar, is_colvec = (
        _classify_activation_bias(bias, M, N)
    )
    matrix_bias = not (is_vector or is_scalar or is_colvec)

    key = (M, N, K, mat1.dtype, matrix_bias)
    config = _CONFIG_CACHE.get(key)
    if config is None:
        config = _select_activation_out_config(M, N, K, mat1.dtype, matrix_bias)
        _CONFIG_CACHE[key] = config
    block_m, block_n, block_k, num_warps, num_stages = config
    even = bool(
        USE_EVEN_FASTPATH and M % block_m == 0 and N % block_n == 0 and K % block_k == 0
    )
    fast_scale = bool(USE_FAST_SCALE and alpha == 1 and beta == 1)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    # Skips a context switch when the current device already matches (~2-3us/call, same as addmm)
    device_scope = (
        _NULL_CONTEXT
        if mat1.device.index == torch_device_fn.current_device()
        else torch_device_fn.device(mat1.device)
    )
    with device_scope:
        addmm_activation_out_kernel[grid](
            mat1,
            mat2,
            bias_arg,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            stride_im,
            stride_in,
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_M_VALUE=GROUP_M,
            EVEN=even,
            USE_GELU=use_gelu,
            IS_FP64=mat1.dtype == torch.float64,
            USE_FAST_GELU_VALUE=USE_FAST_GELU,
            USE_FAST_RELU_VALUE=USE_FAST_RELU,
            USE_FAST_SCALE_VALUE=fast_scale,
            USE_FAST_TANH_VALUE=USE_FAST_TANH,
            BIAS_IS_VECTOR=is_vector,
            BIAS_IS_SCALAR=is_scalar,
            BIAS_IS_COLVEC=is_colvec,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out
