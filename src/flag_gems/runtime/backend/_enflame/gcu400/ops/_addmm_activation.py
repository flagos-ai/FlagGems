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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry, libtuner, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

tanh = tl_extra_shim.tanh
pow = tl_extra_shim.pow

logger = logging.getLogger(__name__)


# --- TUNING PARAMS START ---
# out 变体的编译期/启动期配置，按优化轮次单变量调整，不改变算子语义。
# 默认入口是否复用 out 变体的确定性 kernel；@libtuner autotune 在 GCU400 上
# 的 do_bench 会触发设备级 abort，打开后可完全绕开该路径。
USE_DETERMINISTIC_FOR_DEFAULT = 1
FALLBACK_TO_ORIGINAL = 0
GROUP_M = 8
USE_EVEN_FASTPATH = 1
USE_FAST_GELU = 1
USE_FAST_RELU = 0
USE_FAST_SCALE = 1
USE_FAST_TANH = 1
# 默认直接使用 bias 的广播视图，不再物化成 M×N 连续张量。
# 物化是早期为规避 @libtuner autotune 的 do_bench 设备级 abort 引入的
# workaround；当前确定性 tile 分派已绕开 autotune，stride=0 广播视图在
# GCU400 上可正常执行（全量 72 case 与多轮微基准均验证无 abort），
# 省去每次调用一次 M×N 分配与整块拷贝。
BIAS_CONTIGUOUS = 0
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
        # 复用 out 变体的确定性 Triton kernel，避免 @libtuner 的 autotune
        # 在 GCU400 上执行 do_bench 时触发设备级 abort；语义与 tile 精度一致。
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
    if FALLBACK_TO_ORIGINAL:
        return _addmm_activation_original(
            bias, mat1, mat2, beta=beta, alpha=alpha, use_gelu=use_gelu, out=out
        )
    if out is None:
        out = torch.empty(
            (mat1.shape[0], mat2.shape[1]), device=mat1.device, dtype=mat1.dtype
        )
    return _addmm_activation_out(
        bias, mat1, mat2, beta=beta, alpha=alpha, use_gelu=use_gelu, out=out
    )


GCU_MAX_GRID_Y = 255


def _select_activation_out_config(M, N, K, dtype):
    """按 shape 与 dtype 确定性选择 GEMM tile，规避 autotune 在 GCU 上的设备级 abort。"""
    if K <= 64 or M <= 64 or N <= 64:
        # 小矩阵用浅 tile，避免大量被 mask 掉的无效计算。
        block_m, block_n = 32, 64
        block_k = 32 if K <= 32 else 64
        num_warps = 4
    else:
        block_m = 128
        block_n = 128 if N <= 512 else 256
        block_k = 256 if dtype != torch.float32 and K % 256 == 0 else 128
        num_warps = 2 if dtype == torch.float32 or M >= 1024 else 4

    # GCU400 GEMM 的两条硬约束：K 不对齐时 BK 必须 <=64；N 不对齐时 BN 不能取 256。
    if K % block_k != 0 and block_k > 64:
        block_k = 64
    if N % block_n != 0 and block_n > 128:
        block_n = 128
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
    num_stages = NUM_STAGES_OVERRIDE or 3
    return block_m, block_n, block_k, num_warps, num_stages


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
):
    """确定性 tile 的 addmm + bias + activation 融合 kernel，一维 grouped grid。"""
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
    # gcu400 masked loads still touch their lanes and leave them holding stale
    # memory, so every masked lane must address a valid element (clamped to 0)
    # and its value must be re-zeroed before it reaches the dot product.
    safe_am = tl.where(offs_am < M, offs_am, 0)
    safe_bn = tl.where(offs_bn < N, offs_bn, 0)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if IS_FP64:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN:
            # 三维都能整除时编译期去掉所有越界 mask，只保留点积流水。
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_local = k * BLOCK_SIZE_K + offs_k
            k_ok = k_local < K
            safe_k = tl.where(k_ok, k_local, 0)
            a_tile_ptrs = (
                a_ptr + safe_am[:, None] * stride_am + safe_k[None, :] * stride_ak
            )
            b_tile_ptrs = (
                b_ptr + safe_k[:, None] * stride_bk + safe_bn[None, :] * stride_bn
            )
            a_mask = (offs_am[:, None] < M) & k_ok[None, :]
            b_mask = k_ok[:, None] & (offs_bn[None, :] < N)
            a = tl.load(a_tile_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_tile_ptrs, mask=b_mask, other=0.0)
            a = tl.where(a_mask, a, 0.0)
            b = tl.where(b_mask, b, 0.0)
        if IS_FP64:
            accumulator += tl.dot(a, b, allow_tf32=False)
        else:
            accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    safe_cm = tl.where(offs_cm < M, offs_cm, 0)
    safe_cn = tl.where(offs_cn < N, offs_cn, 0)
    c_ptrs = c_ptr + stride_cm * safe_cm[:, None] + stride_cn * safe_cn[None, :]
    i_ptrs = i_ptr + stride_im * safe_cm[:, None] + stride_in * safe_cn[None, :]
    if EVEN:
        bias = tl.load(i_ptrs)
    else:
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        bias = tl.load(i_ptrs, mask=c_mask, other=0.0)
        bias = tl.where(c_mask, bias, 0.0)

    if USE_FAST_SCALE_VALUE:
        # alpha 与 beta 都为 1 时，两次标量乘法可安全化简为一次加法。
        accumulator = accumulator + bias
    else:
        accumulator = accumulator * alpha + bias * beta

    if USE_GELU:
        if IS_FP64:
            acc_gelu = accumulator
        else:
            acc_gelu = accumulator.to(tl.float32)
        if USE_FAST_GELU_VALUE:
            # pow(x, 2) 与 x*x 在 IEEE 语义下逐点一致，省掉 libdevice powf。
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
    """out 变体走确定性分派；原 autotune 路径仅在 FALLBACK_TO_ORIGINAL 时使用。"""
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
    bias_view = bias.broadcast_to((M, N))
    if BIAS_CONTIGUOUS:
        # 仅在显式要求连续布局（或需要回退到旧 workaround）时才物化广播视图。
        bias_view = bias_view.contiguous()

    block_m, block_n, block_k, num_warps, num_stages = _select_activation_out_config(
        M, N, K, mat1.dtype
    )
    even = bool(
        USE_EVEN_FASTPATH and M % block_m == 0 and N % block_n == 0 and K % block_k == 0
    )
    fast_scale = bool(USE_FAST_SCALE and alpha == 1 and beta == 1)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)

    with torch_device_fn.device(mat1.device):
        addmm_activation_out_kernel[grid](
            mat1,
            mat2,
            bias_view,
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
            bias_view.stride(0),
            bias_view.stride(1),
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
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out
