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
import os

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


autotune_decorator = triton.autotune(
    configs=[],
    generate_configs="addmm",
    key=["M", "N", "K"],
)


# 2026-08-31 (XPU 4): the generate_configs="addmm" autotune path is NOT
# numerically sound on this backend, so it is no longer the default.
#
# Evidence (harness/results/performance/addmm_out_xpu4_20260831/):
#   * probe_config_envelope.log - 31 of 336 configs emitted by
#     triton.runtime.autotuner.block_size_candidates(..., "addmm") return
#     grossly wrong results (max abs err 4.6 .. 107) on fp16 and fp32. The
#     defect is num_stages-independent (probe_fp32_495_5333_71.log: the same
#     6 tiles are wrong at stages=2 and stages=3) and tracks the generated
#     BLOCK_SIZE_N/BLOCK_SIZE_K tile, i.e. it is the known TritonXPU
#     "large masked tile returns silently wrong values" family.
#   * probe_production_oracle_at1.log - through the real dispatch, fp16
#     1024^3 (max abs 101.7) and fp16 4096^3 (max abs 147.1) are corrupt.
#     probe_production_oracle_at0.log - heuristics path: 0/18 bad cells.
#   * pytest -m addmm --ref cpu: autotune path 8 failed (fp32 495x5333x71),
#     heuristics path 72 passed.
# Autotune also *selects by timing*, so on this backend correctness would be
# nondeterministic. Pruning was rejected: the bad set spans BLOCK_SIZE_N
# 192..512 and overlaps the good set, so no defensible prune rule exists from
# the measured envelope.
#
# Set KLX_USE_AUTOTUNE=1 to opt back into the autotune path for tuning
# experiments; it is known-unsound and must not be used for accuracy runs.
KLX_USE_AUTOTUNE = os.environ.get("KLX_USE_AUTOTUNE", "0") == "1"

if not KLX_USE_AUTOTUNE:

    # XPU tile sweep probe (2026-08-13, XPU 7, 4 unique core shapes x 3 dtypes,
    # direct do_bench): BM=BN=256 / warps=8 / stages=3 wins on all dtypes; the
    # reduction tile BK is dtype-dependent on this backend - fp16 prefers BK=256
    # (4096^3: 0.83x vs 0.56x at BK=128), while bf16/fp32 prefer BK=128
    # (4096^3: 0.81x/0.95x vs 0.54x/0.29x at BK=256). fp32 BK=256 collapses
    # (4.7ms vs 1.46ms on 4096^3). Baseline (128x128x128, no swizzle, warps=4)
    # equal-weight mean speedup ~0.53x vs candidate ~0.67x direct A/B.
    # Small shapes (M,N <= 512) keep the 128-tile warps=4 config: the 256-tile
    # warps=8 launch overhead regresses 384^3 by ~6%.

    def heur_block_m(args):
        M = args["M"]
        if M <= 512:
            return 128
        return 256

    def heur_block_n(args):
        N = args["N"]
        if N <= 512:
            return 128
        return 256

    def heur_block_k(args):
        # Wrapper passes BLOCK_K_CHOICE (fp16 -> 256, else 128).
        if args.get("BLOCK_K_CHOICE", 128) == 256:
            return 256
        return 128

    def heur_warps(args):
        if args["M"] <= 512 and args["N"] <= 512:
            return 4
        return 8

    def heur_stages(args):
        # stages=3 is the value validated by the 2026-08-13 XPU tile sweep above.
        # It must be supplied *here* and never as an explicit launch kwarg: the
        # default path is triton.autotune(generate_configs="addmm"), whose
        # Config.all_kwargs() always carries num_stages, so a caller-side
        # num_stages= kwarg raises
        #   TypeError: JITFunction.run() got multiple values for keyword
        #   argument 'num_stages'
        # See harness/solution/performance/
        #     addmm_family_num_stages_fix_xpu4_20260831.md
        return 3

    autotune_decorator = triton.heuristics(
        {
            "BLOCK_SIZE_M": heur_block_m,
            "BLOCK_SIZE_N": heur_block_n,
            "BLOCK_SIZE_K": heur_block_k,
            "num_warps": heur_warps,
            "num_stages": heur_stages,
        }
    )


@libentry()
@autotune_decorator
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
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
    GROUP_M: tl.constexpr,
    BLOCK_K_CHOICE,
):
    pid = ext.program_id(0)
    if GROUP_M > 1:
        grid_m = tl.cdiv(M, BLOCK_SIZE_M)
        grid_n = tl.cdiv(N, BLOCK_SIZE_N)
        # re-order program ID for better L2 reuse along the N dimension
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size
    else:
        pid_m = ext.program_id(1)
        pid_n = ext.program_id(2)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

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
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
    bias = tl.load(i_ptrs, mask=c_mask, other=0.0)

    accumulator = accumulator * alpha + bias * beta
    # Let tl.store convert to the output pointer dtype. The dtype-out variant
    # may use fp32 output with fp16/bf16 inputs and an input-dtype bias.
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _bias_with_unit_inner_stride(bias, shape):
    """Broadcast ``bias`` to ``shape`` while keeping its inner stride equal to 1.

    A 2D block load whose inner (N) stride is not 1 is mis-lowered on this
    backend: the inner stride is silently treated as 1.  A bias that broadcasts
    along N therefore reads ``bias[m + n]`` instead of ``bias[m]``, and a
    1-element bias reads far past its own allocation (measured: fp32 (1,1) bias
    picks up the caller's uninitialised ``out`` bytes).  Broadcasting along M
    (``stride_im == 0``) uses a unit inner stride and stays free of charge, so
    only the inner dimension has to be materialised.
    """
    b = bias.broadcast_to(shape)
    if b.stride(1) != 1 and b.shape[1] > 1:
        if b.stride(0) == 0:
            # A single distinct row: materialise that row only and keep the
            # broadcast along M (strides become (0, 1), which is exact).
            b = b[:1].contiguous().broadcast_to(shape)
        else:
            b = b.contiguous()
    return b


def _dest_with_unit_inner_stride(out, M, N):
    """Pick the tensor the kernel stores into, plus its (row, column) strides.

    ``tl.store`` of a 2D block whose inner (N) stride is not 1 is mis-lowered the
    same way as the bias load: the inner stride is treated as 1, so each program
    writes a contiguous run of N elements per row.  Measured on HEAD with
    ``out = big[:, ::2]`` (M=65, N=64, fp32): 2080 of the 4160 ``out`` elements
    keep their previous content while 2080 interleaved neighbour columns of the
    base allocation are clobbered; a column-major ``out`` loses everything but
    the first tile row (4032/4160 stale).  A masked store cannot avoid this - the
    store is bounded by a ``memref.subview``, not by the mask - so the kernel is
    kept on a unit inner stride and the result is placed afterwards.
    """
    stride_cm, stride_cn = out.stride()
    if stride_cn == 1 or N <= 1:
        return out, stride_cm, stride_cn
    dest = torch.empty((M, N), device=out.device, dtype=out.dtype)
    return dest, dest.stride(0), dest.stride(1)


def addmm(bias, mat1, mat2, *, beta=1.0, alpha=1.0):
    logger.debug("GEMS_KUNLUNXIN ADDMM")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    mat1 = mat1.contiguous()
    # mat2 = mat2.contiguous()
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    bias = _bias_with_unit_inner_stride(bias, out.shape)

    block_k_choice = 256 if mat1.dtype == torch.float16 else 128
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    dest, stride_cm, stride_cn = _dest_with_unit_inner_stride(out, M, N)
    with torch_device_fn.device(mat1.device):
        addmm_kernel[grid](
            mat1,
            mat2,
            bias,
            dest,
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
            stride_cm,
            stride_cn,
            GROUP_M=8,
            BLOCK_K_CHOICE=block_k_choice,
            # NOTE: do NOT pass num_stages here. The default decorator is
            # triton.autotune(generate_configs="addmm"), which injects
            # num_stages from every generated Config -> duplicate keyword.
            # KLX_USE_AUTOTUNE=0 gets stages=3 from heur_stages instead.
        )
    if dest is not out:
        out.copy_(dest)
    return out


def addmm_out(bias, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    logger.debug("GEMS_KUNLUNXIN ADDMM_OUT")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"

    mat1 = mat1.contiguous()
    bias = _bias_with_unit_inner_stride(bias, out.shape)

    block_k_choice = 256 if mat1.dtype == torch.float16 else 128
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    dest, stride_cm, stride_cn = _dest_with_unit_inner_stride(out, M, N)
    with torch_device_fn.device(mat1.device):
        addmm_kernel[grid](
            mat1,
            mat2,
            bias,
            dest,
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
            stride_cm,
            stride_cn,
            GROUP_M=8,
            BLOCK_K_CHOICE=block_k_choice,
            # NOTE: do NOT pass num_stages here. The default decorator is
            # triton.autotune(generate_configs="addmm"), which injects
            # num_stages from every generated Config -> duplicate keyword.
            # KLX_USE_AUTOTUNE=0 gets stages=3 from heur_stages instead.
        )
    if dest is not out:
        out.copy_(dest)
    return out


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    logger.debug("GEMS_KUNLUNXIN ADDMM_DTYPE")
    out = torch.empty(
        (mat1.shape[0], mat2.shape[1]), device=mat1.device, dtype=out_dtype
    )
    return addmm_dtype_out(bias, mat1, mat2, out_dtype, beta=beta, alpha=alpha, out=out)


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    logger.debug("GEMS_KUNLUNXIN ADDMM_DTYPE_OUT")
    if mat1.dtype != mat2.dtype:
        raise RuntimeError(
            f"mat1 and mat2 must have the same dtype, but got {mat1.dtype} and {mat2.dtype}"
        )
    if out.dtype != out_dtype:
        raise RuntimeError(
            "out_dtype must be the same as the provided out tensor dtype"
        )
    if not (
        out_dtype == mat1.dtype
        or (
            out_dtype == torch.float32 and mat1.dtype in (torch.float16, torch.bfloat16)
        )
    ):
        raise RuntimeError(
            "out_dtype must be the input dtype or fp32 for fp16/bf16 inputs"
        )
    if bias.dtype != out_dtype and bias.dtype != mat1.dtype:
        raise RuntimeError("self dtype must match either out_dtype or mat1 dtype")
    if mat1.shape[1] != mat2.shape[0]:
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied")
    if not broadcastable_to(bias.shape, (mat1.shape[0], mat2.shape[1])):
        raise RuntimeError("self is not broadcastable to the result shape")
    if out.shape != (mat1.shape[0], mat2.shape[1]):
        raise RuntimeError("out has an incompatible shape")

    M, K = mat1.shape
    _, N = mat2.shape
    mat1 = mat1.contiguous()
    bias = _bias_with_unit_inner_stride(bias, out.shape)
    block_k_choice = 256 if mat1.dtype == torch.float16 else 128
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    dest, stride_cm, stride_cn = _dest_with_unit_inner_stride(out, M, N)
    with torch_device_fn.device(mat1.device):
        addmm_kernel[grid](
            mat1,
            mat2,
            bias,
            dest,
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
            stride_cm,
            stride_cn,
            GROUP_M=8,
            BLOCK_K_CHOICE=block_k_choice,
            # NOTE: do NOT pass num_stages here. The default decorator is
            # triton.autotune(generate_configs="addmm"), which injects
            # num_stages from every generated Config -> duplicate keyword.
            # KLX_USE_AUTOTUNE=0 gets stages=3 from heur_stages instead.
        )
    if dest is not out:
        out.copy_(dest)
    return out
