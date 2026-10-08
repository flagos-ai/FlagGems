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
import math
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


@libentry()
@triton.jit
def _addmm_gather_2d(
    src,
    dst,
    rows,
    cols,
    src_stride0,
    src_stride1,
    BLOCK: tl.constexpr,
):
    """Gather a logical rank-2 view into contiguous storage."""
    pid = ext.program_id(axis=0)
    offsets = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    count = tl.cast(rows, tl.int64) * cols
    mask = offsets < count
    row = offsets // cols
    col = offsets - row * cols
    values = tl.load(
        src + row * src_stride0 + col * src_stride1,
        mask=mask,
        other=0,
    )
    tl.store(dst + offsets, values, mask=mask)


@libentry()
@triton.jit
def _addmm_scatter_2d(
    src,
    dst,
    rows,
    cols,
    dst_stride0,
    dst_stride1,
    BLOCK: tl.constexpr,
):
    """Scatter contiguous rank-2 storage into an arbitrary rank-2 view."""
    pid = ext.program_id(axis=0)
    offsets = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    count = tl.cast(rows, tl.int64) * cols
    mask = offsets < count
    row = offsets // cols
    col = offsets - row * cols
    values = tl.load(src + offsets, mask=mask, other=0)
    tl.store(
        dst + row * dst_stride0 + col * dst_stride1,
        values,
        mask=mask,
    )


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

    def heur_even(args):
        M = args["M"]
        N = args["N"]
        K = args["K"]
        return (
            M % heur_block_m(args) == 0
            and N % heur_block_n(args) == 0
            and K % heur_block_k(args) == 0
        )

    def heur_bias_1d(args):
        return args.get("stride_im", -1) == 0 and args.get("stride_in", -1) == 1

    autotune_decorator = triton.heuristics(
        {
            "BLOCK_SIZE_M": heur_block_m,
            "BLOCK_SIZE_N": heur_block_n,
            "BLOCK_SIZE_K": heur_block_k,
            "num_warps": heur_warps,
            "num_stages": heur_stages,
            "EVEN": heur_even,
            "BIAS_1D": heur_bias_1d,
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
    EVEN: tl.constexpr = False,
    BIAS_1D: tl.constexpr = False,
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
        if EVEN:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
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
    i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]

    if EVEN:
        if beta == 0:
            # Beta zero must ignore bias (including NaN/Inf), matching aten::addmm.
            accumulator = accumulator * alpha
        elif BIAS_1D:
            bias1d = tl.load(i_ptr + stride_in * offs_cn)
            accumulator = accumulator * alpha + bias1d[None, :] * beta
        else:
            bias = tl.load(i_ptrs)
            accumulator = accumulator * alpha + bias * beta
        tl.store(c_ptrs, accumulator)
    else:
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if beta == 0:
            accumulator = accumulator * alpha
        else:
            bias = tl.load(i_ptrs, mask=c_mask, other=0.0)
            accumulator = accumulator * alpha + bias * beta
        tl.store(c_ptrs, accumulator, mask=c_mask)


def _addmm_contiguous_2d(x):
    """Materialize a rank-2 logical view without native contiguous/copy ops."""
    rows, cols = x.shape
    out = torch.empty((rows, cols), dtype=x.dtype, device=x.device)
    if rows and cols:
        block = 1024
        grid = (triton.cdiv(rows * cols, block),)
        with torch_device_fn.device(x.device):
            _addmm_gather_2d[grid](
                x,
                out,
                rows,
                cols,
                x.stride(0),
                x.stride(1),
                BLOCK=block,
            )
    return out


def _bias_with_unit_inner_stride(bias, shape, beta):
    """Broadcast ``bias`` while avoiding native materialization operations."""
    b = bias.broadcast_to(shape)
    if beta == 0:
        return b
    if b.stride(1) != 1 and b.shape[1] > 1:
        # The kernel's 2-D load requires a unit inner stride.  Gather one
        # logical row when M broadcasts, then retain stride-zero broadcasting.
        if b.stride(0) == 0:
            row = _addmm_contiguous_2d(b[:1])
            b = row.broadcast_to(shape)
        else:
            b = _addmm_contiguous_2d(b)
    return b


def _check_addmm_output(out):
    rows, cols = out.shape
    if not rows or not cols:
        return
    s0, s1 = out.stride()
    overlap = (rows > 1 and s0 == 0) or (cols > 1 and s1 == 0)
    if rows > 1 and cols > 1 and s0 and s1:
        divisor = math.gcd(s0, s1)
        overlap = overlap or (s1 // divisor < rows and s0 // divisor < cols)
    if overlap:
        raise RuntimeError("addmm output has internally overlapping elements")


def _dest_with_unit_inner_stride(out, M, N, force_temp=False):
    """Use a temporary for strided stores or storage shared with an input."""
    stride_cm, stride_cn = out.stride()
    if not force_temp and (stride_cn == 1 or N <= 1):
        return out, stride_cm, stride_cn
    dest = torch.empty((M, N), device=out.device, dtype=out.dtype)
    return dest, dest.stride(0), dest.stride(1)


def _scatter_addmm_result(dest, out, M, N):
    if dest is not out and M and N:
        block = 1024
        grid = (triton.cdiv(M * N, block),)
        with torch_device_fn.device(out.device):
            _addmm_scatter_2d[grid](
                dest,
                out,
                M,
                N,
                out.stride(0),
                out.stride(1),
                BLOCK=block,
            )


def addmm(bias, mat1, mat2, *, beta=1.0, alpha=1.0):
    logger.debug("GEMS_KUNLUNXIN ADDMM")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    if not mat1.is_contiguous():
        mat1 = _addmm_contiguous_2d(mat1)
    # mat2 = mat2.contiguous()
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    bias = _bias_with_unit_inner_stride(bias, out.shape, beta)

    block_k_choice = 256 if mat1.dtype == torch.float16 else 128
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    dest, stride_cm, stride_cn = _dest_with_unit_inner_stride(
        out, M, N,
        force_temp=any(torch._C._is_alias_of(out, x) for x in (bias, mat1, mat2)),
    )
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
    _scatter_addmm_result(dest, out, M, N)
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
    _check_addmm_output(out)

    if not mat1.is_contiguous():
        mat1 = _addmm_contiguous_2d(mat1)
    bias = _bias_with_unit_inner_stride(bias, out.shape, beta)

    block_k_choice = 256 if mat1.dtype == torch.float16 else 128
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    dest, stride_cm, stride_cn = _dest_with_unit_inner_stride(
        out, M, N,
        force_temp=any(torch._C._is_alias_of(out, x) for x in (bias, mat1, mat2)),
    )
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
    _scatter_addmm_result(dest, out, M, N)
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
    _check_addmm_output(out)

    M, K = mat1.shape
    _, N = mat2.shape
    if not mat1.is_contiguous():
        mat1 = _addmm_contiguous_2d(mat1)
    bias = _bias_with_unit_inner_stride(bias, out.shape, beta)
    block_k_choice = 256 if mat1.dtype == torch.float16 else 128
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    dest, stride_cm, stride_cn = _dest_with_unit_inner_stride(
        out, M, N,
        force_temp=any(torch._C._is_alias_of(out, x) for x in (bias, mat1, mat2)),
    )
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
    _scatter_addmm_result(dest, out, M, N)
    return out
