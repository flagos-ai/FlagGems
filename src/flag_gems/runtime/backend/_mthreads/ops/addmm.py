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
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

# Reuse the warp-specialized mm building blocks.  addmm is a single GEMM with
# a bias epilogue, so it shares the tuned 256x256x64 tile pipeline, the
# column-panel rasterization and the TLE producer with mm; only the consumer
# gains the extra beta * bias term before the store.
from .mm import (  # noqa: E402
    _WS_BLOCK_K,
    _WS_BLOCK_M,
    _WS_BLOCK_N,
    _WS_NUM_WARPS,
    _WS_PANEL_WIDTH,
    _WS_STAGES,
    TLE_WS_ON,
    _ws_mm_producer,
    _ws_rasterization_2d_column,
)

try:
    import triton.experimental.tle.language as tle
except ImportError:  # pragma: no cover - depends on the installed Triton build
    tle = None

logger = logging.getLogger(__name__)

# A full 2-D bias tile needs one block_m x block_n epilogue staging buffer in
# shared memory, which does not fit next to mm's 3-stage 256x256x64 TME ring on
# a 192 KiB part.  This narrower, deeper-K ring measured fastest while keeping
# the epilogue fused; broadcast biases still use the mm ring unchanged.
_WS_BIAS_BLOCK_M = 256
_WS_BIAS_BLOCK_N = 128
_WS_BIAS_BLOCK_K = 32
_WS_BIAS_STAGES = 5


EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "addmm_mthreads_expand.yaml")
)


def is_supported_sqmma_layout(tensor):
    return tensor.is_contiguous() or (
        tensor.stride(0) == 1 and tensor.stride(1) == tensor.shape[0]
    )


def is_sqmma_compatible(a, b, N, K):
    return (
        a.dim() == 2
        and b.dim() == 2
        and a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and is_supported_sqmma_layout(a)
        and is_supported_sqmma_layout(b)
        and a.shape[0] > 0
        and N > 0
        and K > 0
        and N % 8 == 0
        and K % 8 == 0
    )


def _prepare_bias(bias, out):
    # Keep vector/scalar bias compact; broadcast strides cover other valid shapes.
    bias_is_vector = bias.ndim == 1 and bias.shape[0] == out.shape[1]
    bias_is_scalar = not bias_is_vector and bias.numel() == 1
    if bias_is_vector:
        return bias, 0, bias.stride(0), True, False
    if bias_is_scalar:
        return bias, 0, 0, False, True
    bias = bias.broadcast_to(out.shape)
    return bias, bias.stride(0), bias.stride(1), False, False


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 16},
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 16},
            num_stages=1,
            num_warps=16,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=1,
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
    warmup=5,
    rep=5,
)
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
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    if IS_FP64:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N),
            other=0.0,
        )
        if IS_FP64:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if BIAS_IS_VECTOR:
        bias = tl.load(
            i_ptr + stride_in * offs_cn,
            mask=offs_cn < N,
            other=0.0,
        )[None, :]
    elif BIAS_IS_SCALAR:
        bias = tl.load(i_ptr)
    else:
        i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
        bias = tl.load(i_ptrs, mask=c_mask, other=0.0)

    accumulator = accumulator * alpha + bias * beta
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


def addmm_fma(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_MTHREADS ADDMM_FMA")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    if mat1.stride(0) > 1 and mat1.stride(1) > 1:
        mat1 = mat1.contiguous()
    mat2_k_contiguous = mat2.stride(0) == 1 and mat2.stride(1) > 1
    # Direct K-contiguous loads win for small M. With more than eight 128-row
    # tiles, a coalesced copy is amortized across enough reuse of the B matrix.
    if (mat2_k_contiguous and M > 8 * 128) or (
        mat2.stride(0) > 1 and mat2.stride(1) > 1
    ):
        mat2 = mat2.contiguous()
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    bias_is_vector = bias.ndim == 1 and bias.shape[0] == N
    bias_is_scalar = not bias_is_vector and bias.numel() == 1
    if bias_is_vector:
        bias_stride_m = 0
        bias_stride_n = bias.stride(0)
    elif bias_is_scalar:
        bias_stride_m = 0
        bias_stride_n = 0
    else:
        bias = bias.broadcast_to(out.shape).contiguous()
        bias_stride_m = bias.stride(0)
        bias_stride_n = bias.stride(1)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(mat1.device):
        addmm_kernel[grid](
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
            bias_stride_m,
            bias_stride_n,
            out.stride(0),
            out.stride(1),
            BIAS_IS_VECTOR=bias_is_vector,
            BIAS_IS_SCALAR=bias_is_scalar,
            IS_FP64=mat1.dtype == torch.float64,
        )
    return out


def addmm_sqmma_descriptor_pre_hook(nargs):
    nargs["a_desc"].block_shape = [nargs["BLOCK_SIZE_M"], nargs["BLOCK_SIZE_K"]]
    nargs["b_desc"].block_shape = [nargs["BLOCK_SIZE_K"], nargs["BLOCK_SIZE_N"]]
    nargs["c_desc"].block_shape = [nargs["BLOCK_SIZE_M"], nargs["BLOCK_SIZE_N"]]


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
            pre_hook=addmm_sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=4,
            pre_hook=addmm_sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_stages=1,
            num_warps=4,
            pre_hook=addmm_sqmma_descriptor_pre_hook,
        ),
    ],
    key=["M", "N", "K"],
    strategy=["default", "default", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="addmm",
    flagtune_expand_op_name="addmm_sqmma",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_pre_hook=addmm_sqmma_descriptor_pre_hook,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_sqmma_kernel(
    a_desc,
    b_desc,
    bias_ptr,
    c_desc,
    M,
    N,
    K,
    alpha,
    beta,
    stride_im,
    stride_in,
    DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_SIZE_M).to(tl.int32)
    offs_bn = (pid_n * BLOCK_SIZE_N).to(tl.int32)
    offs_k = 0
    offs_k = offs_k.to(tl.int32)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load_tensor_descriptor(a_desc, [offs_am, offs_k])
        b = tl.load_tensor_descriptor(b_desc, [offs_k, offs_bn])
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if BIAS_IS_VECTOR:
        bias = tl.load(
            bias_ptr + offs_n * stride_in,
            mask=offs_n < N,
            other=0.0,
        )[None, :]
    elif BIAS_IS_SCALAR:
        bias = tl.load(bias_ptr)
    else:
        bias_ptrs = bias_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
        bias = tl.load(bias_ptrs, mask=mask, other=0.0)
    result = (alpha * accumulator + beta * bias).to(c_desc.dtype)
    tl.store_tensor_descriptor(c_desc, [offs_am, offs_bn], result)


def addmm_sqmma(mat1, mat2, bias, elem_type, alpha, beta, M, N, K, out=None):
    logger.debug("GEMS_MTHREADS ADDMM_SQMMA")
    device = mat1.device
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    if not mat1.is_contiguous():
        mat1 = mat1.contiguous()
    if not mat2.is_contiguous():
        mat2 = mat2.contiguous()
    a_type = mat1.dtype
    b_type = mat2.dtype
    assert a_type == b_type, "Mat A and Mat B should have the same dtype"
    c_type = a_type
    if out is None:
        out = torch.empty((M, N), dtype=c_type, device=device)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    bias, stride_im, stride_in, bias_is_vector, bias_is_scalar = _prepare_bias(
        bias, out
    )
    desc_a = TensorDescriptor.from_tensor(mat1, [1, 1])
    desc_b = TensorDescriptor.from_tensor(mat2, [1, 1])
    desc_c = TensorDescriptor.from_tensor(out, [1, 1])
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        1,
        1,
    )
    addmm_sqmma_kernel[grid](
        desc_a,
        desc_b,
        bias,
        desc_c,
        M,
        N,
        K,
        alpha,
        beta,
        stride_im,
        stride_in,
        str(a_type).split(".")[-1],
        BIAS_IS_VECTOR=bias_is_vector,
        BIAS_IS_SCALAR=bias_is_scalar,
    )
    return out


# ---------------------------------------------------------------------------
# Warp-specialized SQMMA for addmm.  A producer warp group drives TME copies
# into a pipelined shared-memory ring while a consumer warp group runs SQMMA,
# so the next K tile is already in flight during the MMA.  The alpha/beta bias
# epilogue is folded into the consumer, so the extra bias pass costs no second
# launch.  This is what closes most of the gap to the vendor addmm.
# ---------------------------------------------------------------------------


@triton.jit
def _ws_addmm_consumer(
    a_reader,
    b_reader,
    c_ptr,
    bias_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    alpha,
    beta,
    stride_im,
    stride_in,
    pid_m,
    pid_n,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
):
    offset_m = pid_m * block_m + tl.arange(0, block_m)
    offset_n = pid_n * block_n + tl.arange(0, block_n)
    mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    k_tiles: tl.constexpr = tl.cdiv(K, block_k)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_iter in tl.range(
        0,
        k_tiles,
        num_stages=1,
        loop_unroll_factor=k_tiles,
    ):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        acc = tle.gpu.wgmma(a_wait.slot.a, b_wait.slot.b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    if BIAS_IS_VECTOR:
        bias = tl.load(
            bias_ptr + offset_n * stride_in,
            mask=offset_n < N,
            other=0.0,
        )[None, :]
    elif BIAS_IS_SCALAR:
        bias = tl.load(bias_ptr)
    else:
        bias_ptrs = (
            bias_ptr + offset_m[:, None] * stride_im + offset_n[None, :] * stride_in
        )
        bias = tl.load(bias_ptrs, mask=mask, other=0.0)

    result = (alpha * acc + beta * bias).to(c_ptr.dtype.element_ty)
    c_block_ptr = c_ptr + N * offset_m[:, None] + offset_n[None, :]
    tl.store(c_block_ptr, result, mask=mask)


@triton.jit
def addmm_sqmma_ws_kernel(
    a_desc,
    b_desc,
    c_ptr,
    bias_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    alpha,
    beta,
    stride_im,
    stride_in,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    panel_width: tl.constexpr,
    pipeline_stages: tl.constexpr,
    full_panel: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
):
    raw_bx = tl.program_id(axis=0)
    raw_by = tl.program_id(axis=1)
    grid_x = tl.num_programs(axis=0)
    grid_y = tl.num_programs(axis=1)
    block_idx = raw_bx + raw_by * grid_x
    pid_n, pid_m = _ws_rasterization_2d_column(
        block_idx,
        grid_x,
        grid_y,
        panel_width,
        full_panel,
    )

    a_smem = tle.gpu.alloc(
        (pipeline_stages, block_m, block_k),
        dtype=c_ptr.dtype.element_ty,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (pipeline_stages, block_k, block_n),
        dtype=c_ptr.dtype.element_ty,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    a_pipe = tle.pipe(capacity=pipeline_stages, scope="cta", name="addmm_a", a=a_smem)
    b_pipe = tle.pipe(capacity=pipeline_stages, scope="cta", name="addmm_b", b=b_smem)

    tle.gpu.warp_specialize(
        [
            (
                _ws_addmm_consumer,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    c_ptr,
                    bias_ptr,
                    M,
                    N,
                    K,
                    alpha,
                    beta,
                    stride_im,
                    stride_in,
                    pid_m,
                    pid_n,
                    block_m,
                    block_n,
                    block_k,
                    BIAS_IS_VECTOR,
                    BIAS_IS_SCALAR,
                ),
            ),
            (
                _ws_mm_producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    K,
                    pid_m,
                    pid_n,
                    block_m,
                    block_n,
                    block_k,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


def is_addmm_ws_compatible(mat1, mat2, out_dtype, M, N, K):
    """Warp-specialized SQMMA for addmm needs the TLE frontend, half-tile
    shapes and a row-major fp16/bf16 output that matches the operand dtype."""
    return (
        TLE_WS_ON
        and out_dtype in (torch.float16, torch.bfloat16)
        and M >= _WS_BLOCK_M // 2
        and N >= _WS_BLOCK_N // 2
        and is_sqmma_compatible(mat1, mat2, N, K)
    )


def addmm_sqmma_ws(mat1, mat2, bias, alpha, beta, M, N, K, out=None):
    logger.debug("GEMS_MTHREADS ADDMM_SQMMA_WS")
    device = mat1.device
    assert broadcastable_to(bias.shape, (M, N)), "Incompatible input shape"
    if not mat1.is_contiguous():
        mat1 = mat1.contiguous()
    if not mat2.is_contiguous():
        mat2 = mat2.contiguous()
    assert mat1.dtype == mat2.dtype, "Mat A and Mat B should have the same dtype"
    if out is None:
        out = torch.empty((M, N), dtype=mat1.dtype, device=device)
    bias, stride_im, stride_in, bias_is_vector, bias_is_scalar = _prepare_bias(
        bias, out
    )
    # A full 2-D bias tile forces an epilogue layout conversion that needs one
    # block_m x block_n staging buffer on top of the TME ring.  mm's 3-stage
    # 256x256x64 ring already fills the 192 KiB part, so a broadcast bias keeps
    # that tile while a full bias switches to a narrower, deeper-K ring that
    # leaves room for the epilogue.
    if bias_is_vector or bias_is_scalar:
        block_m, block_n, block_k, stages = (
            _WS_BLOCK_M,
            _WS_BLOCK_N,
            _WS_BLOCK_K,
            _WS_STAGES,
        )
    else:
        block_m, block_n, block_k, stages = (
            _WS_BIAS_BLOCK_M,
            _WS_BIAS_BLOCK_N,
            _WS_BIAS_BLOCK_K,
            _WS_BIAS_STAGES,
        )
    # The SQMMA consumer stores straight into a row-major buffer of the operand
    # dtype, so `out` must already match the operands; callers gate on that.
    desc_a = TensorDescriptor.from_tensor(mat1, [block_m, block_k])
    desc_b = TensorDescriptor.from_tensor(mat2, [block_k, block_n])
    grid = (triton.cdiv(N, block_n), triton.cdiv(M, block_m), 1)
    with torch_device_fn.device(mat1.device):
        addmm_sqmma_ws_kernel[grid](
            desc_a,
            desc_b,
            out,
            bias,
            M,
            N,
            K,
            alpha,
            beta,
            stride_im,
            stride_in,
            block_m,
            block_n,
            block_k,
            _WS_PANEL_WIDTH,
            stages,
            grid[0] % _WS_PANEL_WIDTH == 0,
            bias_is_vector,
            bias_is_scalar,
            num_warps=_WS_NUM_WARPS,
            num_stages=stages,
        )
    return out


def _addmm_impl(bias, mat1, mat2, out, beta, alpha):
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    a_dtype = mat1.dtype
    M, K = mat1.shape
    _, N = mat2.shape
    if out is not None:
        assert out.shape == (M, N), "Incompatible output shape"

    # The warp-specialized SQMMA path stores straight into a row-major buffer
    # of the operand dtype, so it needs a contiguous fp16/bf16 `out` (or no
    # `out` at all) before it can be used.
    if (
        bias.dtype == a_dtype
        and (out is None or (out.is_contiguous() and out.dtype == a_dtype))
        and is_addmm_ws_compatible(mat1, mat2, a_dtype, M, N, K)
    ):
        return addmm_sqmma_ws(
            mat1,
            mat2,
            bias,
            alpha,
            beta,
            M,
            N,
            K,
            out=out,
        )
    if (
        is_sqmma_compatible(mat1, mat2, N, K)
        and bias.dtype == a_dtype
        and (out is None or out.is_contiguous())
    ):
        return addmm_sqmma(
            mat1,
            mat2,
            bias,
            a_dtype,
            alpha,
            beta,
            M,
            N,
            K,
            out=out,
        )
    return addmm_fma(bias, mat1, mat2, alpha=alpha, beta=beta, out=out)


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_MTHREADS ADDMM")
    return _addmm_impl(bias, mat1, mat2, None, beta, alpha)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_MTHREADS ADDMM_OUT")
    return _addmm_impl(bias, mat1, mat2, out, beta, alpha)


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    logger.debug("GEMS_MTHREADS ADDMM_DTYPE")
    out = torch.empty(
        (mat1.shape[0], mat2.shape[1]),
        device=mat1.device,
        dtype=out_dtype,
    )
    return addmm_dtype_out(bias, mat1, mat2, out_dtype, beta=beta, alpha=alpha, out=out)


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    logger.debug("GEMS_MTHREADS ADDMM_DTYPE_OUT")
    if mat1.dtype != mat2.dtype:
        raise RuntimeError(
            f"mat1 and mat2 must have the same dtype, but got {mat1.dtype} and {mat2.dtype}"
        )
    if out.dtype != out_dtype:
        raise RuntimeError(
            "out_dtype must be the same as the dtype of the provided out tensor"
        )
    if not (
        out_dtype == mat1.dtype
        or (
            out_dtype == torch.float32 and mat1.dtype in (torch.float16, torch.bfloat16)
        )
    ):
        raise RuntimeError(
            "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs"
        )
    if bias.dtype != out_dtype and bias.dtype != mat1.dtype:
        raise RuntimeError("self dtype must match either out_dtype or mat1 dtype")

    bias_c = bias.to(out_dtype)
    M, K = mat1.shape
    _, N = mat2.shape
    a_dtype = mat1.dtype

    # Keep dtype promotion on FMA so FP32 output has no low-precision intermediate.
    if (
        out_dtype == mat1.dtype
        and out.is_contiguous()
        and is_sqmma_compatible(mat1, mat2, N, K)
    ):
        return addmm_sqmma(
            mat1,
            mat2,
            bias_c,
            a_dtype,
            alpha,
            beta,
            M,
            N,
            K,
            out=out,
        )
    return addmm_fma(bias_c, mat1, mat2, alpha=alpha, beta=beta, out=out)
