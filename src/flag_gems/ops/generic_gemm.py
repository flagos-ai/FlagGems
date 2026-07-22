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
import weakref

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_fp8_quant_cache = {}
_amax_buf_cache = {}
_FP8_MAX: float = 448.0
_FP8_QUANT_BLOCK: int = 1024


@triton.jit
def _amax_reduce_kernel(x_ptr, amax_out_ptr, N, BLOCK_N: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    local_max = tl.max(tl.abs(vals))
    tl.atomic_max(amax_out_ptr, local_max)


@triton.jit
def _fp8_quant_kernel(x_ptr, x_fp8_ptr, scale_ptr, N, BLOCK_N: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr)
    q = tl.clamp(vals / scale, -448.0, 448.0)
    tl.store(x_fp8_ptr + offsets, q.to(tl.float8e4nv), mask=mask)


def _cached_per_tensor_quantize(x: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    key = x.data_ptr()
    cached = _fp8_quant_cache.get(key)
    if cached is not None:
        x_fp8, scale = cached
        if x_fp8.shape == x.shape:
            return x_fp8, scale
    N = x.numel()
    grid = (triton.cdiv(N, _FP8_QUANT_BLOCK),)
    device_key = x.device.index
    amax_buf = _amax_buf_cache.get(device_key)
    if amax_buf is None:
        amax_buf = torch.zeros(1, dtype=torch.float32, device=x.device)
        _amax_buf_cache[device_key] = amax_buf
    else:
        amax_buf.fill_(0)
    _amax_reduce_kernel[grid](x, amax_buf, N, BLOCK_N=_FP8_QUANT_BLOCK)
    scale = amax_buf / _FP8_MAX
    x_fp8 = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    _fp8_quant_kernel[grid](x, x_fp8, scale, N, BLOCK_N=_FP8_QUANT_BLOCK)
    _fp8_quant_cache[key] = (x_fp8, scale)
    _set_cache_finalizer(key, x)
    return x_fp8, scale


def _cleanup_cache_entry(key):
    _fp8_quant_cache.pop(key, None)


def _set_cache_finalizer(key, src_tensor):
    try:
        _wref = weakref.ref(  # noqa: F841
            src_tensor, lambda _ref, k=key: _cleanup_cache_entry(k)
        )
    except TypeError:
        pass


MM_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=5, num_warps=4
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=5, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=5, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=5, num_warps=4
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 128}, num_stages=4, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 128}, num_stages=4, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=5, num_warps=4
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=5, num_warps=4
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=8
    ),
]

FP8_MM_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_stages=3, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128}, num_stages=4, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 128}, num_stages=4, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256}, num_stages=3, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 256}, num_stages=3, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256}, num_stages=3, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 256}, num_stages=3, num_warps=8
    ),
]


@triton.jit
def _prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@triton.jit
def _gelu(x):
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = 2.0 * inner
    sigmoid = tl.where(
        y >= 0,
        1.0 / (1.0 + tl.math.exp(-y)),
        tl.math.exp(y) / (1.0 + tl.math.exp(y)),
    )
    tanh_inner = 2.0 * sigmoid - 1.0
    return 0.5 * x * (1.0 + tanh_inner)


@triton.jit
def _dgelu(x):
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = 2.0 * inner
    sigmoid = tl.where(
        y >= 0,
        1.0 / (1.0 + tl.math.exp(-y)),
        tl.math.exp(y) / (1.0 + tl.math.exp(y)),
    )
    tanh_inner = 2.0 * sigmoid - 1.0
    sech2 = 1.0 - tanh_inner * tanh_inner
    inner_grad = 0.7978845608028654 * (1.0 + 0.134145 * x * x)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * inner_grad


@triton.autotune(
    configs=MM_CONFIGS + FP8_MM_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def mm_kernel_epilogue(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BIAS,
    PRE_GELU,
    stride_bias,
    stride_pg_m,
    stride_pg_n,
    ALPHA,
    BETA,
    SCALE_A,
    SCALE_B,
    HAS_BIAS: tl.constexpr,
    HAS_GELU: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    HAS_DGELU: tl.constexpr,
    HAS_FP8_INPUT: tl.constexpr,
    HAS_FP8_OUTPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
):
    pid = tl.program_id(0).to(tl.int64)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
    rm, rn = rm.to(tl.int64), rn.to(tl.int64)
    prev_multiple = _prev_multiple_of(K, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_multiple, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
        b = tl.load(B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn))
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    rk = (prev_multiple + tl.arange(0, BLOCK_K)).to(tl.int64)
    mask_k = rk < K
    a = tl.load(
        A + (ram[:, None] * stride_am + rk[None, :] * stride_ak),
        mask=mask_k[None, :],
        other=0.0,
    )
    b = tl.load(
        B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn),
        mask=mask_k[:, None],
        other=0.0,
    )
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    acc = acc * ALPHA

    if HAS_FP8_INPUT:
        acc = acc * tl.load(SCALE_A) * tl.load(SCALE_B)

    rm_out = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn_out = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    out_mask = (rm_out < M)[:, None] & (rn_out < N)[None, :]

    if ACCUMULATE:
        old = tl.load(
            C + (rm_out[:, None] * stride_cm + rn_out[None, :] * stride_cn),
            mask=out_mask,
            other=0.0,
        ).to(tl.float32)
        acc = acc + BETA * old

    if HAS_BIAS:
        b = tl.load(BIAS + rn_out * stride_bias, mask=rn_out < N, other=0.0).to(
            tl.float32
        )
        acc = acc + b[None, :]

    if HAS_DGELU:
        gelu_in = tl.load(
            PRE_GELU + (rm_out[:, None] * stride_pg_m + rn_out[None, :] * stride_pg_n),
            mask=out_mask,
            other=0.0,
        ).to(tl.float32)
        dgelu = _dgelu(gelu_in)
        acc = acc * dgelu

    if HAS_FP8_OUTPUT:
        FP8_MAX: tl.constexpr = 448.0
        acc_out = tl.clamp(acc, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
    else:
        acc_out = acc.to(C.dtype.element_ty)

    if HAS_GELU:
        pg_ptr = PRE_GELU + (
            rm_out[:, None] * stride_pg_m + rn_out[None, :] * stride_pg_n
        )
        tl.store(pg_ptr, acc_out, mask=out_mask)
        acc_out = _gelu(acc.to(tl.float32)).to(C.dtype.element_ty)

    C_ptr = C + (rm_out[:, None] * stride_cm + rn_out[None, :] * stride_cn)
    tl.store(C_ptr, acc_out, mask=out_mask)


def generic_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    layout: str = "NN",
    bias: torch.Tensor = None,
    bias_type: torch.dtype = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
    alpha: float = 1.0,
    beta: float = None,
    accumulate: bool = False,
    out: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    grad: bool = False,
    scale_a: torch.Tensor = None,
    scale_b: torch.Tensor = None,
    fp8_output: bool = False,
) -> tuple:
    logger.debug("GEMS GENERIC_GEMM")

    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"generic_gemm expects 2D inputs, got a{a.shape} b{b.shape}")

    assert len(layout) == 2 and all(c in "TN" for c in layout), f"bad layout: {layout}"
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    if transa:
        M, K_a = a.shape[1], a.shape[0]
    else:
        M, K_a = a.shape[0], a.shape[1]
    if transb:
        K_b, N = b.shape[1], b.shape[0]
    else:
        K_b, N = b.shape[0], b.shape[1]
    assert K_a == K_b, f"K mismatch: a={a.shape}, b={b.shape}, layout={layout}"
    K = K_a

    if transa:
        stride_am, stride_ak = 1, a.stride(0)
    else:
        stride_am, stride_ak = a.stride(0), a.stride(1)
    if transb:
        stride_bk, stride_bn = 1, b.stride(0)
    else:
        stride_bk, stride_bn = b.stride(0), b.stride(1)

    has_fp8_input = (scale_a is not None) and (scale_b is not None)
    if has_fp8_input:
        assert (
            scale_a.numel() == 1 and scale_a.dtype == torch.float32
        ), "scale_a must be a float32 scalar tensor"
        assert (
            scale_b.numel() == 1 and scale_b.dtype == torch.float32
        ), "scale_b must be a float32 scalar tensor"

    if bias_type is not None:
        if bias is not None:
            bias = bias.to(bias_type)
    elif has_fp8_input and bias is not None and bias.dtype != torch.bfloat16:
        bias = bias.to(torch.bfloat16)

    if bias_type is not None:
        _bias_dtype = bias_type
    elif bias is not None:
        _bias_dtype = bias.dtype
    elif has_fp8_input:
        _bias_dtype = torch.bfloat16
    else:
        _bias_dtype = a.dtype

    if out_dtype is not None:
        dtype = out_dtype
    elif fp8_output:
        dtype = torch.float8_e4m3fn
    elif has_fp8_input:
        dtype = torch.bfloat16
    else:
        dtype = a.dtype

    if out is not None:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")
        assert out.shape == (M, N), f"out shape {out.shape} != ({M}, {N})"
        c = out
    elif fp8_output:
        c = torch.empty((M, N), device=a.device, dtype=torch.float8_e4m3fn)
    else:
        c = torch.empty((M, N), device=a.device, dtype=dtype)

    if alpha == 0.0:
        raise ValueError("alpha must be non-zero for GEMM")

    if beta is not None:
        _beta = float(beta)
        _accumulate = _beta != 0.0
    else:
        _beta = 1.0
        _accumulate = accumulate

    if grad and gelu:
        assert gelu_in is not None, "gelu_in is required when grad=True and gelu=True"

    _compute_dbias = (bias is not None) and grad

    pre_gelu_out = None
    _has_dgelu = False
    if gelu and grad:
        _has_dgelu = True
        _gelu_in_ptr = gelu_in
        _gelu_in_stride_m = gelu_in.stride(0)
        _gelu_in_stride_n = gelu_in.stride(1)
    elif gelu and not grad:
        pre_gelu_dtype = (
            torch.float8_e4m3fn
            if fp8_output
            else (_bias_dtype if has_fp8_input else a.dtype)
        )
        pre_gelu_out = torch.empty((M, N), device=a.device, dtype=pre_gelu_dtype)
        _gelu_in_ptr = pre_gelu_out
        _gelu_in_stride_m = pre_gelu_out.stride(0)
        _gelu_in_stride_n = pre_gelu_out.stride(1)
    else:
        _gelu_in_ptr = c
        _gelu_in_stride_m = 0
        _gelu_in_stride_n = 0

    bias_ptr = bias if (bias is not None and not grad) else c
    bias_stride = bias.stride(0) if (bias is not None and not grad) else 0

    has_bias = (bias is not None) and (not grad)
    has_gelu = gelu and (not grad)

    _scale_a_ptr = scale_a if has_fp8_input else a
    _scale_b_ptr = scale_b if has_fp8_input else a

    def grid_fn(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    mm_kernel_epilogue[grid_fn](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        c.stride(0),
        c.stride(1),
        bias_ptr,
        _gelu_in_ptr,
        bias_stride,
        _gelu_in_stride_m,
        _gelu_in_stride_n,
        float(alpha),
        _beta,
        _scale_a_ptr,
        _scale_b_ptr,
        has_bias,
        has_gelu,
        _accumulate,
        _has_dgelu,
        has_fp8_input,
        fp8_output,
    )

    bias_grad_out = c.float().sum(dim=0) if _compute_dbias else None

    return c, bias_grad_out, pre_gelu_out, None
