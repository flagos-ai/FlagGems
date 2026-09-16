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

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def mm_w8a8_fp8_kernel(
    A,
    B,
    C,
    SA,
    SB,
    Bias,
    SR,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK_STRIDE: tl.constexpr,
    BN_STRIDE: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    DESCRIPTOR: tl.constexpr,
    SA_STRIDE: tl.constexpr,
    SB_STRIDE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    HAS_SR: tl.constexpr,
    SR_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    group = pid // (8 * grid_n)
    group_m = tl.minimum(8, grid_m - group * 8)
    pm = group * 8 + pid % group_m
    pn = pid % (8 * grid_n) // group_m
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    split = tl.program_id(1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for block in range(split, tl.cdiv(K, BLOCK_K), SPLIT_K):
        if DESCRIPTOR:
            a = tl.load_tensor_descriptor(A, [pm * BLOCK_M, block * BLOCK_K])
            bt = tl.load_tensor_descriptor(B, [pn * BLOCK_N, block * BLOCK_K])
        else:
            k = block * BLOCK_K + rk
            a = tl.load(
                A + rm[:, None].to(tl.int64) * AM + k[None, :].to(tl.int64) * AK,
                (rm[:, None] < M) & (k[None, :] < K),
                0.0,
            )
            bt = tl.load(
                B
                + rn[:, None].to(tl.int64) * BN_STRIDE
                + k[None, :].to(tl.int64) * BK_STRIDE,
                (rn[:, None] < N) & (k[None, :] < K),
                0.0,
            )
        acc = tl.dot(a, tl.trans(bt), acc)
    if SA_STRIDE == 0:
        acc *= tl.load(SA)
    else:
        acc *= tl.load(SA + rm * SA_STRIDE, rm < M, 0.0)[:, None]
    if SB_STRIDE == 0:
        acc *= tl.load(SB)
    else:
        acc *= tl.load(SB + rn * SB_STRIDE, rn < N, 0.0)[None, :]
    if SPLIT_K > 1:
        ptr = C + split * M * N + rm[:, None] * N + rn[None, :]
        tl.store(ptr, acc, (rm[:, None] < M) & (rn[None, :] < N))
    else:
        if HAS_BIAS:
            acc += tl.load(Bias + rn * BIAS_STRIDE, rn < N, 0.0)[None, :].to(tl.float32)
        if HAS_SR:
            if SR_STRIDE == 0:
                acc /= tl.load(SR)
            else:
                acc /= tl.load(SR + rm * SR_STRIDE, rm < M, 1.0)[:, None]
        ptr = C + rm[:, None].to(tl.int64) * CM + rn[None, :].to(tl.int64) * CN
        tl.store(ptr, acc, (rm[:, None] < M) & (rn[None, :] < N))


@libentry()
@triton.jit
def _reduce_split_k(
    P,
    C,
    Bias,
    SR,
    M: tl.constexpr,
    N: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    HAS_SR: tl.constexpr,
    SR_STRIDE: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(P + s * M * N + x, x < M * N, 0.0)
    if HAS_BIAS:
        acc += tl.load(Bias + (x % N) * BIAS_STRIDE, x < M * N, 0.0).to(tl.float32)
    if HAS_SR:
        if SR_STRIDE == 0:
            acc /= tl.load(SR)
        else:
            acc /= tl.load(SR + (x // N) * SR_STRIDE, x < M * N, 1.0)
    tl.store(C + (x // N).to(tl.int64) * CM + (x % N).to(tl.int64) * CN, acc, x < M * N)


def _select_config(m, n, k, descriptor):
    if k <= 128:
        return (
            min(64, max(16, triton.next_power_of_2(m))),
            64,
            max(32, triton.next_power_of_2(k)),
            1,
            1,
        )
    if k >= 2048 and m <= 128 and n <= 512:
        # Increase CTA count only for skinny grids. Partial sums stay in FP32.
        bm = 16 if m <= 64 else 32
        tiles = triton.cdiv(m, bm) * triton.cdiv(n, 64)
        split = min(
            32,
            triton.next_power_of_2(triton.cdiv(120, tiles)),
            triton.next_power_of_2(triton.cdiv(k, 256)),
        )
        return bm, 64, 256, 2, split
    if descriptor and k >= 8192 and m <= 512 and m * n <= 512 * 1024:
        # Bound workspace/reduction traffic while filling underoccupied TME
        # grids. Wider M tiles amortize loads when a full tile is available.
        bm = min(64 if n <= 512 else 128, max(32, triton.next_power_of_2(m)))
        bn = 64 if m <= 64 else 128
        bk = 256 if bm <= 64 else 128
        tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        target_ctas = 128 if m <= 64 else 64
        split = min(16, triton.next_power_of_2(triton.cdiv(target_ctas, tiles)))
        if split > 1:
            return bm, bn, bk, 2 if bk == 256 else 3, split
    if m <= 32:
        return max(16, triton.next_power_of_2(m)), 64, 128, 3, 1
    if descriptor and m >= 512 and n >= 1024:
        return 128, 128, 128, 2 if k <= 2048 else 3, 1
    return 64, 128, 256, 2, 1


def _launch(a, b, out, sa, sb, sa_stride, sb_stride, bias, sr, sr_stride):
    m, k = a.shape
    n = b.shape[1]
    # E5M2 descriptor loads fail S5000 accuracy checks. Masked FP8 loads also
    # cover unaligned and non-contiguous inputs without BF16 dequantization.
    descriptor = (
        a.dtype == b.dtype == torch.float8_e4m3fn
        and a.stride(1) == 1
        and b.stride(0) == 1
        and a.stride(0) > 0
        and b.stride(1) > 0
        and a.stride(0) % 16 == 0
        and b.stride(1) % 16 == 0
        and a.data_ptr() % 16 == 0
        and b.data_ptr() % 16 == 0
        and k >= 16
    )
    bm, bn, bk, stages, split = _select_config(m, n, k, descriptor)
    if descriptor:
        aa = TensorDescriptor(a, [m, k], list(a.stride()), [bm, bk])
        bb = TensorDescriptor(b, [n, k], [b.stride(1), b.stride(0)], [bn, bk])
    else:
        aa, bb = a, b
    partial = (
        torch.empty((split, m, n), device=a.device, dtype=torch.float32)
        if split > 1
        else out
    )
    epilogue = dict(
        HAS_BIAS=bias is not None,
        BIAS_STRIDE=bias.stride(0) if bias is not None else 0,
        HAS_SR=sr is not None,
        SR_STRIDE=sr_stride,
    )
    mm_w8a8_fp8_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn), split)](
        aa,
        bb,
        partial,
        sa,
        sb,
        bias,
        sr,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        SPLIT_K=split,
        DESCRIPTOR=descriptor,
        SA_STRIDE=sa_stride,
        SB_STRIDE=sb_stride,
        **epilogue,
        num_warps=4,
        num_stages=stages,
    )
    if split > 1:
        _reduce_split_k[(triton.cdiv(m * n, 512),)](
            partial,
            out,
            bias,
            sr,
            m,
            n,
            out.stride(0),
            out.stride(1),
            SPLIT_K=split,
            BLOCK=512,
            **epilogue,
            num_warps=4,
        )
    return out


def _scale_stride(scale, x, axis, name):
    if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32:
        raise TypeError(f"{name} must be a float32 tensor")
    if scale.device != x.device:
        raise ValueError(f"{name} must be on the same device as the inputs")
    if scale.numel() == 1 and scale.ndim <= 2:
        return 0
    size = x.shape[axis]
    shape = (size, 1) if axis == 0 else (1, size)
    if scale.shape == (size,):
        return scale.stride(0)
    if scale.shape == shape:
        return scale.stride(axis)
    raise ValueError(f"{name} must be a scalar, ({size},), or {shape}")


def mm_w8a8_fp8(
    input,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
    *,
    out=None,
):
    """FP8 matmul with the torch._scaled_mm interface on MThreads.

    scale_a is scalar, (M,), or (M, 1); scale_b is scalar, (N,), or (1, N).
    Scales must be float32 tensors on the input device. Following torch_musa,
    scale_result divides the scaled product plus bias, for all output dtypes;
    it can be scalar, (M,), or (M, 1). Output defaults to input.dtype.
    Both use_fast_accum settings use FP32 accumulation, as in torch_musa.
    BF16/FP16 inputs and K-block scales are not supported.
    """
    a, b = input, mat2
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("mm_w8a8_fp8 expects two-dimensional inputs")
    if out is None:
        dtype = a.dtype if out_dtype is None else out_dtype
        out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=dtype)
    return mm_w8a8_fp8_out(
        a, b, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out=out
    )


def mm_w8a8_fp8_out(
    input,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
    *,
    out,
):
    """torch._scaled_mm.out-compatible variant with a reusable output."""
    a, b = input, mat2
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("mm_w8a8_fp8 expects compatible two-dimensional inputs")
    if a.device != b.device or a.device != out.device:
        raise ValueError("inputs and out must be on the same device")
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    if a.dtype not in fp8_dtypes or b.dtype not in fp8_dtypes:
        raise TypeError("mm_w8a8_fp8 requires FP8 inputs")
    if out.dtype not in (*fp8_dtypes, torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("unsupported output dtype")
    if out_dtype is not None and out_dtype != out.dtype:
        raise ValueError("out_dtype must match out.dtype")
    if not isinstance(use_fast_accum, bool):
        raise TypeError("use_fast_accum must be a bool")
    sa_stride = _scale_stride(scale_a, a, 0, "scale_a")
    sb_stride = _scale_stride(scale_b, b, 1, "scale_b")
    sr_stride = (
        _scale_stride(scale_result, a, 0, "scale_result")
        if scale_result is not None
        else 0
    )
    if bias is not None:
        bias_dtypes = (
            (torch.float32,)
            if out.dtype == torch.float32
            else (torch.float16, torch.bfloat16)
        )
        if not isinstance(bias, torch.Tensor) or bias.dtype not in bias_dtypes:
            raise TypeError("bias has an unsupported dtype for the output")
        if bias.device != a.device or bias.numel() != b.shape[1]:
            raise ValueError("bias must contain N elements on the input device")
        bias = bias.reshape(-1)
    if tuple(out.shape) != (a.shape[0], b.shape[1]):
        out.resize_(a.shape[0], b.shape[1])
    if out.numel() == 0:
        return out
    with torch_device_fn.device(a.device):
        return _launch(
            a,
            b,
            out,
            scale_a,
            scale_b,
            sa_stride,
            sb_stride,
            bias,
            scale_result,
            sr_stride,
        )
