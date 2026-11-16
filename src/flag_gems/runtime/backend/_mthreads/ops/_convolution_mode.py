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

from flag_gems.ops._convolution_mode import (
    _convolution_mode as default_convolution_mode,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _tf32_trunc(v):
    """Truncate the fp32 mantissa to 10 bits (tf32), matching the vendor conv."""
    b = v.to(tl.int32, bitcast=True)
    return (b & -8192).to(tl.float32, bitcast=True)


# This backend emits no async copies, so num_stages does not prefetch: the
# dot operands are staged through a single shared-memory buffer and deep
# pipelines only inflate the per-CTA footprint.  Latency hiding therefore has
# to come from CTA residency, so the config space trades tile size for a small
# shared-memory / register footprint and keeps num_stages shallow.
_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=1
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=2, num_stages=1
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2
    ),
]


@triton.autotune(configs=_CONFIGS, key=["S_OUT", "C_IN_G", "C_OUT_G", "W_OUT", "H_OUT"])
@triton.jit
def _conv_gemm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    C_IN,
    C_OUT,
    D_IN,
    H_IN,
    W_IN,
    H_OUT,
    W_OUT,
    S_OUT,
    C_OUT_G,
    GROUPS,
    SD,
    SH,
    SW,
    DD,
    DH,
    DW,
    PD,
    PH,
    PW,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    KV: tl.constexpr,
    C_IN_G: tl.constexpr,
    TF32_TRUNC: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_z = tl.program_id(2)

    nb = pid_z // GROUPS
    g = pid_z - nb * GROUPS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < C_OUT_G
    n_mask = offs_n < S_OUT

    w_idx = offs_n % W_OUT
    t = offs_n // W_OUT
    h_idx = t % H_OUT
    d_idx = t // H_OUT

    HW = H_IN * W_IN
    plane = D_IN * HW
    x0 = (
        x_ptr
        + (nb * C_IN + g * C_IN_G) * plane
        + (d_idx * SD - PD) * HW
        + (h_idx * SH - PH) * W_IN
        + (w_idx * SW - PW)
    )

    w_row = (g * C_OUT_G + offs_m) * (C_IN_G * KV)
    c_offs = tl.arange(0, BLOCK_K)
    c_off_plane = tl.minimum(c_offs, C_IN_G - 1) * plane

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    NCH: tl.constexpr = (C_IN_G + BLOCK_K - 1) // BLOCK_K
    NITER: tl.constexpr = KV * NCH

    for it in range(0, NITER):
        tap = it // NCH
        c0 = (it - tap * NCH) * BLOCK_K
        kd = tap // (KH * KW)
        r = tap - kd * (KH * KW)
        kh = r // KW
        kw = r - kh * KW

        in_d = d_idx * SD + kd * DD - PD
        in_h = h_idx * SH + kh * DH - PH
        in_w = w_idx * SW + kw * DW - PW
        m = (
            n_mask
            & (in_d >= 0)
            & (in_d < D_IN)
            & (in_h >= 0)
            & (in_h < H_IN)
            & (in_w >= 0)
            & (in_w < W_IN)
        )

        p = x0 + ((kd * DD) * HW + (kh * DH) * W_IN + kw * DW + c0 * plane)
        bv = tl.load(p[None, :] + c_off_plane[:, None], mask=m[None, :], other=0.0)
        av = tl.load(
            w_ptr + w_row[:, None] + (tap + (c0 + c_offs) * KV)[None, :],
            mask=m_mask[:, None] & ((c0 + c_offs) < C_IN_G)[None, :],
            other=0.0,
        )
        if TF32_TRUNC:
            acc += tl.dot(_tf32_trunc(av), _tf32_trunc(bv), input_precision="tf32")
        else:
            acc += tl.dot(av, bv)

    if HAS_BIAS:
        bias = tl.load(b_ptr + g * C_OUT_G + offs_m, mask=m_mask, other=0.0)
        acc += bias.to(tl.float32)[:, None]

    o_off = (nb * C_OUT + g * C_OUT_G + offs_m)[:, None] * S_OUT + offs_n[None, :]
    tl.store(
        o_ptr + o_off,
        acc.to(o_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _convolution_mode(input, weight, bias, stride, padding, dilation, groups):
    logger.debug("GEMS_MTHREADS _CONVOLUTION_MODE")

    if not _use_triton_kernel(input) or not _use_triton_kernel(weight):
        return default_convolution_mode(
            input, weight, bias, stride, padding, dilation, groups
        )

    with torch_device_fn.device(input.device):
        return _launch_convolution_mode(
            input, weight, bias, stride, padding, dilation, groups
        )


def _launch_convolution_mode(input, weight, bias, stride, padding, dilation, groups):
    nd = weight.dim() - 2

    if padding == "same":
        for st in stride:
            if st != 1:
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )
    elif padding != "valid":
        raise ValueError("Invalid padding string: {!r}".format(padding))

    if not isinstance(stride, (tuple, list)):
        stride = (stride,) * nd
    if not isinstance(dilation, (tuple, list)):
        dilation = (dilation,) * nd

    N = input.shape[0]
    C_in = input.shape[1]
    C_out = weight.shape[0]

    if nd == 1:
        sz = (1, 1, input.shape[2])
        kw = (1, 1, weight.shape[2])
        st = (1, 1, stride[0])
        dl = (1, 1, dilation[0])
    elif nd == 2:
        sz = (1, input.shape[2], input.shape[3])
        kw = (1, weight.shape[2], weight.shape[3])
        st = (1, stride[0], stride[1])
        dl = (1, dilation[0], dilation[1])
    else:
        sz = (input.shape[2], input.shape[3], input.shape[4])
        kw = (weight.shape[2], weight.shape[3], weight.shape[4])
        st = (stride[0], stride[1], stride[2])
        dl = (dilation[0], dilation[1], dilation[2])

    if padding == "same":
        t0 = dl[0] * (kw[0] - 1)
        t1 = dl[1] * (kw[1] - 1)
        t2 = dl[2] * (kw[2] - 1)
        pd = (t0 // 2, t1 // 2, t2 // 2)
        pr = (t0 - pd[0], t1 - pd[1], t2 - pd[2])
    else:
        pd = (0, 0, 0)
        pr = (0, 0, 0)

    D_out = (sz[0] + pd[0] + pr[0] - dl[0] * (kw[0] - 1) - 1) // st[0] + 1
    H_out = (sz[1] + pd[1] + pr[1] - dl[1] * (kw[1] - 1) - 1) // st[1] + 1
    W_out = (sz[2] + pd[2] + pr[2] - dl[2] * (kw[2] - 1) - 1) // st[2] + 1

    S_out = D_out * H_out * W_out
    C_in_g = C_in // groups
    C_out_g = C_out // groups
    kv = kw[0] * kw[1] * kw[2]

    if nd == 1:
        oshape = (N, C_out, W_out)
    elif nd == 2:
        oshape = (N, C_out, H_out, W_out)
    else:
        oshape = (N, C_out, D_out, H_out, W_out)

    out = torch.empty(oshape, device=input.device, dtype=input.dtype)

    grid = lambda META: (
        -(-S_out // META["BLOCK_N"]),
        -(-C_out_g // META["BLOCK_M"]),
        N * groups,
    )

    _conv_gemm_kernel[grid](
        input,
        weight,
        weight if bias is None else bias,
        out,
        C_in,
        C_out,
        sz[0],
        sz[1],
        sz[2],
        H_out,
        W_out,
        S_out,
        C_out_g,
        groups,
        st[0],
        st[1],
        st[2],
        dl[0],
        dl[1],
        dl[2],
        pd[0],
        pd[1],
        pd[2],
        kw[0],
        kw[1],
        kw[2],
        kv,
        C_in_g,
        input.dtype == torch.float32,
        bias is not None,
    )
    return out
