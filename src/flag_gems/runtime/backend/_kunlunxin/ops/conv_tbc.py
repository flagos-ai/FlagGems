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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# The reduction tile BK must stay <= 128 for fp32 (KERNEL_OPT_EXPERIENCE.md #19:
# fp32 BK=256 silently returns garbage). fp16/bf16 are upcast to fp32 before the
# kernel, so BK is a single fp32-safe value for all dtypes.
_BLOCK_K = 128
_NUM_STAGES = 3


def _pick_config(M, N):
    """Choose (BLOCK_M, BLOCK_N, num_warps) for a (M, N) output tile.

    Follows the addmm.py heuristics: small tiles (128, num_warps=4) for small
    problems, and large tiles (256, num_warps=8) for the sequence-model sized
    problems seen in the benchmark. conv_tbc M = Tout * B and N = Cout.
    """
    if M <= 512 and N <= 512:
        return 128, 128, 4
    block_m = 256 if M > 512 else 128
    block_n = 256 if N >= 256 else 128
    return block_m, block_n, 8


# NOTE: this kernel MUST NOT contain "conv" in its name. The Kunlunxin Triton
# `strip-all-ir` pass (StripAllOp.cpp) erases the body of every tt.func whose
# name contains "conv" (but not "transpose"), assuming its launch is redirected
# to a native handler. There is no native conv_tbc handler, so a "conv"-named
# kernel would be stripped to an empty shell and return garbage.
@triton.jit
def _tbc_padded_kernel(
    inp_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B,
    kW,
    Cin_pad,
    Cout_pad,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # out[m, n] = bias[n] + sum_k sum_ci inp_row(m, k)[ci] * weight[k, ci, n]
    # where m = to * B + b and inp_row(m, k) = m + k * B. The input tensor has
    # been pre-padded along the time axis (T_pad = T + 2*pad) and along Cin
    # (Cin_pad), so every load below is a fully-affine unmasked load. This is
    # required on the Kunlunxin backend: any masked 2D load inside a tl.dot
    # kernel silently miscompiles (KERNEL_OPT_EXPERIENCE.md entry #17).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k = Cin_pad // BLOCK_K
    for k in range(kW):
        in_row = offs_m + k * B
        w_base = k * Cin_pad * Cout_pad
        for ki in range(num_k):
            ck = ki * BLOCK_K + offs_k
            a = tl.load(inp_ptr + in_row[:, None] * Cin_pad + ck[None, :])
            w = tl.load(weight_ptr + w_base + ck[:, None] * Cout_pad + offs_n[None, :])
            acc += tl.dot(a, w, allow_tf32=False)

    bias = tl.load(bias_ptr + offs_n)
    acc += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * Cout_pad + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
    )


def _round_up(x, multiple):
    return (x + multiple - 1) // multiple * multiple


def _conv_tbc_impl(input, weight, bias, pad, out=None):
    assert input.dim() == 3, "Input must be 3-dimensional (Time, Batch, Channels)"
    assert weight.dim() == 3, "Weight must be 3-dimensional (kW, Cin, Cout)"

    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    T, B, Cin = input.shape
    kW, Cin_w, Cout = weight.shape
    assert Cin == Cin_w, "Input channels must match weight input channels"

    Tout = T + 2 * pad - kW + 1
    if Tout <= 0:
        if out is None:
            return torch.empty((0, B, Cout), dtype=input.dtype, device=input.device)
        return out

    M = Tout * B
    N = Cout

    block_m, block_n, num_warps = _pick_config(M, N)

    # The dot inputs must be fully unmasked, so pad Cin / Cout / M up to the
    # block boundaries and zero-fill the padding lanes. fp16/bf16 inputs are
    # upcast to fp32 before entering the kernel: loading bf16/fp16 inside a
    # tl.dot kernel miscompiles on this backend (KERNEL_OPT_EXPERIENCE.md #11).
    Cin_pad = _round_up(Cin, _BLOCK_K)
    Cout_pad = _round_up(Cout, block_n)
    M_pad = _round_up(M, block_m)
    T_pad = T + 2 * pad

    inp_pad = torch.zeros((T_pad, B, Cin_pad), dtype=torch.float32, device=input.device)
    inp_pad[pad : pad + T, :, :Cin] = input.float()
    w_pad = torch.zeros(
        (kW, Cin_pad, Cout_pad), dtype=torch.float32, device=weight.device
    )
    w_pad[:, :Cin, :Cout] = weight.float()
    bias_pad = torch.zeros((Cout_pad,), dtype=torch.float32, device=bias.device)
    bias_pad[:Cout] = bias.float()

    out_pad = torch.empty((M_pad, Cout_pad), dtype=torch.float32, device=input.device)

    grid = (M_pad // block_m, Cout_pad // block_n)
    with torch_device_fn.device(input.device):
        _tbc_padded_kernel[grid](
            inp_pad,
            w_pad,
            bias_pad,
            out_pad,
            B,
            kW,
            Cin_pad,
            Cout_pad,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=_BLOCK_K,
            num_warps=num_warps,
            num_stages=_NUM_STAGES,
        )

    result = out_pad[:M, :Cout].to(input.dtype)
    if out is None:
        return result.reshape(Tout, B, Cout)
    out.view(M, N).copy_(result)
    return out


def conv_tbc(input, weight, bias, pad=0):
    logger.debug("GEMS_KUNLUNXIN CONV_TBC")
    return _conv_tbc_impl(input, weight, bias, pad)


def conv_tbc_out(input, weight, bias, pad=0, *, out):
    logger.debug("GEMS_KUNLUNXIN CONV_TBC_OUT")
    return _conv_tbc_impl(input, weight, bias, pad, out=out)
