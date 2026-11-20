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

"""Triton implementation of aten::conv_tbc (time-batch-channel convolution).

out[t, b, oc] = bias[oc] + sum_{k, ic} self[t + k - pad, b, ic] * weight[k, ic, oc]
Tout = T + 2 * pad - K + 1

Fast path: packed (fully contiguous) operands -> kernel with 10 arguments and
strides derived inside the kernel.  Fallbacks: strided dot kernel for
non-contiguous operands, FMA kernel for tiny channels / float64.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_tbc_dot_packed(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    T,
    B,
    Cin,
    Cout,
    K,
    pad,
    BLOCK_T: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    CI_FULL: tl.constexpr,
):
    Tout = T + 2 * pad - K + 1

    pid_t = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci = tl.arange(0, BLOCK_CI)

    mask_t = offs_t < Tout
    mask_co = offs_co < Cout

    acc = tl.zeros((BLOCK_T, BLOCK_CO), dtype=tl.float32)

    n_ci = (Cin + BLOCK_CI - 1) // BLOCK_CI
    for it in range(K * n_ci):
        k = it // n_ci
        ci0 = (it % n_ci) * BLOCK_CI
        t_in = offs_t + (k - pad)
        mask_in = mask_t & (t_in >= 0) & (t_in < T)
        a_base = x_ptr + t_in * (B * Cin) + pid_b * Cin + ci0
        w_base = w_ptr + (k * Cin + ci0) * Cout
        if CI_FULL:
            a = tl.load(
                a_base[:, None] + offs_ci[None, :],
                mask=mask_in[:, None],
                other=0.0,
            )
            w = tl.load(
                w_base + offs_ci[:, None] * Cout + offs_co[None, :],
                mask=mask_co[None, :],
                other=0.0,
            )
        else:
            m_ci = (ci0 + offs_ci) < Cin
            a = tl.load(
                a_base[:, None] + offs_ci[None, :],
                mask=mask_in[:, None] & m_ci[None, :],
                other=0.0,
            )
            w = tl.load(
                w_base + offs_ci[:, None] * Cout + offs_co[None, :],
                mask=m_ci[:, None] & mask_co[None, :],
                other=0.0,
            )
        if x_ptr.dtype.element_ty == tl.float32:
            acc = tl.dot(a, w, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, w, acc)

    bias = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    acc += bias[None, :]

    o_base = o_ptr + offs_t[:, None] * (B * Cout) + pid_b * Cout + offs_co[None, :]
    tl.store(
        o_base,
        acc.to(o_ptr.dtype.element_ty),
        mask=mask_t[:, None] & mask_co[None, :],
    )


@triton.jit
def _conv_tbc_dot_strided(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    T,
    B,
    Cin,
    Cout,
    K,
    pad,
    Tout,
    xsT,
    xsB,
    xsC,
    wsK,
    wsI,
    wsO,
    bstr,
    osT,
    osB,
    osC,
    BLOCK_T: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci = tl.arange(0, BLOCK_CI)

    mask_t = offs_t < Tout
    mask_co = offs_co < Cout

    acc = tl.zeros((BLOCK_T, BLOCK_CO), dtype=tl.float32)

    for k in range(K):
        t_in = offs_t + (k - pad)
        mask_in = mask_t & (t_in >= 0) & (t_in < T)
        a_base = x_ptr + t_in * xsT + pid_b * xsB
        w_base = w_ptr + k * wsK
        for ci0 in range(0, Cin, BLOCK_CI):
            ci = ci0 + offs_ci
            m_ci = ci < Cin
            a = tl.load(
                a_base[:, None] + ci[None, :] * xsC,
                mask=mask_in[:, None] & m_ci[None, :],
                other=0.0,
            )
            w = tl.load(
                w_base + ci[:, None] * wsI + offs_co[None, :] * wsO,
                mask=m_ci[:, None] & mask_co[None, :],
                other=0.0,
            )
            if x_ptr.dtype.element_ty == tl.float32:
                acc = tl.dot(a, w, acc, input_precision="ieee")
            else:
                acc = tl.dot(a, w, acc)

    bias = tl.load(b_ptr + offs_co * bstr, mask=mask_co, other=0.0).to(tl.float32)
    acc += bias[None, :]

    o_base = o_ptr + pid_b * osB + offs_t[:, None] * osT + offs_co[None, :] * osC
    tl.store(
        o_base,
        acc.to(o_ptr.dtype.element_ty),
        mask=mask_t[:, None] & mask_co[None, :],
    )


@triton.jit
def _conv_tbc_fma_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    T,
    B,
    Cin,
    Cout,
    K,
    pad,
    Tout,
    xsT,
    xsB,
    xsC,
    wsK,
    wsI,
    wsO,
    bstr,
    osT,
    osB,
    osC,
    BLOCK_T: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    mask_t = offs_t < Tout
    mask_co = offs_co < Cout

    if x_ptr.dtype.element_ty == tl.float64:
        acc = tl.zeros((BLOCK_T, BLOCK_CO), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_T, BLOCK_CO), dtype=tl.float32)

    for k in range(K):
        t_in = offs_t + (k - pad)
        mask_in = mask_t & (t_in >= 0) & (t_in < T)
        a_row = x_ptr + t_in * xsT + pid_b * xsB
        w_row = w_ptr + k * wsK + offs_co * wsO
        for ci in range(Cin):
            a = tl.load(a_row + ci * xsC, mask=mask_in, other=0.0)
            wv = tl.load(w_row + ci * wsI, mask=mask_co, other=0.0)
            acc += a[:, None] * wv[None, :]

    bias = tl.load(b_ptr + offs_co * bstr, mask=mask_co, other=0.0)
    acc += bias[None, :].to(acc.dtype)

    o_base = o_ptr + pid_b * osB + offs_t[:, None] * osT + offs_co[None, :] * osC
    tl.store(
        o_base,
        acc.to(o_ptr.dtype.element_ty),
        mask=mask_t[:, None] & mask_co[None, :],
    )


def _next_pow2(n):
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


_NUM_SMS = 0
_CFG_CACHE = {}


def _num_sms():
    global _NUM_SMS
    if _NUM_SMS == 0:
        try:
            _NUM_SMS = int(
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).multi_processor_count
            )
        except Exception:
            _NUM_SMS = 1
        if _NUM_SMS <= 0:
            _NUM_SMS = 1
    return _NUM_SMS


def _pick_config(dtype, Tout, Cout, Cin, B):
    """Static per-(dtype, shape) tile choice derived from measured winners.

    fp32: 8 warps with BLOCK_CI=32 (registers were the cap at 64x64x64/4 warps).
    fp16/bf16: 4 warps with BLOCK_CI=32 (faster in the round-2/3 eval pair).
    BLOCK_T is halved (down to 16) while the grid stays under half the SM count
    so small batches keep enough CTAs in flight.
    """
    key = (dtype, Tout, Cout, Cin, B)
    cfg = _CFG_CACHE.get(key)
    if cfg is not None:
        return cfg

    block_t = min(64, max(16, _next_pow2(Tout)))
    block_co = min(64, max(16, _next_pow2(Cout)))
    warps = 8 if dtype is torch.float32 else 4
    block_ci = 32

    n_co = (Cout + block_co - 1) // block_co
    half_sms = max(1, _num_sms() // 2)
    while block_t > 16 and ((Tout + block_t - 1) // block_t) * n_co * B < half_sms:
        block_t //= 2

    cfg = (block_t, block_co, block_ci, warps)
    _CFG_CACHE[key] = cfg
    return cfg


def conv_tbc(input, weight, bias, pad=0, out=None, **kwargs):
    # Tolerate the out-variant forms:
    #   run(x, w, b, out_tensor)              -> out passed positionally
    #   run(x, w, b, pad=0, out=out_tensor)   -> keyword out
    if out is None:
        if isinstance(pad, torch.Tensor):
            out = pad
            pad = 0
        else:
            out = kwargs.get("out")

    T, B, Cin = input.shape
    K, Cin_w, Cout = weight.shape
    assert Cin == Cin_w, "Input channels must match weight input channels"

    Tout = T + 2 * pad - K + 1

    if out is None:
        out = torch.empty((Tout, B, Cout), dtype=input.dtype, device=input.device)

    if Tout <= 0 or B == 0 or Cout == 0 or T == 0 or Cin == 0 or K == 0:
        return out

    dtype = input.dtype
    use_dot = Cin >= 32 and Cout >= 32 and dtype is not torch.float64

    if use_dot:
        block_t, block_co, block_ci, warps = _pick_config(dtype, Tout, Cout, Cin, B)
        grid = ((Tout + block_t - 1) // block_t, (Cout + block_co - 1) // block_co, B)
        if (
            input.is_contiguous()
            and weight.is_contiguous()
            and bias.is_contiguous()
            and out.is_contiguous()
        ):
            _conv_tbc_dot_packed[grid](
                input,
                weight,
                bias,
                out,
                T,
                B,
                Cin,
                Cout,
                K,
                pad,
                BLOCK_T=block_t,
                BLOCK_CO=block_co,
                BLOCK_CI=block_ci,
                CI_FULL=(Cin % block_ci == 0),
                num_warps=warps,
                num_stages=3,
            )
            return out
        _conv_tbc_dot_strided[grid](
            input,
            weight,
            bias,
            out,
            T,
            B,
            Cin,
            Cout,
            K,
            pad,
            Tout,
            *input.stride(),
            *weight.stride(),
            bias.stride(0),
            *out.stride(),
            BLOCK_T=block_t,
            BLOCK_CO=block_co,
            BLOCK_CI=block_ci,
            num_warps=warps,
            num_stages=3,
        )
        return out

    block_t = min(128, max(16, _next_pow2(Tout)))
    block_co = min(128, max(16, _next_pow2(Cout)))
    grid = ((Tout + block_t - 1) // block_t, (Cout + block_co - 1) // block_co, B)
    _conv_tbc_fma_kernel[grid](
        input,
        weight,
        bias,
        out,
        T,
        B,
        Cin,
        Cout,
        K,
        pad,
        Tout,
        *input.stride(),
        *weight.stride(),
        bias.stride(0),
        *out.stride(),
        BLOCK_T=block_t,
        BLOCK_CO=block_co,
        num_warps=4,
        num_stages=2,
    )
    return out
