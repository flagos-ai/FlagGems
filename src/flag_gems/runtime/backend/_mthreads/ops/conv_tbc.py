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

"""conv_tbc (time-batch-channel 1-D cross-correlation) as a single Triton kernel.

Semantics (PyTorch aten::conv_tbc, no kernel flip):
  x:(T,B,Cin), w:(K,Cin,Cout), bias:(Cout), pad:int
  TO = T + 2*pad - K + 1
  out[t,b,o] = bias[o] + sum_{k<K} sum_{i<Cin} x[t+k-pad, b, i] * w[k, i, o]
with x[j,b,i] = 0 outside 0 <= j < T.  Batch is never reduced.

Architecture: implicit GEMM.  Program (pid_c, pid_t, b) owns the tile
out[pid_t*BLOCK_T : +BLOCK_T, b, pid_c*BLOCK_CO : +BLOCK_CO] and accumulates the
(k, ci) reduction in fp32 with tl.dot over a (BLOCK_T, BLOCK_CI) x
(BLOCK_CI, BLOCK_CO) operand pair per tap.  All three tensors are consumed
through their real strides, so no host-side repacking is needed.

Numerics: the run-time pytest tolerance for fp32 is atol=1e-4 with rtol=1.3e-6.
This build's default fp32 dot precision is tf32-class (max-abs error ~1e-1, fails
the tolerance) and plain "ieee" lowers to a spill-prone scalar expansion
(0.32 ms vs 0.14 ms for ATen on the largest fp32 geometry), so fp32 uses
input_precision="tf32x3": a target numerical probe measured it to be *more*
accurate than ieee on all three benchmark geometries (worst per-element margin
0.30 vs 0.49 of the tolerance, zero elements over tolerance).  fp16/bf16 keep
the native dot (fp32 accumulate) and are rounded once at the store.

This round runs a bounded Triton autotune over tile/warp/stage shapes, keyed on
the shape scalars plus a dtype tag, to measure per-config timings on the target.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.conv_tbc import conv_tbc as default_conv_tbc
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_CONFIGS = [
    triton.Config(
        {"BLOCK_T": 32, "BLOCK_CO": 64, "BLOCK_CI": 32}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"BLOCK_T": 32, "BLOCK_CO": 64, "BLOCK_CI": 32}, num_warps=8, num_stages=3
    ),
    triton.Config(
        {"BLOCK_T": 32, "BLOCK_CO": 128, "BLOCK_CI": 32}, num_warps=8, num_stages=3
    ),
    triton.Config(
        {"BLOCK_T": 32, "BLOCK_CO": 64, "BLOCK_CI": 64}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_T": 64, "BLOCK_CO": 64, "BLOCK_CI": 32}, num_warps=8, num_stages=3
    ),
    triton.Config(
        {"BLOCK_T": 64, "BLOCK_CO": 64, "BLOCK_CI": 64}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_T": 16, "BLOCK_CO": 64, "BLOCK_CI": 32}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_T": 32, "BLOCK_CO": 32, "BLOCK_CI": 32}, num_warps=4, num_stages=2
    ),
]


@triton.autotune(
    configs=_CONFIGS,
    key=["T", "B", "Cin", "Cout", "pad", "dt_tag", "k_tag"],
)
@triton.jit
def _conv_tbc_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    T,
    B,
    Cin,
    Cout,
    pad,
    TO,
    dt_tag,
    k_tag,
    sx0,
    sx1,
    sx2,
    sw0,
    sw1,
    sw2,
    sb0,
    so0,
    so1,
    so2,
    K: tl.constexpr,
    FP32: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)
    b = tl.program_id(2)

    t_vec = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    co_vec = pid_c * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_co = co_vec < Cout

    acc = tl.zeros((BLOCK_T, BLOCK_CO), dtype=tl.float32)
    x_row = x_ptr + b * sx1
    for k in tl.static_range(0, K):
        tt = t_vec + k - pad
        mask_t = (tt >= 0) & (tt < T)
        w_k = w_ptr + k * sw0
        for ci0 in range(0, Cin, BLOCK_CI):
            ci_vec = ci0 + tl.arange(0, BLOCK_CI)
            mask_ci = ci_vec < Cin
            a = tl.load(
                x_row + tt[:, None] * sx0 + ci_vec[None, :] * sx2,
                mask=mask_t[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w = tl.load(
                w_k + ci_vec[:, None] * sw1 + co_vec[None, :] * sw2,
                mask=mask_ci[:, None] & mask_co[None, :],
                other=0.0,
            )
            if FP32:
                acc = tl.dot(a, w, acc, input_precision="tf32x3")
            else:
                acc = tl.dot(a, w, acc)

    bias = tl.load(b_ptr + co_vec * sb0, mask=mask_co, other=0.0).to(tl.float32)
    acc += bias[None, :]

    tl.store(
        o_ptr + t_vec[:, None] * so0 + b * so1 + co_vec[None, :] * so2,
        acc.to(o_ptr.dtype.element_ty),
        mask=(t_vec < TO)[:, None] & mask_co[None, :],
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


def conv_tbc(input, weight, bias, pad=0, out=None):
    """conv_tbc entry point (also serves the out variant when out= is given).

    The first parameter is named ``input`` to match the generic implementation
    in ``flag_gems.ops.conv_tbc``, which is the name callers pass by keyword.
    """
    logger.debug("GEMS_MTHREADS CONV_TBC")

    if not _use_triton_kernel(input) or not _use_triton_kernel(weight):
        return default_conv_tbc(input, weight, bias, pad=pad, out=out)

    with torch_device_fn.device(input.device):
        return _launch_conv_tbc(input, weight, bias, pad, out)


def _launch_conv_tbc(input, weight, bias, pad, out):
    x = input
    T, B, Cin = x.shape
    K, _, Cout = weight.shape
    TO = T + 2 * pad - K + 1

    if out is None:
        out = torch.empty((TO, B, Cout), dtype=x.dtype, device=x.device)

    if TO > 0 and B > 0 and Cout > 0 and Cin > 0:
        grid = lambda meta: (  # noqa: E731
            triton.cdiv(Cout, meta["BLOCK_CO"]),
            triton.cdiv(TO, meta["BLOCK_T"]),
            B,
        )
        _conv_tbc_kernel[grid](
            x,
            weight,
            bias,
            out,
            T,
            B,
            Cin,
            Cout,
            pad,
            TO,
            1 if x.dtype == torch.float32 else 0,
            K,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            bias.stride(0),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            K=K,
            FP32=(x.dtype == torch.float32),
        )
    return out
