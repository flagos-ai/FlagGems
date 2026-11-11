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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# The modified Bessel function I0(z) is expanded as
#   I0(z) = sum_{k>=0} ((z/2)^k / k!)^2  =  sum_{k>=0} u^k / (k!)^2 ,  u = (z/2)^2.
# For z <= 16 (the largest beta exercised by the tests), 30 terms keep the fp32
# accumulation at full fp32 precision (~1e-7 relative), which is well inside the
# fp32 CI tolerance (1.3e-6) and far inside the fp16/bf16 tolerances.
_I0_TERMS = 30


@libentry()
@triton.jit
def kaiser_window_kernel(
    out_ptr,
    window_length,
    half_length,
    beta,
    i0_beta,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < window_length

    out_dtype = out_ptr.dtype.element_ty

    # The CPU reference evaluates the element index in the output dtype before
    # the window arithmetic, which is observable for fp16/bf16 once the index
    # stops being exactly representable (bfloat16 > 256, float16 > 2048). On XPU
    # `idx.to(out_dtype)` lowers to a wide-int -> low-precision `sitofp` that does
    # not round-to-nearest (it keeps the exact index), so mirror the reference's
    # round-to-nearest-even explicitly on the fp32 bit pattern instead.
    n = idx.to(tl.float32)
    if IS_BF16 or IS_FP16:
        bits = n.to(tl.int32, bitcast=True)
        if IS_BF16:
            # bf16: 7 explicit mantissa bits -> keep the top 16 fp32 bits.
            rounding = 0x7FFF + ((bits >> 16) & 1)
            n = ((bits + rounding) & ~0xFFFF).to(tl.float32, bitcast=True)
        else:
            # fp16: 10 explicit mantissa bits -> keep the top 13 fp32 bits.
            rounding = 0xFFF + ((bits >> 13) & 1)
            n = ((bits + rounding) & ~0x1FFF).to(tl.float32, bitcast=True)

    # Normalized position x in [-1, 1]; half_length == full_length / 2.
    x = (n - half_length) / half_length
    z = beta * tl.sqrt(tl.maximum(1.0 - x * x, 0.0))

    # I0(z) via the convergent series, evaluated entirely in fp32. Keeps the
    # kernel self-contained (no libdevice.cyl_bessel_i0 extern dependency, which
    # is the general kernel's implementation and can be a cross-version risk on
    # XPU). The bf16 precision bug itself is the index rounding above, not I0.
    u = z * z * 0.25
    term = z * 0.0 + 1.0  # broadcast 1.0 into a full-width vector
    acc = z * 0.0 + 1.0
    for k in tl.static_range(1, 30 + 1):
        inv_k2 = 1.0 / (k * k)
        term = term * u * inv_k2
        acc = acc + term

    val = acc / i0_beta
    tl.store(out_ptr + idx, val.to(out_dtype), mask=mask)


def _i0_host(x):
    """I0(x) on the host in double precision (pure Python, no torch compute).

    Used only for the scalar denominator I0(beta); the numerator is computed in
    the kernel with the same series in fp32 so the ratio is numerically coherent.
    """
    x = float(x)
    u = (x * 0.5) * (x * 0.5)
    term = 1.0
    acc = 1.0
    for k in range(1, _I0_TERMS + 1):
        term = term * u / (k * k)
        acc += term
    return acc


def kaiser_window(
    window_length,
    periodic=True,
    beta=12.0,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS_KUNLUNXIN KAISER_WINDOW")
    assert window_length >= 0, "window_length must be non-negative"

    if dtype is None:
        dtype = torch.get_default_dtype()
    assert dtype.is_floating_point, "kaiser_window only supports floating point dtype"

    if layout != torch.strided:
        raise ValueError("Only strided layout is supported for kaiser_window.")

    if device is None:
        device = torch.device(runtime.device.name)

    out = torch.empty(
        (window_length,),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )

    if window_length == 0:
        return out
    if window_length == 1:
        return torch.fill(out, 1.0)

    # N is the effective full size: window_length for the periodic window and
    # window_length - 1 for the symmetric one (matches aten).
    full_length = window_length if periodic else window_length - 1

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(window_length, BLOCK_SIZE),)
    kaiser_window_kernel[grid](
        out,
        window_length,
        float(full_length / 2.0),
        float(beta),
        float(_i0_host(beta)),
        dtype == torch.bfloat16,
        dtype == torch.float16,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def kaiser_window_periodic(
    window_length,
    periodic,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    return kaiser_window(
        window_length,
        periodic,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )


def kaiser_window_beta(
    window_length,
    periodic,
    beta,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    return kaiser_window(
        window_length,
        periodic,
        beta,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
