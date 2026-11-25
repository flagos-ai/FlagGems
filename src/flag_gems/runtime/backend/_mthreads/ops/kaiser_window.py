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
"""Kaiser window (FlagGems `kaiser_window`) implemented as a single Triton kernel.

Semantics follow torch.kaiser_window:
  * L == 0 -> empty tensor
  * L == 1 -> exactly 1.0
  * otherwise, with M = L when periodic else L - 1 (M >= 2):
        t = (n - M/2) / (M/2)          # = 2n/(M-1) - 1
        out[n] = I0(beta * sqrt(1 - t^2)) / I0(beta)
Precision boundaries of the reference:
  * float64 output -> fp64 arithmetic with correctly rounded sqrt/div
  * float16/bfloat16 output -> the element index is rounded to the output dtype
    before the arithmetic (that is what the reference kernel does), then all
    arithmetic is fp32 with a single cast at the store
  * float32 output -> fp32 arithmetic
I0 is evaluated with the self-contained Numerical-Recipes bessi0 polynomial
(max relative error ~1.6e-8) so no vendor libdevice symbol is required.  The
per-element divisions are replaced with reciprocals computed once per program
and with rsqrt, which measured the same ~1.2e-7 relative error as the hardware
division it replaces.  The per-element chain is long (sqrt -> rsqrt -> exp -> 8
dependent FMAs), so the block shape is a bounded autotune list whose axis is
elements per thread (instruction-level parallelism); the launch-bound
small-window cases can still select a small block.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.kaiser_window import kaiser_window as default__kaiser_window

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)
_CONFIGS = [
    triton.Config({"BLOCK": 256}, num_warps=4),
    triton.Config({"BLOCK": 512}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=8),
    triton.Config({"BLOCK": 2048}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=8),
]


@triton.jit
def _i0(x):
    """Modified Bessel function of the first kind, order 0 (NR bessi0, fast)."""
    ax = tl.abs(x)
    axs = tl.maximum(ax, 1e-20)
    y = axs / 3.75
    y2 = y * y
    small = 1.0 + y2 * (
        3.5156229
        + y2
        * (
            3.0899424
            + y2 * (1.2067492 + y2 * (0.2659732 + y2 * (0.0360768 + y2 * 0.0045813)))
        )
    )
    # z = 3.75 / axs and exp(axs) / sqrt(axs) without a division or a second sqrt.
    r = tl.rsqrt(axs)
    z = 3.75 * r * r
    large = (
        tl.exp(axs)
        * r
        * (
            0.39894228
            + z
            * (
                0.01328592
                + z
                * (
                    0.00225319
                    + z
                    * (
                        -0.00157565
                        + z
                        * (
                            0.00916281
                            + z
                            * (
                                -0.02057706
                                + z * (0.02635537 + z * (-0.01647633 + z * 0.00392377))
                            )
                        )
                    )
                )
            )
        )
    )
    return tl.where(ax < 3.75, small, large)


@triton.jit
def _i0_precise(x):
    """Same polynomial with correctly rounded sqrt/divide (fp64 path)."""
    ax = tl.abs(x)
    axs = tl.maximum(ax, 1e-20)
    y = axs / 3.75
    y2 = y * y
    small = 1.0 + y2 * (
        3.5156229
        + y2
        * (
            3.0899424
            + y2 * (1.2067492 + y2 * (0.2659732 + y2 * (0.0360768 + y2 * 0.0045813)))
        )
    )
    z = 3.75 / axs
    large = (tl.exp(axs) / tl.sqrt(axs)) * (
        0.39894228
        + z
        * (
            0.01328592
            + z
            * (
                0.00225319
                + z
                * (
                    -0.00157565
                    + z
                    * (
                        0.00916281
                        + z
                        * (
                            -0.02057706
                            + z * (0.02635537 + z * (-0.01647633 + z * 0.00392377))
                        )
                    )
                )
            )
        )
    )
    return tl.where(ax < 3.75, small, large)


@triton.autotune(configs=_CONFIGS, key=["length", "IS_FP64", "IS_HALF"])
@triton.jit
def _kaiser_kernel(
    out_ptr,
    length,
    half,
    beta,
    IS_FP64: tl.constexpr,
    IS_HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < length
    out_dtype = out_ptr.dtype.element_ty

    if IS_FP64:
        n = offs.to(tl.float64)
        h = half.to(tl.float64)
        b = beta.to(tl.float64)
        x = (n - h) / h
        s = tl.sqrt(tl.maximum(1.0 - x * x, 0.0))
        val = _i0_precise(b * s) / _i0_precise(b)
    else:
        if IS_HALF:
            # torch converts the element index into the output dtype first.
            n = offs.to(out_dtype).to(tl.float32)
        else:
            n = offs.to(tl.float32)
        inv_half = 1.0 / half
        x = (n - half) * inv_half
        s = tl.sqrt(tl.maximum(1.0 - x * x, 0.0))
        val = _i0(beta * s) * (1.0 / _i0(beta))

    val = tl.where(length == 1, 1.0, val)
    tl.store(out_ptr + offs, val.to(out_dtype), mask=mask)


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
    logger.debug("GEMS_MTHREADS KAISER_WINDOW")
    assert window_length >= 0, "window_length must be non-negative"

    requested = torch.device(device) if device is not None else None
    if requested is not None and requested.type != "musa":
        return default__kaiser_window(
            window_length,
            periodic,
            beta,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )

    if dtype is None:
        dtype = torch.get_default_dtype()
    assert dtype.is_floating_point, "kaiser_window only supports floating point dtype"
    if isinstance(layout, str):
        if layout != "strided":
            raise ValueError("Only strided layout is supported for kaiser_window.")
        layout = torch.strided
    if layout is not torch.strided:
        raise ValueError("Only strided layout is supported for kaiser_window.")
    if device is None:
        device = torch.device("musa")

    out = torch.empty(
        (window_length,),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    length = out.numel()
    if length == 0:
        return out

    denom = float(length) if periodic else float(length - 1)
    half = 0.5 * denom if denom > 0.0 else 0.5

    grid = lambda meta: ((length + meta["BLOCK"] - 1) // meta["BLOCK"],)

    _kaiser_kernel[grid](
        out,
        length,
        half,
        float(beta),
        dtype == torch.float64,
        dtype is torch.float16 or dtype is torch.bfloat16,
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
