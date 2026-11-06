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

"""KunlunXin (XPU) implementation of `upsample_linear1d_backward`.

With ``align_corners=False`` and an integer size ratio (``output_size == k *
input_size`` or ``input_size == k * output_size``) the source coordinate
``source(o) = (o + 0.5) / k - 0.5`` is exactly ``k``-periodic, so the backward
map becomes a fixed stencil:

* upscale: input ``x`` reads the compile-time taps ``o = k*x + j`` with the
  compile-time weights ``1 - |frac((j + 0.5)/k - 0.5)|``;
* downscale: input ``x = k*o + q + p`` takes ``grad_out[o]`` times the
  compile-time weight of ``p`` (``0.5`` at the two straddling positions when
  ``k`` is even, ``1`` at ``q`` when it is odd).

Both directions then address memory with an affine lane index and constant
offsets, which KunlunXin lowers to block DMA. A runtime ``tl.minimum`` /
``tl.maximum`` anywhere in an address instead turns the access into a per-lane
discrete gather -- measured 10-20x slower on this operator -- so neither fast
path clamps an address:

* upscale: the taps of the two boundary columns of each row leave the row.
  Those two columns are the *only* ones that do (see the proof below), so they
  are recomputed by a 64-lane boundary kernel, and the main kernel is allowed
  to read the neighbouring row's bytes for them. Only the very first and last
  tap of the whole tensor would leave the *allocation*, which is covered by a
  copy of ``grad_output`` with ``k`` zero elements of slack on each side.
* downscale reads and writes need no repair at all: ``k*o + q + 1 <= in_w - 1``
  for every ``o``.

Two further XPU-specific constraints shaped these kernels:

* **The lane index must be the flat element index, not a ``[rows, cols]``
  tile.** The identical address pattern costs 0.72 ms flattened and 7.3 ms as
  a 2-D tile -- a 10x penalty that has nothing to do with the addresses.
* ``out_w == k * in_w`` implies ``row * out_w + k * col == k * (row * in_w +
  col)``, so the flat index keeps the upscale addresses affine *across* rows;
  the block size is therefore free of any row-alignment constraint.

Everything else (``align_corners=True``, non-integer ratios, tiny tensors) uses
the general gather kernel below, which clamps every address and gates the
weight instead of relying on a masked load's ``other`` value: on KunlunXin a
masked ``tl.load`` still issues the access, so an unclamped lane can address
far outside ``grad_output`` and wedge the device.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Lanes per program. 4096 is ~1.5x faster than 2048 for the upscale stencil and
# 8192 is worse; the downscale is flat in this range.
_UP_BLOCK = 4096
_DOWN_BLOCK = 2048
# Below this element count the fast paths lose to the general kernel: they pay
# for an extra launch (the padded copy) and an extra boundary kernel, which a
# launch-bound case cannot absorb.
_MIN_FAST_ELEMS = 1 << 18


@triton.jit
def upsample_linear1d_backward_upscale_kernel(
    grad_out_ptr,
    grad_in_ptr,
    K: tl.constexpr,
    B: tl.constexpr,
):
    flat = tl.program_id(0) * B + tl.arange(0, B)
    # ``grad_out_ptr`` is offset by ``K`` (the leading slack), so a tap that
    # leaves the row reads a zero or the neighbouring row; the latter happens
    # only for the two boundary columns, which the boundary kernel rewrites.
    acc = tl.zeros([B], dtype=tl.float32)
    for j in tl.static_range(-(K // 2), (3 * K) // 2):
        f = (j + 0.5) / K - 0.5
        w = 1.0 - (f if f >= 0 else -f)
        acc += tl.load(grad_out_ptr + K * flat + j).to(tl.float32) * w
    tl.store(grad_in_ptr + flat, acc.to(grad_in_ptr.dtype.element_ty))


@triton.jit
def upsample_linear1d_backward_upscale_boundary_kernel(
    grad_out_ptr,
    grad_in_ptr,
    in_w,
    out_w,
    nc,
    K: tl.constexpr,
    W: tl.constexpr,
):
    # The two boundary columns of each row are not covered by the periodic
    # stencil: ATen clamps the source coordinate at both ends
    # (``r = max(r, 0)`` and ``x1 = min(x1, in_w-1)``), so e.g. the first and
    # last output hand their *full* weight to the boundary input instead of the
    # fractional one. Recompute those two columns with the exact formula.
    #
    # Exactly 64 lanes (KunlunXin executes narrower tiles as 64 lanes, and the
    # extra lanes would store garbage). 32 rows x 2 columns per program; when
    # ``nc`` is not a multiple of 32 the row index is clamped, so the duplicate
    # lanes store identical values to identical addresses.
    pid = tl.program_id(0)
    idx = tl.arange(0, 64)
    row = tl.minimum(pid * 32 + idx // 2, nc - 1)
    x = tl.where(idx % 2 == 0, 0, in_w - 1)
    x_f = x.to(tl.float32)
    center = (x_f + 0.5) * K - 0.5
    base = tl.floor(center).to(tl.int32)
    acc = tl.zeros([64], dtype=tl.float32)
    for i in tl.static_range(-W, W + 1):
        o = base + i
        valid = (o >= 0) & (o < out_w)
        o_c = tl.minimum(tl.maximum(o, 0), out_w - 1)
        x_real = tl.maximum((o_c.to(tl.float32) + 0.5) / K - 0.5, 0.0)
        x0_f = tl.floor(x_real)
        w1 = x_real - x0_f
        w0 = 1.0 - w1
        x0_i = x0_f.to(tl.int32)
        x1_i = tl.minimum(x0_f + 1.0, (in_w - 1).to(tl.float32)).to(tl.int32)
        g = tl.load(grad_out_ptr + row * out_w + o_c).to(tl.float32)
        same = x0_i == x1_i
        weight = tl.where(x == x0_i, tl.where(same, w0 + w1, w0), 0.0)
        weight += tl.where((~same) & (x == x1_i), w1, 0.0)
        acc += tl.where(valid, g * weight, 0.0)
    tl.store(
        grad_in_ptr + row * in_w + x,
        acc.to(grad_in_ptr.dtype.element_ty),
    )


@triton.jit
def upsample_linear1d_backward_downscale_kernel(
    grad_out_ptr,
    grad_in_ptr,
    K: tl.constexpr,
    B: tl.constexpr,
):
    # Input ``x = K*o + p`` reads ``grad_out[o]`` with the compile-time weight
    # of ``p``: the load is lane-indexed (each address used K times) and the
    # store is the raw flat index, so both sides stay block DMA. Storing
    # K*o + p per lane instead -- i.e. one lane per output -- costs 20-40x
    # because each such store is stride-K.
    flat = tl.program_id(0) * B + tl.arange(0, B)
    g = tl.load(grad_out_ptr + flat // K).to(tl.float32)
    p = flat - K * (flat // K)
    q: tl.constexpr = (K - 1) // 2
    if K % 2 == 0:
        w = tl.where((p == q) | (p == q + 1), 0.5, 0.0)
    else:
        w = tl.where(p == q, 1.0, 0.0)
    tl.store(grad_in_ptr + flat, (g * w).to(grad_in_ptr.dtype.element_ty))


@triton.jit
def upsample_linear1d_backward_gather_kernel(
    grad_out_ptr,
    grad_in_ptr,
    total,
    in_w,
    out_w,
    c_scale,
    c_in_off,
    c_out_off,
    r_scale,
    r_in_off,
    r_out_off,
    in_w_m1_f,
    WL: tl.constexpr,
    WR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # General path: one program owns a contiguous chunk of the input gradient,
    # scans the analytic contributor window and clamps every address. The clamp
    # is why this stays the *slow* path (see the module docstring), but it is
    # the only formulation that is correct for every ratio.
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total

    # Tail lanes must not produce an out-of-range *load* address: a masked
    # `tl.load` still issues the access on KunlunXin. Clamping `o` for the
    # gather index keeps the store index the raw stride-1 `o`.
    o_c = tl.minimum(o, total - 1)
    nc = o_c // in_w
    x_in = o_c - nc * in_w

    x_in_f = x_in.to(tl.float32)
    # center(x_in): continuous output coordinate that maps back to x_in.
    center = (x_in_f + c_in_off) * c_scale + c_out_off
    base = tl.floor(center).to(tl.int32)
    base_f = base.to(tl.float32)
    # x_real(x_out) is affine in x_out, so the per-iteration value is
    # xr_base + i * r_scale (no division).
    xr_base = (base_f + r_in_off) * r_scale + r_out_off

    go_base = nc * out_w
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for i in tl.static_range(-WL, WR + 1):
        x_out = base + i
        valid = (x_out >= 0) & (x_out < out_w)
        x_out_c = tl.minimum(tl.maximum(x_out, 0), out_w - 1)
        x_real = xr_base + i * r_scale
        x_real = tl.maximum(x_real, 0.0)
        x0_f = tl.floor(x_real)
        w1 = x_real - x0_f
        w0 = 1.0 - w1
        x0_i = x0_f.to(tl.int32)
        x1_i = tl.minimum(x0_f + 1.0, in_w_m1_f).to(tl.int32)

        g = tl.load(grad_out_ptr + go_base + x_out_c).to(tl.float32)

        same = x0_i == x1_i
        weight = tl.where(x_in == x0_i, tl.where(same, w0 + w1, w0), 0.0)
        weight += tl.where((~same) & (x_in == x1_i), w1, 0.0)
        acc += tl.where(valid, g * weight, 0.0)

    tl.store(grad_in_ptr + o, acc.to(grad_in_ptr.dtype.element_ty), mask=mask)


@triton.jit
def upsample_linear1d_backward_iw1_kernel(
    grad_out_ptr,
    grad_in_ptr,
    nc,
    out_w,
    BLOCK: tl.constexpr,
):
    # in_w == 1 is degenerate: every output position maps to input position 0,
    # so grad_in[nc, 0] is a plain row sum.
    row = tl.program_id(0)
    base = row * out_w
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, out_w, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        keep = offs < out_w
        offs_c = tl.minimum(offs, out_w - 1)
        g = tl.load(grad_out_ptr + base + offs_c).to(tl.float32)
        acc += tl.where(keep, g, 0.0)
    row_sum = tl.sum(acc, axis=0)
    tl.store(grad_in_ptr + row, row_sum.to(grad_in_ptr.dtype.element_ty))


def _window(c_scale):
    # Contributors to input x are the x_out with x_real(x_out) in [x-1, x+1),
    # i.e. x_out in [center - s, center + s) with s = c_scale. Since
    # base = floor(center), i = x_out - base satisfies frac(center) - s <= i <
    # 1 + s, hence |i| <= ceil(s).
    return max(1, int(math.ceil(c_scale - 1e-6)))


def _scale_value(scale_factors):
    # ``scale_factors`` is a float in the ATen schema and a sequence in some of
    # the reference wrappers.
    if scale_factors is None:
        return None
    if isinstance(scale_factors, (list, tuple)):
        return float(scale_factors[0]) if len(scale_factors) > 0 else None
    return float(scale_factors)


def _flat_with_slack(t, front, back):
    """``t`` flattened, preceded by ``front`` and followed by ``back`` zeros."""
    flat = t.contiguous().view(-1)
    n = flat.numel()
    buf = torch.empty(n + front + back, device=t.device, dtype=t.dtype)
    if front:
        buf[:front].zero_()
    buf[front : front + n].copy_(flat)
    if back:
        buf[front + n :].zero_()
    return buf


def _general_upsample_linear1d_backward(
    grad_output, n, c, in_w, out_w, align_corners, scale_factors
):
    grad_out_flat = grad_output.contiguous().view(n * c, out_w)
    grad_in = torch.empty(
        (n, c, in_w), device=grad_output.device, dtype=grad_output.dtype
    )

    if in_w == 1:
        with torch_device_fn.device(grad_output.device):
            upsample_linear1d_backward_iw1_kernel[(n * c,)](
                grad_out_flat, grad_in, n * c, out_w, BLOCK=1024
            )
        return grad_in

    if align_corners:
        c_scale = (out_w - 1.0) / (in_w - 1.0) if in_w > 1 else 0.0
        c_in_off, c_out_off = 0.0, 0.0
        r_scale = (in_w - 1.0) / (out_w - 1.0) if out_w > 1 else 0.0
        r_in_off, r_out_off = 0.0, 0.0
    else:
        # ATen uses scale/1-scale when a scale factor is supplied and the size
        # ratio otherwise; they differ whenever the ratio is not exactly the
        # reciprocal of the scale (``int(in_w * scale) != in_w * scale``).
        scale = _scale_value(scale_factors)
        if scale is not None and scale > 0:
            c_scale = scale
            r_scale = 1.0 / scale
        else:
            c_scale = out_w / in_w
            r_scale = in_w / out_w
        c_in_off, c_out_off = 0.5, -0.5
        r_in_off, r_out_off = 0.5, -0.5

    w = _window(c_scale)
    total = n * c * in_w
    block = 2048
    with torch_device_fn.device(grad_output.device):
        upsample_linear1d_backward_gather_kernel[(triton.cdiv(total, block),)](
            grad_out_flat,
            grad_in,
            total,
            in_w,
            out_w,
            c_scale,
            c_in_off,
            c_out_off,
            r_scale,
            r_in_off,
            r_out_off,
            float(in_w - 1),
            WL=w,
            WR=w,
            BLOCK=block,
        )
    return grad_in


def upsample_linear1d_backward(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scale_factors=None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_LINEAR1D_BACKWARD")

    if len(input_size) == 3:
        n, c, in_w = input_size
    elif len(input_size) == 2:
        n, c, in_w = input_size[0], 1, input_size[1]
    elif len(input_size) == 1:
        n, c, in_w = 1, 1, input_size[0]
    else:
        raise ValueError

    if output_size is not None:
        out_w = output_size[0]
    else:
        assert scale_factors is not None
        out_w = int(in_w * scale_factors[0])

    assert grad_output.shape[-1] == out_w

    if (
        not align_corners
        and in_w > 1
        and n * c * in_w >= _MIN_FAST_ELEMS
        and (out_w % in_w == 0 or in_w % out_w == 0)
    ):
        # The same reciprocal ATen derives: 1/scale when a scale factor is
        # given, input/output otherwise.
        # (``scale_factors`` is a float in the ATen schema, a sequence in some
        # of the reference wrappers.)
        scale = _scale_value(scale_factors)
        r_scale = 1.0 / scale if scale is not None and scale > 0 else in_w / out_w
        upscale = out_w % in_w == 0
        k = out_w // in_w if upscale else in_w // out_w
        # Integer-ratio stencils are only valid when the stride really is k;
        # ATen keeps the requested scale even when it disagrees with the
        # integer shape ratio.
        exact = abs(r_scale * k - 1.0) < 1e-6 if upscale else abs(r_scale - k) < 1e-6

        if exact:
            dtype = grad_output.dtype
            total = n * c * in_w

            if upscale:
                b = _UP_BLOCK
                in_pad = triton.cdiv(total, b) * b
                # ``k`` zero elements in front, then grad_output, then zeros for
                # the tail lanes and for the last row's last tap.
                go_pad = _flat_with_slack(grad_output, k, k * (in_pad - total) + k)
                grad_in_buf = torch.empty(
                    in_pad, device=grad_output.device, dtype=dtype
                )
                with torch_device_fn.device(grad_output.device):
                    upsample_linear1d_backward_upscale_kernel[(in_pad // b,)](
                        go_pad[k:], grad_in_buf, K=k, B=b
                    )
                    # Overwrite the two columns whose stencil leaves the row.
                    upsample_linear1d_backward_upscale_boundary_kernel[
                        (triton.cdiv(n * c, 32),)
                    ](
                        grad_output.contiguous().view(-1),
                        grad_in_buf,
                        in_w,
                        out_w,
                        n * c,
                        K=k,
                        W=k,
                    )
                return grad_in_buf[:total].view(n, c, in_w)

            b = _DOWN_BLOCK
            # Every input element ``k*o + p`` is written exactly once, so the
            # buffer needs no zero fill; only the tail lanes of the last block
            # read past ``grad_out``, which the copy's slack absorbs.
            grad_in_buf = torch.empty(
                triton.cdiv(total, b) * b, device=grad_output.device, dtype=dtype
            )
            go_pad = _flat_with_slack(grad_output, 0, b)
            with torch_device_fn.device(grad_output.device):
                upsample_linear1d_backward_downscale_kernel[(triton.cdiv(total, b),)](
                    go_pad, grad_in_buf, K=k, B=b
                )
            return grad_in_buf[:total].view(n, c, in_w)

    return _general_upsample_linear1d_backward(
        grad_output, n, c, in_w, out_w, align_corners, scale_factors
    )
