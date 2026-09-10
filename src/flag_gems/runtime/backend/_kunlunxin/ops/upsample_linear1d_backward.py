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


@triton.jit
def upsample_linear1d_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    n,
    c,
    in_w,
    out_w,
    go_stride_n,
    go_stride_c,
    go_stride_w,
    align_corners: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One lane per input element; each lane walks the output window that can
    # scatter into it, so no atomics are needed.
    #
    # Every read is clamped into range and zeroed with `tl.where` instead of
    # relying on a masked load. A data-dependent predicate (`x_out` inside
    # [0, out_w)) does not reliably suppress lanes on this backend: the load
    # leaks the real in-range value, and the leaked gradient then gets summed
    # into the accumulator. That showed up as exactly the first and last input
    # element of every (n, c) plane being wrong - the only lanes whose window
    # crosses the boundary - i.e. 538 of 1080 cases.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    total = n * c * in_w
    mask = offs < total
    offs_c = tl.minimum(offs, total - 1)

    x_in = offs_c % in_w
    tmp = offs_c // in_w
    c_idx = tmp % c
    n_idx = tmp // c

    x_in_f = x_in.to(tl.float32)
    in_w_f = tl.cast(in_w, tl.float32)
    out_w_f = tl.cast(out_w, tl.float32)

    if align_corners:
        if in_w > 1:
            center = x_in_f * (out_w_f - 1.0) / (in_w_f - 1.0)
        else:
            center = tl.zeros([BLOCK], dtype=tl.float32)
    else:
        center = (x_in_f + 0.5) * out_w_f / in_w_f - 0.5

    base = tl.floor(center).to(tl.int32)

    go_base = grad_out_ptr + n_idx * go_stride_n + c_idx * go_stride_c

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for i in tl.static_range(-WINDOW, WINDOW + 1):
        x_out = base + i
        valid = (x_out >= 0) & (x_out < out_w)
        x_out_c = tl.maximum(tl.minimum(x_out, out_w - 1), 0)
        x_out_f = x_out_c.to(tl.float32)

        if align_corners:
            if out_w > 1:
                x_real = x_out_f * (in_w_f - 1.0) / (out_w_f - 1.0)
            else:
                x_real = tl.zeros([BLOCK], dtype=tl.float32)
        else:
            x_real = (x_out_f + 0.5) * in_w_f / out_w_f - 0.5

        x0_f = tl.floor(x_real)
        w1 = x_real - x0_f
        w0 = 1.0 - w1

        x0_i = tl.maximum(x0_f, 0.0).to(tl.int32)
        x1_i = tl.minimum(x0_f + 1.0, in_w_f - 1.0).to(tl.int32)

        g = tl.load(go_base + x_out_c * go_stride_w).to(tl.float32)
        g = tl.where(valid, g, 0.0)

        same = x0_i == x1_i
        is_x0 = x_in == x0_i
        is_x1 = x_in == x1_i

        acc += tl.where(same & is_x0, g * (w0 + w1), 0.0)
        acc += tl.where((~same) & is_x0, g * w0, 0.0)
        acc += tl.where((~same) & is_x1, g * w1, 0.0)

    # Raw stride-1 store index (clamping it would collapse the contiguous block
    # DMA); the tail is handled by the bool mask, which does suppress stores.
    tl.store(grad_in_ptr + offs, acc.to(grad_in_ptr.dtype.element_ty), mask=mask)


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
        raise ValueError(f"unsupported input_size {input_size}")

    if output_size is not None:
        out_w = output_size[0]
    else:
        assert scale_factors is not None
        out_w = int(in_w * scale_factors[0])

    assert grad_output.shape[-1] == out_w

    # Read grad_output through its own strides: routing a non-contiguous source
    # through torch's copy_ fails with `invalid device function` here.
    go = grad_output.reshape(n, c, out_w) if grad_output.ndim != 3 else grad_output
    go_stride_n, go_stride_c, go_stride_w = go.stride()

    grad_in = torch.empty(
        (n, c, in_w),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    BLOCK = 512
    WINDOW = triton.cdiv(out_w, in_w) + 2
    grid = (triton.cdiv(n * c * in_w, BLOCK),)

    with torch_device_fn.device(grad_output.device):
        upsample_linear1d_backward_kernel[grid](
            go,
            grad_in,
            n,
            c,
            in_w,
            out_w,
            go_stride_n,
            go_stride_c,
            go_stride_w,
            align_corners,
            WINDOW=WINDOW,
            BLOCK=BLOCK,
        )

    return grad_in.reshape(input_size)
