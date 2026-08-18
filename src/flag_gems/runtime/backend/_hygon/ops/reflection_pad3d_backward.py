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

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def reflection_pad3d_backward_kernel(
    grad_ptr,
    out_ptr,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    pad_d0,
    pad_h0,
    pad_w0,
    stride_grad_n,
    stride_grad_c,
    stride_grad_d,
    stride_grad_h,
    stride_grad_w,
    stride_out_n,
    stride_out_c,
    stride_out_d,
    stride_out_h,
    stride_out_w,
    BLOCK_DHW: tl.constexpr,
):
    # Parallelize over the input (grad_input) space so that every element is
    # written exactly once. This avoids the atomic_add scatter used by the
    # generic implementation and lets bfloat16 accumulate in float32 and store
    # directly, without a separate float32 buffer + cast pass.
    pid_bc = tle.program_id(1)
    pid_dhw = tle.program_id(0)

    pid_n = pid_bc % N
    pid_c = pid_bc // N

    dhw_start = pid_dhw * BLOCK_DHW
    dhw_offs = dhw_start + tl.arange(0, BLOCK_DHW)
    in_mask = dhw_offs < D_in * H_in * W_in

    # Decode 3D coordinates of the input position.
    d_in = dhw_offs // (H_in * W_in)
    h_in = (dhw_offs // W_in) % H_in
    w_in = dhw_offs % W_in

    # For each input coordinate there are up to three grad_output positions that
    # reflect onto it (guaranteed when pad < dim size, which the tests enforce):
    #   - direct  : o = i + pad0
    #   - left    : o = pad0 - i           (valid for 1 <= i <= pad0)
    #   - right   : o = pad0 + 2*(L-1) - i (valid for L-1-pad1 <= i <= L-2)

    # Depth candidates
    d_c0 = d_in + pad_d0
    d_m0 = in_mask
    d_c1 = pad_d0 - d_in
    d_m1 = in_mask & (d_in >= 1) & (d_in <= pad_d0)
    d_c2 = pad_d0 + 2 * (D_in - 1) - d_in
    d_m2 = in_mask & (d_c2 >= pad_d0 + D_in) & (d_c2 < D_out)

    # Height candidates
    h_c0 = h_in + pad_h0
    h_m0 = in_mask
    h_c1 = pad_h0 - h_in
    h_m1 = in_mask & (h_in >= 1) & (h_in <= pad_h0)
    h_c2 = pad_h0 + 2 * (H_in - 1) - h_in
    h_m2 = in_mask & (h_c2 >= pad_h0 + H_in) & (h_c2 < H_out)

    # Width candidates
    w_c0 = w_in + pad_w0
    w_m0 = in_mask
    w_c1 = pad_w0 - w_in
    w_m1 = in_mask & (w_in >= 1) & (w_in <= pad_w0)
    w_c2 = pad_w0 + 2 * (W_in - 1) - w_in
    w_m2 = in_mask & (w_c2 >= pad_w0 + W_in) & (w_c2 < W_out)

    grad_base = pid_n * stride_grad_n + pid_c * stride_grad_c
    acc = tl.zeros((BLOCK_DHW,), dtype=tl.float32)

    for di in tl.static_range(3):
        if di == 0:
            d_c, d_m = d_c0, d_m0
        elif di == 1:
            d_c, d_m = d_c1, d_m1
        else:
            d_c, d_m = d_c2, d_m2
        for hi in tl.static_range(3):
            if hi == 0:
                h_c, h_m = h_c0, h_m0
            elif hi == 1:
                h_c, h_m = h_c1, h_m1
            else:
                h_c, h_m = h_c2, h_m2
            for wi in tl.static_range(3):
                if wi == 0:
                    w_c, w_m = w_c0, w_m0
                elif wi == 1:
                    w_c, w_m = w_c1, w_m1
                else:
                    w_c, w_m = w_c2, w_m2
                m = d_m & h_m & w_m
                grad_offs = (
                    grad_base
                    + d_c * stride_grad_d
                    + h_c * stride_grad_h
                    + w_c * stride_grad_w
                )
                val = tl.load(grad_ptr + grad_offs, mask=m, other=0.0).to(tl.float32)
                acc += val

    out_offs = (
        pid_n * stride_out_n
        + pid_c * stride_out_c
        + d_in * stride_out_d
        + h_in * stride_out_h
        + w_in * stride_out_w
    )
    tl.store(out_ptr + out_offs, acc, mask=in_mask)


def reflection_pad3d_backward(grad_output, self, padding):
    """Compute gradient of reflection_pad3d forward pass.

    Args:
        grad_output: Gradient of the loss with respect to the output of reflection_pad3d.
                    Shape: (N, C, D + pad_d0 + pad_d1, H + pad_h0 + pad_h1, W + pad_w0 + pad_w1)
        self: Original input tensor before padding. Shape: (N, C, D, H, W)
        padding: Tuple of 6 ints (pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1)

    Returns:
        Gradient with respect to self. Shape: (N, C, D, H, W)
    """
    logger.debug("GEMS_HYGON REFLECTION_PAD3D_BACKWARD")

    if isinstance(padding, int):
        pad_d0 = pad_d1 = pad_h0 = pad_h1 = pad_w0 = pad_w1 = padding
    else:
        pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1 = padding

    if self.dim() != 5:
        raise ValueError("input must be a 5D tensor")

    N, C, D_in, H_in, W_in = self.shape
    D_out, H_out, W_out = (
        D_in + pad_d0 + pad_d1,
        H_in + pad_h0 + pad_h1,
        W_in + pad_w0 + pad_w1,
    )

    expected_grad_shape = (N, C, D_out, H_out, W_out)
    if tuple(grad_output.shape) != expected_grad_shape:
        raise ValueError(
            f"grad_output has shape {tuple(grad_output.shape)}, expected {expected_grad_shape}"
        )

    # Handle empty padding case - just copy.
    if (
        pad_d0 == 0
        and pad_d1 == 0
        and pad_h0 == 0
        and pad_h1 == 0
        and pad_w0 == 0
        and pad_w1 == 0
    ):
        return grad_output.clone()

    grad_output = grad_output.contiguous()

    # Output is written exactly once per element, so we can allocate directly in
    # the input dtype (float32 accumulation happens inside the kernel).
    out = torch.zeros_like(self)

    # Parallelize over the input space: each program writes BLOCK_DHW input
    # elements for a given (n, c) slice.
    BLOCK_DHW = 256
    grid = (
        triton.cdiv(D_in * H_in * W_in, BLOCK_DHW),
        N * C,
    )

    reflection_pad3d_backward_kernel[grid](
        grad_output,
        out,
        N,
        C,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        pad_d0,
        pad_h0,
        pad_w0,
        *grad_output.stride(),
        *out.stride(),
        BLOCK_DHW=BLOCK_DHW,
    )

    return out
