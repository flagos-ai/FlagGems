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
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.reflection_pad3d_backward import (
    reflection_pad3d_backward as default_reflection_pad3d_backward,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

# Map a torch dtype to its triton counterpart for the kernel store.
_TORCH_TO_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK": 4096}, num_warps=16, num_stages=1),
    ],
    key=["dhw_in", "dtype_size"],
)
@triton.jit(do_not_specialize=["pad_d0", "pad_h0", "pad_w0"])
def reflection_pad3d_backward_kernel(
    out_ptr,
    grad_ptr,
    D_out,
    H_out,
    W_out,
    D_in: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    pad_d0,
    pad_h0,
    pad_w0,
    dhw_in,
    dtype_size,  # used for autotune key
    OUT_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather-based backward kernel.

    Each program handles ``BLOCK`` input positions. For every input position
    ``i`` the forward reflection maps it to one output position directly, plus
    up to two extra output positions produced by the reflection on the padded
    borders (``-i`` and ``2*(size-1)-i``). The gradient w.r.t. an input element
    is therefore the sum of ``grad_output`` over (at most) three output
    positions per spatial axis -- ``3*3*3 = 27`` candidates, statically
    unrolled. This avoids atomic operations and the associated fp32
    accumulator, which matters on mthreads where atomic_add of half precision
    is not directly supported.
    """
    pid = ext.program_id(0)
    pid_nc = ext.program_id(1)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < dhw_in

    HW_in: tl.constexpr = H_in * W_in
    d_in = offs // HW_in
    hw_rem = offs - d_in * HW_in
    w_in = hw_rem % W_in
    h_in = (hw_rem - w_in) // W_in

    # Candidate output coordinates per axis. ``a`` is the central copy
    # (``pad + i``); ``b`` (``pad - i``) and ``c`` (``pad + 2*(size-1) - i``)
    # are the border reflections. ``b`` duplicates ``a`` at ``i == 0`` and
    # ``c`` duplicates ``a`` at ``i == size-1``, so those are masked out.
    Dm1: tl.constexpr = D_in - 1
    Dp: tl.constexpr = 2 * Dm1
    od_a = pad_d0 + d_in
    od_b = pad_d0 - d_in
    od_c = pad_d0 + (Dp - d_in)
    vd_a = mask & (od_a < D_out) & (od_a >= 0)
    vd_b = mask & (od_b < D_out) & (od_b >= 0) & (d_in != 0)
    vd_c = mask & (od_c < D_out) & (od_c >= 0) & (d_in != Dm1)

    Hm1: tl.constexpr = H_in - 1
    Hp: tl.constexpr = 2 * Hm1
    oh_a = pad_h0 + h_in
    oh_b = pad_h0 - h_in
    oh_c = pad_h0 + (Hp - h_in)
    vh_a = mask & (oh_a < H_out) & (oh_a >= 0)
    vh_b = mask & (oh_b < H_out) & (oh_b >= 0) & (h_in != 0)
    vh_c = mask & (oh_c < H_out) & (oh_c >= 0) & (h_in != Hm1)

    Wm1: tl.constexpr = W_in - 1
    Wp: tl.constexpr = 2 * Wm1
    ow_a = pad_w0 + w_in
    ow_b = pad_w0 - w_in
    ow_c = pad_w0 + (Wp - w_in)
    vw_a = mask & (ow_a < W_out) & (ow_a >= 0)
    vw_b = mask & (ow_b < W_out) & (ow_b >= 0) & (w_in != 0)
    vw_c = mask & (ow_c < W_out) & (ow_c >= 0) & (w_in != Wm1)

    HW_out = H_out * W_out
    grad_base = pid_nc * (D_out * HW_out)

    # Accumulate in float32 (mthreads has no fp64) and downcast on store.
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for di in tl.static_range(0, 3):
        od = tl.where(di == 0, od_a, tl.where(di == 1, od_b, od_c))
        vd = tl.where(di == 0, vd_a, tl.where(di == 1, vd_b, vd_c))
        for hi in tl.static_range(0, 3):
            oh = tl.where(hi == 0, oh_a, tl.where(hi == 1, oh_b, oh_c))
            vh = tl.where(hi == 0, vh_a, tl.where(hi == 1, vh_b, vh_c))
            for wi in tl.static_range(0, 3):
                ow = tl.where(wi == 0, ow_a, tl.where(wi == 1, ow_b, ow_c))
                vw = tl.where(wi == 0, vw_a, tl.where(wi == 1, vw_b, vw_c))
                vmask = vd & vh & vw
                grad_off = grad_base + od * HW_out + oh * W_out + ow
                grad_val = tl.load(grad_ptr + grad_off, mask=vmask, other=0.0).to(
                    tl.float32
                )
                acc += grad_val

    out_off = pid_nc * (D_in * HW_in) + offs
    tl.store(out_ptr + out_off, acc.to(OUT_DTYPE), mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_DHW": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_DHW": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_DHW": 1024}, num_warps=8, num_stages=2),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit
def _copy_kernel(in_ptr, out_ptr, n_elements, dtype_size, BLOCK_DHW: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK_DHW + tl.arange(0, BLOCK_DHW)
    mask = offs < n_elements
    vals = tl.load(in_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, vals, mask=mask)


def _use_triton_kernel(grad_output: torch.Tensor, self: torch.Tensor) -> bool:
    if not isinstance(grad_output, torch.Tensor) or not isinstance(self, torch.Tensor):
        return False
    if grad_output.device.type != "musa" or self.device.type != "musa":
        return False
    if (
        grad_output.dtype not in _SUPPORTED_DTYPES
        or self.dtype not in _SUPPORTED_DTYPES
    ):
        return False
    if grad_output.dtype != self.dtype:
        return False
    if grad_output.numel() == 0 or self.numel() == 0:
        return False
    if not grad_output.is_contiguous() or not self.is_contiguous():
        return False
    if self.dim() != 5:
        return False
    return True


def _launch_reflection_pad3d_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    padding: Tuple[int, ...],
) -> torch.Tensor:
    if isinstance(padding, int):
        pad_d0 = pad_d1 = pad_h0 = pad_h1 = pad_w0 = pad_w1 = padding
    else:
        pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1 = padding

    N, C, D_in, H_in, W_in = self.shape
    D_out = D_in + pad_d0 + pad_d1
    H_out = H_in + pad_h0 + pad_h1
    W_out = W_in + pad_w0 + pad_w1

    output_dtype = self.dtype
    grad_output = grad_output.contiguous()

    # No padding: grad_input is a copy of grad_output.
    if (
        pad_d0 == 0
        and pad_d1 == 0
        and pad_h0 == 0
        and pad_h1 == 0
        and pad_w0 == 0
        and pad_w1 == 0
    ):
        out = torch.empty_like(self)
        n_elements = out.numel()
        dtype_size = out.element_size()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_DHW"]),)
        with torch_device_fn.device(out.device):
            _copy_kernel[grid](grad_output, out, n_elements, dtype_size)
        return out

    # For pure reflection padding (pad < corresponding dim) each input element
    # receives at most 3 contributions per axis, so the backward can be computed
    # with a gather (no atomics). When padding is large enough that a single
    # output could reflect to the same input more than 3 times per axis the
    # closed-form fold used by the general kernel is required: fall back there.
    if (
        pad_d0 >= D_in
        or pad_d1 >= D_in
        or pad_h0 >= H_in
        or pad_h1 >= H_in
        or pad_w0 >= W_in
        or pad_w1 >= W_in
    ):
        return default_reflection_pad3d_backward(grad_output, self, padding)

    out = torch.empty_like(self)
    dhw_in = D_in * H_in * W_in
    dtype_size = grad_output.element_size()
    out_dtype = _TORCH_TO_TL_DTYPE[output_dtype]
    grid = lambda meta: (triton.cdiv(dhw_in, meta["BLOCK"]), N * C)

    with torch_device_fn.device(out.device):
        reflection_pad3d_backward_kernel[grid](
            out,
            grad_output,
            D_out,
            H_out,
            W_out,
            D_in,
            H_in,
            W_in,
            pad_d0,
            pad_h0,
            pad_w0,
            dhw_in,
            dtype_size,
            OUT_DTYPE=out_dtype,
        )
    return out


def reflection_pad3d_backward(grad_output, self, padding):
    logger.debug("GEMS_MTHREADS REFLECTION_PAD3D_BACKWARD")
    if not _use_triton_kernel(grad_output, self):
        return default_reflection_pad3d_backward(grad_output, self, padding)
    return _launch_reflection_pad3d_backward(grad_output, self, padding)
