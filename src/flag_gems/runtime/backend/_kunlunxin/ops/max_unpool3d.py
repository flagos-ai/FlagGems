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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _max_unpool3d_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    in_plane_size,
    out_plane_size,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    indices = tl.load(indices_ptr + offsets, mask=mask, other=-1)
    plane = offsets // in_plane_size
    valid = mask & (indices >= 0) & (indices < out_plane_size)
    tl.store(output_ptr + plane * out_plane_size + indices, values, mask=valid)


def _max_unpool3d_forward(
    input: torch.Tensor,
    indices: torch.Tensor,
    kernel_size,
    stride,
    padding,
    output_size,
):
    input = input.contiguous()
    indices = indices.contiguous()

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride,) * 3
    if isinstance(padding, int):
        padding = (padding,) * 3

    in_n, in_c, in_d, in_h, in_w = input.shape
    if output_size is None:
        out_d = (in_d - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
        out_h = (in_h - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
        out_w = (in_w - 1) * stride[2] - 2 * padding[2] + kernel_size[2]
    else:
        out_d, out_h, out_w = output_size

    output = torch.zeros(
        (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
    )
    numel = input.numel()
    if numel == 0:
        return output

    in_plane_size = in_d * in_h * in_w
    out_plane_size = out_d * out_h * out_w
    with torch_device_fn.device(input.device):
        _max_unpool3d_kernel[(triton.cdiv(numel, 256),)](
            input,
            indices,
            output,
            in_plane_size,
            out_plane_size,
            numel,
            BLOCK_SIZE=256,
        )
    return output


# ---------------------------------------------------------------------------
# backward (gather the output gradient at the saved pooling indices)
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _max_unpool3d_backward_kernel(
    grad_ptr,
    indices_ptr,
    grad_in_ptr,
    in_plane,
    out_plane,
    gnumel,
    numel,
    INT64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """XPU-safe gather: grad_input[j] = grad_output[indices[j]] per (n, c) plane.

    The upstream helper stages ``grad_output`` and selects with ``tl.gather``,
    which does not legalize on XPU3 (``ConvertTritonToTritonXPU`` fails on
    ``tt.gather``), so the vendor side keeps a plain gather.  Every address is
    clamped inside its allocation instead of relying on the mask: a masked-off
    lane on this backend still issues the access, and an out-of-allocation
    address faults the card (status 700).
    """
    pid = tl.program_id(0)
    if INT64:
        offsets = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    else:
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    off_safe = tl.minimum(offsets, numel - 1)
    idx = tl.load(indices_ptr + off_safe)
    if INT64:
        idx = idx.to(tl.int64)
    else:
        idx = idx.to(tl.int32)
    src = (offsets // in_plane) * out_plane + idx
    in_range = (idx >= 0) & (idx < out_plane)
    src_safe = tl.minimum(tl.maximum(src, 0), gnumel - 1)
    vals = tl.load(grad_ptr + src_safe)
    vals = tl.where(mask & in_range, vals, 0.0)
    tl.store(grad_in_ptr + off_safe, vals, mask=mask)


_BWD_BLOCK_SIZE = 1024


def _spatial_numel(t: torch.Tensor) -> int:
    numel = 1
    for size in t.shape[-3:]:
        numel *= size
    return numel


def max_unpool3d_backward(grad_output: torch.Tensor, indices: torch.Tensor):
    """Gradient of ``max_unpool3d`` w.r.t. the pooled input.

    Same contract as ``flag_gems.ops.max_unpool3d.max_unpool3d_backward``:
    ``grad_output`` has the unpooled shape and ``indices`` the pooled shape,
    with the indices linear inside each ``(out_d, out_h, out_w)`` plane.
    """
    logger.debug("GEMS_KUNLUNXIN MAX_UNPOOL3D_BACKWARD")
    if grad_output.dim() != indices.dim():
        raise ValueError(
            "max_unpool3d_backward expects grad_output and indices to have the "
            f"same rank, got {grad_output.dim()} and {indices.dim()}"
        )
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if not indices.is_contiguous():
        indices = indices.contiguous()

    in_plane = _spatial_numel(indices)
    out_plane = _spatial_numel(grad_output)
    numel = indices.numel()
    gnumel = grad_output.numel()
    grad_input = torch.empty_like(indices, dtype=grad_output.dtype)
    if numel == 0 or gnumel == 0 or in_plane == 0:
        return grad_input

    int64 = max(numel, gnumel) > 2**31 - 1
    with torch_device_fn.device(grad_output.device):
        _max_unpool3d_backward_kernel[(triton.cdiv(numel, _BWD_BLOCK_SIZE),)](
            grad_output,
            indices,
            grad_input,
            in_plane,
            out_plane,
            gnumel,
            numel,
            INT64=int64,
            BLOCK_SIZE=_BWD_BLOCK_SIZE,
        )
    return grad_input


class _MaxUnpool3dFunction(torch.autograd.Function):
    """Autograd wrapper mirroring the upstream ``MaxUnpool3dFunction``.

    Only the integer indices are needed by the backward, so nothing else is
    saved; the geometry is recomputed from the shapes inside the backward.
    """

    @staticmethod
    def forward(ctx, input, indices, kernel_size, stride, padding, output_size):
        output = _max_unpool3d_forward(
            input, indices, kernel_size, stride, padding, output_size
        )
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return (None,) * 6
        (indices,) = ctx.saved_tensors
        return (max_unpool3d_backward(grad_output, indices), None, None, None, None, None)


def max_unpool3d(
    input: torch.Tensor,
    indices: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    output_size=None,
):
    logger.debug("GEMS_KUNLUNXIN MAX_UNPOOL3D")
    # aten::max_unpool3d(Tensor self, Tensor indices, SymInt[3] output_size,
    #                    int[3] stride, int[3] padding) -> Tensor
    # When invoked through the ATen dispatcher (e.g. F.max_unpool3d under
    # flag_gems.enable()/use_gems()), the kernel receives 5 positional args in
    # ATen order: (self, indices, output_size, stride, padding) -- i.e. the
    # output_size lands in our `kernel_size` slot and `output_size` is None.
    # Rewrite to the internal convention: (input, indices, kernel_size, stride,
    # padding, output_size), keeping the direct 6-arg functional form intact.
    if (
        output_size is None
        and not isinstance(kernel_size, int)
        and isinstance(kernel_size, (tuple, list))
        and len(kernel_size) == 3
    ):
        output_size = tuple(int(k) for k in kernel_size)
        kernel_size = None

    if torch.is_grad_enabled() and input.requires_grad:
        return _MaxUnpool3dFunction.apply(
            input, indices, kernel_size, stride, padding, output_size
        )
    return _max_unpool3d_forward(
        input, indices, kernel_size, stride, padding, output_size
    )


def _install_xpu_backward_on_common_module():
    """Point the common module's public backward at the XPU-safe kernel.

    ``flag_gems.ops.max_unpool3d.max_unpool3d_backward`` is a public symbol
    imported directly by the accuracy tests, but its small-problem path uses
    ``tl.gather``, which does not legalize on XPU3.  The vendor backend is
    loaded after ``flag_gems.ops``, so replacing the symbols here (same
    import-time patch pattern as ``_patch_sobol_engine_ff_ops_module``) makes
    every later ``from flag_gems.ops.max_unpool3d import ...`` see the XPU
    implementation.
    """
    import sys

    common = sys.modules.get("flag_gems.ops.max_unpool3d")
    if common is None:
        return
    common.max_unpool3d_backward = max_unpool3d_backward
    common.MaxUnpool3dFunction = _MaxUnpool3dFunction


_install_xpu_backward_on_common_module()
