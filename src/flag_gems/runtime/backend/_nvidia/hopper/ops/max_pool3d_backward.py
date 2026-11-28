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

"""Triton implementation of the max_pool3d input gradient.

``aten::max_pool3d_with_indices_backward`` is the registered backward of
``aten::max_pool3d``.  The forward stores one int64 in-plane linear offset per
pooled element, so the gradient maps each pooled element onto exactly one input
voxel; overlapping windows therefore accumulate additively into the same voxel.

Two device kernels implement that map, and ``run`` dispatches between them on
measured grounds:

``max_pool3d_backward_scatter``  walks the pooled tensor once: one int64 index
read plus one gradient read per pooled element, accumulated with a relaxed
atomic add straight into the output-dtype buffer.  Its work is proportional to
``numel_out`` and it touches each real contribution exactly once.

``max_pool3d_backward_gather``  inverts the map and walks the forward *input*
once.  Each voxel is written by exactly one program with one plain store, so it
needs neither atomics nor a zero-initialisation pass.  Its cost, however, is
proportional to ``numel_in`` times the ``ceil(k/s)**3`` candidate windows per
axis, since only ``in_plane/out_plane`` of the input voxels are ever selected.

On sm_90 there is no native half-width atomic datapath, so ptxas rewrites a
16-bit ``atomic_add`` into a CAS retry loop.  That penalty is what makes the
atomic-free gather the faster of the two for small 16-bit problems, while for
larger problems the gather's amplified int64 index traffic dominates and the
scatter wins.  ``_GATHER_MAX_INPUT`` records the measured crossover.

Only first-order gradients are provided, matching the FlagGems operator.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.max_pool3d import _MaxPool3DFunction, _parse_pool3d_params
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Measured crossover between the two kernels for 16-bit inputs: the gather wins
# at 0.60M / 0.80M input voxels and loses at 1.61M / 3.21M.
_GATHER_MAX_INPUT = 1_200_000


_SCATTER_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
]

_GATHER_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
]


# The scatter accumulates with atomic_add, so autotune's repeated trial launches
# would multiply the result.  ``reset_to_zero`` installs Triton's own pre-hook,
# which zeroes the output before every trial *and* once more with
# ``reset_only=True`` immediately before the final production launch (verified
# against the installed Autotuner.run source), so the returned buffer holds
# exactly one accumulation.
@libentry()
@triton.autotune(
    configs=_SCATTER_CONFIGS, key=["total"], reset_to_zero=["grad_input_ptr"]
)
@triton.jit
def max_pool3d_backward_scatter(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    out_plane,
    in_plane,
    total,
    LARGE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if LARGE:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
            0, BLOCK_SIZE
        ).to(tl.int64)
        total = total.to(tl.int64)
        out_plane = out_plane.to(tl.int64)
        in_plane = in_plane.to(tl.int64)
    else:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    plane_start = (offsets // out_plane) * in_plane
    index = tl.load(indices_ptr + offsets, mask=mask, other=-1)
    grad = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    tl.atomic_add(
        grad_input_ptr + plane_start + index,
        grad.to(grad_input_ptr.dtype.element_ty),
        mask=mask & (index >= 0),
        sem="relaxed",
    )


@libentry()
@triton.autotune(configs=_GATHER_CONFIGS, key=["total_in"])
@triton.jit
def max_pool3d_backward_gather(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    total_in,
    IN_PLANE: tl.constexpr,
    OUT_PLANE: tl.constexpr,
    IW: tl.constexpr,
    IH: tl.constexpr,
    OW: tl.constexpr,
    OH: tl.constexpr,
    OD: tl.constexpr,
    PD: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    SD: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    DD: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    CD: tl.constexpr,
    CH: tl.constexpr,
    CW: tl.constexpr,
    LARGE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if LARGE:
        flat = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(
            tl.int64
        )
        total_in = total_in.to(tl.int64)
    else:
        flat = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = flat < total_in

    ihw = IH * IW
    plane = flat // IN_PLANE
    target = flat - plane * IN_PLANE
    ind = target // ihw
    rem = target - ind * ihw
    inh = rem // IW
    inw = rem - inh * IW
    out_base = plane * OUT_PLANE

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for jd in tl.static_range(CD):
        od = (ind + PD) // SD - jd
        rd = (ind + PD) - od * SD
        if DD == 1:
            okd = (rd < KD) & (od >= 0) & (od < OD)
        else:
            okd = (rd < KD * DD) & (rd % DD == 0) & (od >= 0) & (od < OD)
        for jh in tl.static_range(CH):
            oh = (inh + PH) // SH - jh
            rh = (inh + PH) - oh * SH
            if DH == 1:
                okh = okd & (rh < KH) & (oh >= 0) & (oh < OH)
            else:
                okh = okd & (rh < KH * DH) & (rh % DH == 0) & (oh >= 0) & (oh < OH)
            for jw in tl.static_range(CW):
                ow = (inw + PW) // SW - jw
                rw = (inw + PW) - ow * SW
                if DW == 1:
                    okw = okh & (rw < KW) & (ow >= 0) & (ow < OW)
                else:
                    okw = okh & (rw < KW * DW) & (rw % DW == 0) & (ow >= 0) & (ow < OW)
                off = out_base + (od * OH + oh) * OW + ow
                index = tl.load(indices_ptr + off, mask=okw, other=-1)
                if LARGE:
                    sel = index == target
                else:
                    sel = index.to(tl.int32) == target
                grad = tl.load(grad_output_ptr + off, mask=okw, other=0.0)
                acc += tl.where(sel, grad.to(tl.float32), 0.0)

    tl.store(
        grad_input_ptr + flat,
        acc.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


def _parse_pool_param(param, name, default):
    if param is None or (isinstance(param, (list, tuple)) and len(param) == 0):
        return default
    if isinstance(param, int):
        return (param, param, param)
    if isinstance(param, (list, tuple)) and len(param) == 3:
        return tuple(param)
    raise ValueError(f"Invalid {name}: {param}")


def max_pool3d_with_indices_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode: bool = False,
    indices: torch.Tensor = None,
) -> torch.Tensor:
    """Backward pass for 3-D max pooling, matching the ATen signature.

    ``aten::max_pool3d_with_indices_backward(Tensor grad_output, Tensor self,
    int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation,
    bool ceil_mode, Tensor indices) -> Tensor``.

    The window arguments only select the launch geometry: the saved ``indices``
    already encode the window that produced every output element, so the same
    operators serve any kernel/stride/padding/dilation/ceil_mode combination.
    Dtype and layout follow the native contract: the result has ``self``'s dtype
    and is NCDHW-contiguous, like the generic FlagGems wrapper.
    """
    logger.debug("GEMS MAX_POOL3D_WITH_INDICES_BACKWARD (HOPPER)")
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = _parse_pool3d_params(
        kernel_size, stride, padding, dilation
    )
    if min(sd, sh, sw) <= 0 or min(dd, dh, dw) <= 0 or min(pd, ph, pw) < 0:
        raise ValueError("invalid pooling window")

    out_dtype = self.dtype
    total_in = self.numel()
    if total_in == 0:
        return torch.zeros(self.shape, dtype=out_dtype, device=self.device)

    in_d, in_h, in_w = self.shape[2], self.shape[3], self.shape[4]
    out_plane = grad_output.shape[2] * grad_output.shape[3] * grad_output.shape[4]
    in_plane = in_d * in_h * in_w
    total = grad_output.numel()

    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    # The atomic-free gather wins for small 16-bit problems, where sm_90 lowers a
    # 16-bit atomic_add to a ptxas CAS retry loop; beyond the measured crossover
    # its amplified index traffic dominates and the output-dtype scatter wins.
    use_gather = (
        out_dtype in (torch.float16, torch.bfloat16)
        and total_in <= _GATHER_MAX_INPUT
        and total > 0
    )
    if use_gather:
        grad_input = torch.empty(self.shape, dtype=out_dtype, device=self.device)
        grid = lambda meta: (triton.cdiv(total_in, meta["BLOCK_SIZE"]),)
        max_pool3d_backward_gather[grid](
            grad_output,
            indices,
            grad_input,
            total_in,
            IN_PLANE=in_plane,
            OUT_PLANE=out_plane,
            IW=in_w,
            IH=in_h,
            OW=grad_output.shape[4],
            OH=grad_output.shape[3],
            OD=grad_output.shape[2],
            PD=pd,
            PH=ph,
            PW=pw,
            SD=sd,
            SH=sh,
            SW=sw,
            DD=dd,
            DH=dh,
            DW=dw,
            KD=kd,
            KH=kh,
            KW=kw,
            CD=(kd - 1) * dd // sd + 1,
            CH=(kh - 1) * dh // sh + 1,
            CW=(kw - 1) * dw // sw + 1,
            LARGE=total_in > 2**31 - 1,
        )
        return grad_input

    grad_input = torch.zeros(self.shape, dtype=out_dtype, device=self.device)
    if total == 0:
        return grad_input
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
    max_pool3d_backward_scatter[grid](
        grad_output,
        indices,
        grad_input,
        out_plane,
        in_plane,
        total,
        LARGE=total > 2**31 - 1 or total_in > 2**31 - 1,
    )
    return grad_input


def max_pool3d_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode: bool = False,
    indices: torch.Tensor = None,
) -> torch.Tensor:
    """Direct backward entry with the ``max_pool3d_backward`` name.

    Same computation as :func:`max_pool3d_with_indices_backward`; the name
    mirrors ``max_pool2d_backward`` for callers that have the values forward and
    its saved indices.
    """
    return max_pool3d_with_indices_backward(
        grad_output,
        self,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        indices,
    )


class _MaxPool3DBackwardFunction(_MaxPool3DFunction):
    """``max_pool3d`` autograd node wired to the flat scatter above.

    The forward is inherited unchanged, so the values, the saved tensors and the
    metadata stay identical to the generic entry point; only the gradient kernel
    is replaced.
    """

    @staticmethod
    def backward(ctx, grad_output):
        self, indices = ctx.saved_tensors
        kernel_size, stride, padding, dilation, ceil_mode = ctx.pool_params
        return (
            max_pool3d_with_indices_backward(
                grad_output,
                self,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
                indices,
            ),
            None,
            None,
            None,
            None,
            None,
        )


def max_pool3d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Compute 3-D max pooling with the Hopper backward attached.

    The pooled values come from the unchanged generic ``max_pool3d_with_indices``
    forward; only the autograd node is specialized so that
    ``torch.autograd.grad`` reaches the scatter kernel above.
    """
    logger.debug("GEMS MAX_POOL3D (HOPPER)")
    return _MaxPool3DBackwardFunction.apply(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )
