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

"""Kunlunxin (XPU) implementation of ``replication_pad1d_backward``.

The generic kernel does not compile on XPU3:

    RuntimeError: PassManager::run failed: loc("right_start"): operand #1
    does not dominate this use

The bounds of the edge-accumulation loops (``left_stop`` / ``right_start``)
are computed inside data-dependent ``if W_in >= k`` regions, and the Triton
XPU legalize pass rejects a loop bound defined in a nested control-flow region.
There is also a second, non-fatal hazard: the generic kernel's scalar
accumulation loops are unbounded at compile time.

This override keeps the same math but is written as a pure *gather* over the
input width: every input column owns a contiguous span of ``grad_output`` and
the span is summed by the lane itself.  There is no scatter, no atomic, no
cross-lane reduction (so no ``tl.sum`` / custom ``tl.reduce``), and the result
is bit-identical for a given shape regardless of scheduling.  Only the two
replicated edge columns have a span longer than one element; the interior is a
straight 1:1 gather.

Span table (``j`` indexes ``grad_output``; forward maps ``j -> clamp(j-pad_left,
0, W_in-1)``):

* ``W_in == 1`` : single column owns ``[0, W_out)``.
* ``w == 0``    : owns ``[0, clamp(pad_left+1, 0)]``.
* ``w == W_in-1``: owns ``[max(pad_left+W_in-1, 0), W_out)``.
* interior ``w`` : owns the single index ``pad_left + w`` when it lands inside
  ``[0, W_out)`` (negative padding crops columns, which then receive 0).

Addresses are clamped instead of masked: XPU3's masked-load ``other`` value is
unreliable (#51), so every load is an unmasked, in-bounds load and the
out-of-range contribution is zeroed with ``tl.where`` on the lane value.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# ATen's error messages use the C++ scalar-type spelling, e.g. "Float"/"Double".
_SCALAR_NAMES = {
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}


def _scalar_name(dtype):
    return _SCALAR_NAMES.get(dtype, str(dtype).replace("torch.", ""))


@triton.jit
def _replication_pad1d_backward_kernel(
    go_ptr,
    gi_ptr,
    W_OUT,
    PAD_LEFT,
    go_s_n,
    go_s_c,
    go_s_w,
    gi_s_n,
    gi_s_c,
    gi_s_w,
    C,
    W_IN: tl.constexpr,
    MAXG: tl.constexpr,
    STATIC: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    blk = tl.program_id(1)
    n = pid // C
    c = pid % C

    base_o = n.to(tl.int64) * go_s_n + c.to(tl.int64) * go_s_c
    base_i = n.to(tl.int64) * gi_s_n + c.to(tl.int64) * gi_s_c

    iw = blk * BLOCK + tl.arange(0, BLOCK)
    mask = iw < W_IN

    # First (possibly replicated) output index for this input column.
    lo_raw = tl.where(
        iw == 0, 0, tl.where(iw == W_IN - 1, PAD_LEFT + W_IN - 1, PAD_LEFT + iw)
    )
    # Every load below uses this clamped base, so it is always in [0, W_OUT).
    lo = tl.minimum(tl.maximum(lo_raw, 0), W_OUT - 1)

    # Number of output positions that fold into this input column.
    cnt = tl.where(
        iw == 0,
        tl.where(W_IN == 1, W_OUT, tl.maximum(PAD_LEFT + 1, 0)),
        tl.where(
            iw == W_IN - 1,
            tl.where(lo_raw >= W_OUT, 0, W_OUT - tl.maximum(lo_raw, 0)),
            tl.where((lo_raw >= 0) & (lo_raw < W_OUT), 1, 0),
        ),
    )
    hi_off = tl.maximum(cnt - 1, 0)

    acc = tl.zeros((BLOCK,), dtype=ACC_DTYPE)
    if STATIC:
        for g in tl.static_range(MAXG):
            gg = tl.minimum(g, hi_off)
            v = tl.load(go_ptr + base_o + (lo + gg).to(tl.int64) * go_s_w)
            acc += tl.where(g < cnt, v.to(ACC_DTYPE), 0)
    else:
        for g in range(0, MAXG):
            gg = tl.minimum(g, hi_off)
            v = tl.load(go_ptr + base_o + (lo + gg).to(tl.int64) * go_s_w)
            acc += tl.where(g < cnt, v.to(ACC_DTYPE), 0)

    tl.store(
        gi_ptr + base_i + iw.to(tl.int64) * gi_s_w,
        acc.to(gi_ptr.dtype.element_ty),
        mask=mask,
    )


def _launch_replication_pad1d_backward_kernel(
    grad_output: torch.Tensor,
    self_tensor: torch.Tensor,
    padding,
    grad_input: torch.Tensor,
):
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    left, right = int(padding[0]), int(padding[1])

    dim = self_tensor.dim()
    if dim not in (2, 3):
        raise ValueError(
            "replication_pad1d_backward expects 2D (C, W) or 3D (N, C, W) input"
        )

    if dim == 3:
        N, C, W_in = self_tensor.shape
        B = N
        grad_in_s_n, grad_in_s_c, grad_in_s_w = grad_input.stride()
        grad_out_s_n, grad_out_s_c, grad_out_s_w = grad_output.stride()
    else:
        C, W_in = self_tensor.shape
        B = 1
        grad_in_s_c, grad_in_s_w = grad_input.stride()
        grad_in_s_n = 0
        if grad_output.dim() == 2:
            grad_out_s_c, grad_out_s_w = grad_output.stride()
            grad_out_s_n = 0
        elif grad_output.dim() == 3:
            grad_out_s_n, grad_out_s_c, grad_out_s_w = grad_output.stride()
        else:
            raise ValueError("grad_output has invalid dimensions")

    W_out = W_in + left + right

    expected_grad_output_shape = (N, C, W_out) if dim == 3 else (C, W_out)
    if tuple(grad_output.shape) != expected_grad_output_shape:
        raise ValueError(
            f"grad_output has incorrect shape. Expected {expected_grad_output_shape}, got {tuple(grad_output.shape)}"
        )

    if tuple(grad_input.shape) != tuple(self_tensor.shape):
        raise ValueError(
            f"grad_input has incorrect shape. Expected {tuple(self_tensor.shape)}, got {tuple(grad_input.shape)}"
        )

    if W_out <= 0:
        raise ValueError("replication_pad1d_backward requires a positive output width")
    rows = B * C
    if rows == 0 or W_in == 0:
        # No input column exists, so the (empty) gradient is already correct.
        return grad_input

    if self_tensor.dtype in (torch.float16, torch.bfloat16):
        acc_dtype = tl.float32
    elif self_tensor.dtype == torch.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32

    maxg = W_out if W_in == 1 else max(left + 1, right + 1, 1)
    static = maxg <= 32

    # XPU3 measures these launch-bound micro shapes best with a fixed 1024-lane
    # tile: narrow tiles leave most of the cluster's vector lanes idle and cost
    # ~3x more than a full-width tile per program (measured on the benchmark
    # matrix; see harness/solution/replication_pad1d_backward/README.md).
    BLOCK = 1024
    grid = (rows, triton.cdiv(int(W_in), BLOCK))

    with torch_device_fn.device(grad_output.device):
        _replication_pad1d_backward_kernel[grid](
            grad_output,
            grad_input,
            W_out,
            left,
            grad_out_s_n,
            grad_out_s_c,
            grad_out_s_w,
            grad_in_s_n,
            grad_in_s_c,
            grad_in_s_w,
            C,
            W_IN=int(W_in),
            MAXG=maxg,
            STATIC=static,
            BLOCK=BLOCK,
            ACC_DTYPE=acc_dtype,
        )
    return grad_input


def replication_pad1d_backward(
    grad_output: torch.Tensor, self_tensor: torch.Tensor, padding
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN REPLICATION_PAD1D_BACKWARD")
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    left, right = int(padding[0]), int(padding[1])

    dim = self_tensor.dim()
    if dim not in (2, 3):
        raise ValueError(
            "replication_pad1d_backward expects 2D (C, W) or 3D (N, C, W) input"
        )

    # ATen rejects a gradient whose dtype differs from the input rather than
    # promoting it, with this wording.
    if grad_output.dtype != self_tensor.dtype:
        raise RuntimeError(
            f"expected scalar type {_scalar_name(self_tensor.dtype)} but found "
            f"{_scalar_name(grad_output.dtype)}"
        )
    if grad_output.device != self_tensor.device:
        raise RuntimeError(
            f"expected grad_output to be on device {self_tensor.device} but found "
            f"{grad_output.device}"
        )

    # The output takes the *input's* options, not the gradient's: ATen returns
    # ``at::empty_like(self)``.
    grad_input = torch.empty(
        self_tensor.shape,
        device=self_tensor.device,
        dtype=self_tensor.dtype,
    )

    if self_tensor.is_complex():
        # Triton has no complex scalar type.  This operator only gathers and sums
        # gradients along W, so run the real kernel once per component on a
        # ``view_as_real`` slice: slicing the last axis keeps the W layout (only
        # the element stride changes), which the kernel already takes as a
        # parameter.
        go_real = torch.view_as_real(grad_output)
        self_real = torch.view_as_real(self_tensor)
        gi_real = torch.view_as_real(grad_input)
        for comp in (0, 1):
            _launch_replication_pad1d_backward_kernel(
                go_real[..., comp],
                self_real[..., comp],
                (left, right),
                gi_real[..., comp],
            )
        return grad_input

    return _launch_replication_pad1d_backward_kernel(
        grad_output, self_tensor, (left, right), grad_input
    )
