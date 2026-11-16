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
#
# Kunlunxin override for replication_pad1d_backward.
#
# The generic implementation launches one program per (N, C) slice and drives
# the edge gradients with *scalar* `for` loops whose bounds come from
# `min`/`max` on runtime padding values.  On this backend the right-edge lower
# bound (`scf.for %i = maxsi(...) to W_out`) makes `TritonXPULegalize` fail with
# "operand #1 does not dominate this use" (raised as
# ``OutOfResources: uni_sram ... PassManager::run failed``) for several padding
# specializations, so the op cannot run at all.
#
# Single-launch design (one program per (n, c) slice, one vector store per row):
#   * edge gradients: the left span ``grad_out[0 : pad_l + 1]`` and the right
#     span ``grad_out[pad_l + W_in - 1 : W_out]`` are summed with one unmasked,
#     contiguous load each plus ``tl.sum``.  Unmasked clamped indexing avoids
#     masked-load semantics; the sum is widened to ``MAXG >= 64`` lanes because
#     ``tl.sum`` over fewer than 64 lanes is silently miscompiled on this
#     backend (small-vector reduction caveat).
#   * interior columns: ``grad_out[clamp(pad_l + w)]`` (zero when the mapping is
#     cropped by negative padding), vectorized over ``BLOCK`` columns.
#   * both edge values are injected into that same row vector, so every write
#     is a full-width contiguous store.  A separate 2-element edge store per
#     program was measured at ~12us/program on this backend (scattered 2-lane
#     stores serialize across programs), which dominated the whole operator.
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
    W_in,
    W_out,
    pad_l,
    go_stride_n,
    go_stride_c,
    go_stride_w,
    gi_stride_n,
    gi_stride_c,
    gi_stride_w,
    BLOCK: tl.constexpr,
    ACC: tl.constexpr,
    MAXG: tl.constexpr,
    SINGLE: tl.constexpr,
):
    """One program per (n, c) slice; one contiguous row store.

    lane r of the edge vectors walks the replicated spans:
      left  -> grad_out[0 : pad_l + 1]            (empty when pad_l < 0)
      right -> grad_out[pad_l + W_in - 1 : W_out] (empty when pad_r < 0)
    Spans that would fall outside the row are replaced by a repeat of a valid
    in-row element (``min(r, cnt - 1)``) and zeroed by the following ``where``,
    so the loads are unmasked and always address in-bounds memory.  ``MAXG`` is
    padded to >= 64 lanes by the launcher: ``tl.sum`` over fewer than 64 lanes
    is silently miscompiled on this backend.
    """
    n = tl.program_id(0)
    c = tl.program_id(1)
    base_out = n.to(tl.int64) * go_stride_n + c.to(tl.int64) * go_stride_c
    base_in = n.to(tl.int64) * gi_stride_n + c.to(tl.int64) * gi_stride_c

    # Left edge: grad_out[0 : pad_l + 1], empty when pad_l < 0.
    cnt_l = tl.minimum(tl.maximum(pad_l + 1, 0), W_out)
    # Right edge: grad_out[pad_l + W_in - 1 : W_out], empty when pad_r < 0.
    lo_raw = pad_l + W_in - 1
    lo_r = tl.minimum(tl.maximum(lo_raw, 0), W_out - 1)
    cnt_r = tl.where(lo_raw >= W_out, 0, W_out - tl.maximum(lo_raw, 0))
    if SINGLE:
        # With W_in == 1 both edges describe column 0 and the whole output row
        # is replicated onto it: one span over [0, W_out).
        cnt_l = W_out
        cnt_r = 0

    r = tl.arange(0, MAXG)
    idx_l = tl.minimum(r, tl.maximum(cnt_l - 1, 0))
    vl = tl.load(go_ptr + base_out + idx_l.to(tl.int64) * go_stride_w)
    left_sum = tl.sum(tl.where(r < cnt_l, vl.to(ACC), 0.0))
    idx_r = lo_r + tl.minimum(r, tl.maximum(cnt_r - 1, 0))
    vr = tl.load(go_ptr + base_out + idx_r.to(tl.int64) * go_stride_w)
    right_sum = tl.sum(tl.where(r < cnt_r, vr.to(ACC), 0.0))

    for start in tl.range(0, W_in, BLOCK):
        w = start + tl.arange(0, BLOCK)
        mask = w < W_in

        # Interior mapping. Columns cropped away by negative padding map
        # outside [0, W_out) and receive no gradient; the address is clamped so
        # the load itself never reads out of bounds.
        k = pad_l + w
        k_lo = tl.where(k < 0, 0, k)
        k_c = tl.where(k_lo > W_out - 1, W_out - 1, k_lo)
        v = tl.load(go_ptr + base_out + k_c.to(tl.int64) * go_stride_w)
        # The interior is a pure copy: keep it in the native dtype (widening
        # 16-bit lanes to fp32 just to round-trip measured slower on XPU).
        v = tl.where((k >= 0) & (k < W_out), v, 0.0)
        # Overwrite the two edge lanes with the replicated-span sums so the row
        # is written by a single contiguous store.
        v = tl.where(w == 0, left_sum.to(v.dtype), v)
        v = tl.where((w == W_in - 1) & (W_in > 1), right_sum.to(v.dtype), v)
        tl.store(
            gi_ptr + base_in + w.to(tl.int64) * gi_stride_w,
            v,
            mask=mask,
        )


def _pad2(padding):
    if isinstance(padding, torch.Tensor):
        padding = tuple(int(p) for p in padding.tolist())
    left, right = int(padding[0]), int(padding[1])
    return left, right


def _edge_span(left, right, W_in, W_out):
    """Upper bound on the number of elements any edge span can cover."""
    if W_in == 1:
        # Both edges collapse onto the single input column: one span over the
        # whole output row.
        return W_out
    span_l = min(max(left + 1, 0), W_out)
    lo_raw = left + W_in - 1
    span_r = W_out if lo_raw < 0 else max(W_out - lo_raw, 0)
    return max(span_l, span_r, 1)


def _launch_kernels(grad_output, grad_input, left, right, acc_dtype):
    dim = grad_input.dim()
    if dim == 3:
        N, C, W_in = grad_input.shape
        gi_sn, gi_sc, gi_sw = grad_input.stride()
        go_sn, go_sc, go_sw = grad_output.stride()
    else:
        C, W_in = grad_input.shape
        N = 1
        gi_sc, gi_sw = grad_input.stride()
        gi_sn = 0
        # The generic op accepts a 3D grad_output for a 2D input (leading unit
        # batch); keep supporting it.
        if grad_output.dim() == 2:
            go_sc, go_sw = grad_output.stride()
            go_sn = 0
        else:
            go_sn, go_sc, go_sw = grad_output.stride()

    W_out = W_in + left + right
    if W_out <= 0:
        raise ValueError(f"replication_pad1d: output spatial dimension is non-positive: " f"output size {W_out}")

    BLOCK = 256
    # Pad the edge-span vector to at least 64 lanes: tl.sum over fewer than 64
    # lanes is silently miscompiled on this backend.
    maxg = 64
    span = _edge_span(left, right, W_in, W_out)
    while maxg < span:
        maxg *= 2

    with torch_device_fn.device(grad_output.device):
        _replication_pad1d_backward_kernel[(N, C)](
            grad_output,
            grad_input,
            W_in,
            W_out,
            left,
            go_sn,
            go_sc,
            go_sw,
            gi_sn,
            gi_sc,
            gi_sw,
            BLOCK=BLOCK,
            ACC=acc_dtype,
            MAXG=maxg,
            SINGLE=W_in == 1,
        )
    return grad_input


def _acc_dtype_for(dtype):
    # Edge gradients are sums over the replicated span, so accumulate in fp32
    # for low-precision dtypes rather than adding in fp16/bf16 element by
    # element.
    if dtype == torch.float64:
        return tl.float64
    if dtype == torch.complex128:
        return tl.float64
    return tl.float32


def replication_pad1d_backward(grad_output: torch.Tensor, self_tensor: torch.Tensor, padding) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN REPLICATION_PAD1D_BACKWARD")
    left, right = _pad2(padding)

    dim = self_tensor.dim()
    if dim not in (2, 3):
        raise ValueError("replication_pad1d_backward expects 2D (C, W) or 3D (N, C, W) input")

    # ATen rejects a gradient whose dtype differs from the input rather than
    # promoting it, with this wording.
    if grad_output.dtype != self_tensor.dtype:
        raise RuntimeError(
            f"expected scalar type {_scalar_name(self_tensor.dtype)} but found " f"{_scalar_name(grad_output.dtype)}"
        )
    if grad_output.device != self_tensor.device:
        raise RuntimeError(
            f"expected grad_output to be on device {self_tensor.device} but found " f"{grad_output.device}"
        )
    W_out = self_tensor.shape[-1] + left + right
    expected_grad_output_shape = tuple(self_tensor.shape[:-1]) + (W_out,)
    if tuple(grad_output.shape) != expected_grad_output_shape:
        raise ValueError(
            f"grad_output has incorrect shape. Expected {expected_grad_output_shape}, "
            f"got {tuple(grad_output.shape)}"
        )

    # The output takes the *input's* options, not the gradient's: ATen returns
    # ``at::empty_like(self)``.
    grad_input = torch.empty(
        self_tensor.shape,
        device=self_tensor.device,
        dtype=self_tensor.dtype,
    )
    if grad_input.numel() == 0:
        return grad_input

    acc_dtype = _acc_dtype_for(self_tensor.dtype)

    if self_tensor.is_complex():
        # Triton has no complex scalar type, and this operator never mixes the
        # real and imaginary parts: it only gathers and sums gradients along W.
        # Run the real kernel once per component on the view_as_real planes.
        # Use the functional torch.view_as_real: torch_xmlir's Tensor object
        # does not expose the ``Tensor.view_as_real`` method.
        go_real = torch.view_as_real(grad_output)
        gi_real = torch.view_as_real(grad_input)
        for comp in (0, 1):
            _launch_kernels(go_real[..., comp], gi_real[..., comp], left, right, acc_dtype)
        return grad_input

    return _launch_kernels(grad_output, self_tensor, left, right, acc_dtype)
