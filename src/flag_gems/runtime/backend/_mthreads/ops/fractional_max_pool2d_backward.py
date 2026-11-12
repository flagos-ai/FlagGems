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

"""Fractional max-pool 2d BACKWARD: fused zero + atomic scatter.

Semantics (matches ATen / autograd):
    grad_input[n,c,h,w] = sum of grad_output[n,c,ph,pw] over every output
    position whose forward argmax index equals (h*W + w) in that plane, with
    untouched positions exactly zero.

The forward indices are int64 in-plane flat offsets (h*W + w), so the backward
is a per-plane scatter-accumulate; kernel_size / output_size carry no geometry
and only satisfy the ABI.

Paths:
  * Fused single launch (used when a plane is small enough that a program can
    own several whole planes, i.e. in_plane <= MAX_PLANE_ELEMS): program p owns
    the contiguous input range of planes [p*G, p*G+own) -- it zeroes that range,
    executes tl.debug_barrier(), then atomically accumulates only the output
    entries of those same planes.  Plane ranges are disjoint across programs, so
    no cross-program ordering is needed and the call costs one launch.  The
    per-element plane index is derived from the flat output offset.
  * Two launches (large planes): a flat masked zero store over all input
    elements followed by a flat atomic scatter.  Programs there cannot own
    whole planes without leaving whole SMs idle, so zeroing stays separate.

Both grids are 1-D and fully occupied: a per-plane grid of one program per
(plane, out-tile) pair would collapse to mostly-empty programs for the
small-plane shapes that dominate these workloads.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.fractional_max_pool2d import (
    fractional_max_pool2d_backward as default_fractional_max_pool2d_backward,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

# A fused program owns about this many input elements (a few vector iterations
# of the zero store, and a matching amount of output scanning).
CHUNK_ELEMS = 4096
# Never let the fused grid drop below this many programs if whole planes allow.
MIN_CTAS = 120
# Largest plane handled by the fused path.
MAX_PLANE_ELEMS = 8192


@triton.jit
def _fused_kernel(
    g_ptr,
    idx_ptr,
    out_ptr,
    n_planes,
    out_plane,
    in_plane,
    G,
    BLOCK_Z: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    pid = tl.program_id(0)
    p0 = pid * G
    own = tl.minimum(G, n_planes - p0)
    in_lo = p0 * in_plane
    in_hi = in_lo + own * in_plane

    zero = tl.zeros((BLOCK_Z,), dtype=out_ptr.dtype.element_ty)
    for s in range(in_lo, in_hi, BLOCK_Z):
        offs = s + tl.arange(0, BLOCK_Z)
        tl.store(out_ptr + offs, zero, mask=offs < in_hi)
    tl.debug_barrier()

    o_lo = p0 * out_plane
    o_hi = o_lo + own * out_plane
    for s in range(o_lo, o_hi, BLOCK_O):
        o = s + tl.arange(0, BLOCK_O)
        mask = o < o_hi
        idx = tl.load(idx_ptr + o, mask=mask, other=-1).to(tl.int32)
        g = tl.load(g_ptr + o, mask=mask, other=0.0).to(out_ptr.dtype.element_ty)
        plane = (o - o_lo) // out_plane
        m = mask & (idx >= 0) & (idx < in_plane)
        tl.atomic_add(
            out_ptr + (in_lo + plane * in_plane + idx),
            g,
            mask=m,
            sem="relaxed",
            scope="cta",
        )


@triton.jit
def _zero_fill_kernel(out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    zero = tl.zeros((BLOCK,), dtype=out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, zero, mask=offs < n_elements)


@triton.jit
def _scatter_add_kernel(
    g_ptr, idx_ptr, out_ptr, total_out, out_plane, in_plane, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_out
    idx = tl.load(idx_ptr + offs, mask=mask, other=-1).to(tl.int32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(out_ptr.dtype.element_ty)
    plane = offs // out_plane
    m = mask & (idx >= 0) & (idx < in_plane)
    tl.atomic_add(out_ptr + (plane * in_plane + idx), g, mask=m, sem="relaxed")


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def fractional_max_pool2d_backward(
    grad_output, self, kernel_size=None, output_size=None, indices=None
):
    logger.debug("GEMS_MTHREADS FRACTIONAL_MAX_POOL2D_BACKWARD")

    if not _use_triton_kernel(self) or not _use_triton_kernel(grad_output):
        return default_fractional_max_pool2d_backward(
            grad_output, self, kernel_size, output_size, indices
        )

    with torch_device_fn.device(self.device):
        return _launch_fractional_max_pool2d_backward(
            grad_output, self, kernel_size, output_size, indices
        )


def _launch_fractional_max_pool2d_backward(
    grad_output, self, kernel_size=None, output_size=None, indices=None
):
    go = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    idx = indices if indices.is_contiguous() else indices.contiguous()

    in_plane = self.shape[-1] * self.shape[-2]
    out_plane = idx.shape[-1] * idx.shape[-2]
    grad_input = torch.empty(self.shape, dtype=self.dtype, device=self.device)

    total_in = grad_input.numel()
    if total_in == 0:
        return grad_input

    total_out = idx.numel()
    if total_out == 0 or in_plane == 0:
        _zero_fill_kernel[(triton.cdiv(total_in, 4096),)](
            grad_input, total_in, BLOCK=4096, num_warps=4
        )
        return grad_input

    n_planes = total_in // in_plane
    if in_plane <= MAX_PLANE_ELEMS and total_out == n_planes * out_plane:
        # One program owns whole planes: about CHUNK_ELEMS input elements each,
        # unless that would leave fewer than MIN_CTAS programs.
        g_chunk = CHUNK_ELEMS // in_plane
        cap = n_planes // MIN_CTAS
        if cap < 1:
            cap = 1
        if g_chunk < 1:
            g_chunk = 1
        G = g_chunk if g_chunk < cap else cap
        _fused_kernel[(triton.cdiv(n_planes, G),)](
            go,
            idx,
            grad_input,
            n_planes,
            out_plane,
            in_plane,
            G,
            BLOCK_Z=2048,
            BLOCK_O=512,
            num_warps=4,
        )
        return grad_input

    _zero_fill_kernel[(triton.cdiv(total_in, 4096),)](
        grad_input, total_in, BLOCK=4096, num_warps=4
    )
    _scatter_add_kernel[(triton.cdiv(total_out, 1024),)](
        go, idx, grad_input, total_out, out_plane, in_plane, BLOCK=1024, num_warps=4
    )
    return grad_input
