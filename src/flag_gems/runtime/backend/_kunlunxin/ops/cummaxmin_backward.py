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

Tensor = torch.Tensor

logger = logging.getLogger(__name__)


@triton.jit
def _seg_fwd_combine(a_key, a_val, b_key, b_val):
    # Segmented-sum combine over a monotonically non-decreasing key sequence.
    # Keys are equal within a segment, so reset the accumulator on a key change.
    same = a_key == b_key
    return b_key, b_val + tl.where(same, a_val, 0.0)


@triton.jit
def _cummaxmin_backward_scan_kernel(
    grad_ptr,
    indices_ptr,
    grad_input_ptr,
    reduction_size,
    inner_size,
    red_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward for cummax/cummin via an in-program segmented scan.

    Single-kernel variant used for small reduction sizes: scan the column in
    *reversed* order and write the segment total directly back to the (reversed)
    head position. The reversed store is inexpensive for small ``reduction_size``
    and avoids the extra scratch tensor / second kernel of the two-pass variant.

    ``scatter_add_`` collapses to an affine store because, for cummax/cummin,
    ``indices[i] == i`` at every segment head, so the scatter target equals the
    segment-head position.
    """
    row = tl.program_id(0)
    outer = row // inner_size
    inner = row % inner_size
    base = outer * (reduction_size * inner_size) + inner
    grad_base = grad_ptr + base
    indices_base = indices_ptr + base
    grad_input_base = grad_input_ptr + base

    r = tl.arange(0, BLOCK_SIZE)
    mask = r < reduction_size
    o = reduction_size - 1 - r
    off_o = o * red_stride
    g = tl.load(grad_base + off_o, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(indices_base + off_o, mask=mask, other=0).to(tl.int32)

    key, fwd = tl.associative_scan((k, g), 0, _seg_fwd_combine)

    nxt_pos = tl.maximum(o - 1, 0)
    nxt_off = nxt_pos * red_stride
    nxt = tl.load(indices_base + nxt_off, mask=mask, other=0).to(tl.int32)
    is_head = (r == reduction_size - 1) | (k != nxt)

    val = tl.where(is_head, fwd, 0.0)
    tl.store(grad_input_base + off_o, val, mask=mask)


@triton.jit
def _cummaxmin_backward_scan_store_fwd_kernel(
    grad_ptr,
    indices_ptr,
    tmp_ptr,
    reduction_size,
    inner_size,
    red_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward for cummax/cummin: reversed-load scan, forward store to scratch.

    First pass of the two-pass variant. The column is loaded in *reversed* order
    and scanned forward, which accumulates each segment total at the segment's
    original head position (the run end in reversed order). The head values are
    stored with a *forward* (coalesced) store into ``tmp`` at the reversed lane
    index ``r``, deliberately avoiding a reversed store to global memory, which
    is extremely slow on this backend.
    """
    row = tl.program_id(0)
    outer = row // inner_size
    inner = row % inner_size
    base = outer * (reduction_size * inner_size) + inner
    grad_base = grad_ptr + base
    indices_base = indices_ptr + base
    tmp_base = tmp_ptr + base

    r = tl.arange(0, BLOCK_SIZE)
    mask = r < reduction_size
    o = reduction_size - 1 - r
    off_o = o * red_stride
    g = tl.load(grad_base + off_o, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(indices_base + off_o, mask=mask, other=0).to(tl.int32)

    key, fwd = tl.associative_scan((k, g), 0, _seg_fwd_combine)

    nxt_pos = tl.maximum(o - 1, 0)
    nxt_off = nxt_pos * red_stride
    nxt = tl.load(indices_base + nxt_off, mask=mask, other=0).to(tl.int32)
    is_head = (r == reduction_size - 1) | (k != nxt)

    val = tl.where(is_head, fwd, 0.0)
    tl.store(tmp_base + r, val, mask=mask)


@triton.jit
def _cummaxmin_backward_reverse_copy_kernel(
    src_ptr,
    dst_ptr,
    reduction_size,
    inner_size,
    red_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Second pass of the two-pass variant: un-reverse the scratch buffer.

    ``tmp[r]`` holds the segment total for original head position
    ``reduction_size - 1 - r``. Loading with a reversed offset and storing with a
    forward offset places each total at its original head position in ``dst``.
    """
    row = tl.program_id(0)
    outer = row // inner_size
    inner = row % inner_size
    base = outer * (reduction_size * inner_size) + inner
    src_base = src_ptr + base
    dst_base = dst_ptr + base

    r = tl.arange(0, BLOCK_SIZE)
    mask = r < reduction_size
    o = reduction_size - 1 - r
    v = tl.load(src_base + o * red_stride, mask=mask, other=0.0)
    tl.store(dst_base + r * red_stride, v, mask=mask)


@triton.jit
def _cummaxmin_backward_atomic_kernel(
    grad_ptr,
    indices_ptr,
    grad_input_ptr,
    reduction_size,
    inner_size,
    red_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward for cummax/cummin via intra-program atomic accumulation.

    One program handles a single ``(outer, inner)`` column and loops over the
    whole reduction axis, so all scatter-add accumulation for a target element
    happens inside one program (no cross-program atomic, which is unreliable on
    this backend). Used for non-power-of-two reduction sizes.
    """
    row = tl.program_id(0)
    outer = row // inner_size
    inner = row % inner_size
    base = outer * (reduction_size * inner_size) + inner
    grad_base = grad_ptr + base
    indices_base = indices_ptr + base
    grad_input_base = grad_input_ptr + base
    for block_start in range(0, reduction_size, BLOCK_SIZE):
        pos = block_start + tl.arange(0, BLOCK_SIZE)
        mask = pos < reduction_size
        off = pos * red_stride
        g = tl.load(grad_base + off, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(indices_base + off, mask=mask, other=0).to(tl.int32)
        tl.atomic_add(grad_input_base + t * red_stride, g, mask=mask)


def cummaxmin_backward(
    grad_output: Tensor, input: Tensor, indices: Tensor, dim: int
) -> Tensor:
    """Backward pass for cummax/cummin operations.

    Implements the scatter-add semantic ``grad_input.scatter_add_(dim, indices,
    grad_output)``: each output-position gradient flows back to the input element
    that produced it (given by ``indices`` from the forward pass). Repeated
    indices accumulate. Accumulation is performed in float32 for numerical
    stability and cast back to the input dtype.

    Routing:

    * Power-of-two reduction size <= 128: single-kernel reversed scan with an
      affine (reversed) store.
    * Power-of-two reduction size > 128: two-pass reversed scan (reversed load +
      forward store to scratch, then a reverse copy), which avoids the very slow
      reversed store to global memory that dominates larger sizes.
    * Otherwise: intra-program atomic accumulation (correct for arbitrary, e.g.
      non-power-of-two, reduction sizes).
    """
    logger.debug("GEMS_KUNLUNXIN CUMMAXMIN_BACKWARD")

    ndim = grad_output.ndim
    if dim < 0:
        dim = dim + ndim

    shape = list(grad_output.shape)
    reduction_size = shape[dim]

    grad_c = grad_output.contiguous()
    # Segment keys are reduction positions (< 2**31), so int32 is sufficient and
    # halves the (reversed) index traffic versus the int64 indices produced by
    # the cummax/cummin forward pass.
    indices_c = indices.contiguous().to(torch.int32)

    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= shape[i]
    outer_size = 1
    for i in range(dim):
        outer_size *= shape[i]

    # Float32 accumulator matching the reference upcast-accumulate semantics.
    grad_input_f32 = torch.zeros(shape, dtype=torch.float32, device=grad_output.device)

    num_rows = outer_size * inner_size
    grid = (num_rows,)
    red_stride = inner_size

    use_scan = reduction_size == triton.next_power_of_2(max(reduction_size, 1))

    with torch_device_fn.device(grad_output.device):
        if use_scan:
            BLOCK_SIZE = triton.next_power_of_2(max(reduction_size, 1))
            if reduction_size <= 128:
                _cummaxmin_backward_scan_kernel[grid](
                    grad_c,
                    indices_c,
                    grad_input_f32,
                    reduction_size,
                    inner_size,
                    red_stride,
                    BLOCK_SIZE,
                )
            else:
                tmp = torch.empty(shape, dtype=torch.float32, device=grad_output.device)
                _cummaxmin_backward_scan_store_fwd_kernel[grid](
                    grad_c,
                    indices_c,
                    tmp,
                    reduction_size,
                    inner_size,
                    red_stride,
                    BLOCK_SIZE,
                )
                _cummaxmin_backward_reverse_copy_kernel[grid](
                    tmp,
                    grad_input_f32,
                    reduction_size,
                    inner_size,
                    red_stride,
                    BLOCK_SIZE,
                )
        else:
            BLOCK_SIZE = min(triton.next_power_of_2(max(reduction_size, 1)), 1024)
            _cummaxmin_backward_atomic_kernel[grid](
                grad_c,
                indices_c,
                grad_input_f32,
                reduction_size,
                inner_size,
                red_stride,
                BLOCK_SIZE,
            )

    return grad_input_f32.to(grad_output.dtype)
