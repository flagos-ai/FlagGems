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

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _diag_scatter_1d(
    out_ptr,
    src_ptr,
    inner,
    diag_len,
    total,
    diag_off,
    diag_stride,
    BS: tl.constexpr,
):
    """Write src[bid * diag_len + k] to out[bid * inner + diag_off + k * diag_stride].

    The output is C-contiguous with the 2-D sub-matrix as its last two dims, so
    every diagonal element is at a constant integer stride (diag_stride) from
    the previous one and the batch offset is a multiple of `inner`.  The store
    addresses are a single fused multiply-add over the 1-D arange-derived index
    (no per-lane tl.where / div by a runtime value), which is what keeps this
    as one affine store per lane on the Kunlunxin backend.
    """
    pid = tl.program_id(0)
    idx = pid * BS + tl.arange(0, BS)
    m = idx < total
    bid = idx // diag_len
    k = idx % diag_len
    val = tl.load(src_ptr + idx, mask=m)
    tl.store(out_ptr + bid * inner + diag_off + k * diag_stride, val, mask=m)


def diagonal_scatter(input, src, offset=0, dim1=0, dim2=1):
    """Scatter src values into the diagonal of a clone of input.

    Only the diagonal elements (batch_size * diag_len) are written by the
    kernel; everything else comes from a device-side clone of `input`.  When
    the 2-D sub-matrix is already the last two dims (the common case) the
    clone is used directly and no permutation/contiguous round-trip happens.
    """
    logger.debug("GEMS_KUNLUNXIN DIAGONAL_SCATTER")

    ndim = input.ndim
    if dim1 < 0:
        dim1 += ndim
    if dim2 < 0:
        dim2 += ndim

    row_size = input.shape[dim1]
    col_size = input.shape[dim2]
    if offset >= 0:
        diag_len = max(0, min(row_size, col_size - offset))
    else:
        diag_len = max(0, min(row_size + offset, col_size))

    inner = row_size * col_size

    if dim1 == ndim - 2 and dim2 == ndim - 1:
        # Fast path: flat C-order indexing applies directly to the clone.
        output = input.clone()
        if not output.is_contiguous():
            output = output.contiguous()
    else:
        # General path: bring the sub-matrix to the last two dims, scatter,
        # then permute back to the original layout.
        perm = [i for i in range(ndim) if i != dim1 and i != dim2] + [dim1, dim2]
        output = input.permute(perm).contiguous()

    if diag_len > 0 and output.numel() > 0:
        src_c = src.contiguous()
        total = (output.numel() // inner) * diag_len
        if offset >= 0:
            diag_off = offset
        else:
            diag_off = -offset * col_size
        diag_stride = col_size + 1
        BS = 1024
        grid = (triton.cdiv(total, BS),)
        _diag_scatter_1d[grid](
            output, src_c, inner, diag_len, total, diag_off, diag_stride, BS=BS
        )

    if dim1 == ndim - 2 and dim2 == ndim - 1:
        return output
    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return output.permute(inv_perm).contiguous()
