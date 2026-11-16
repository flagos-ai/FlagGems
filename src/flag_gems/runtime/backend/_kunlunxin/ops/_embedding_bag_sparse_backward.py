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
"""Kunlunxin XPU3 override for ``aten::_embedding_bag_sparse_backward``.

Semantics: the sparse (COO) gradient has one non-zero per (unfiltered) sample
and its values are ``grad[offset2bag[i]]`` (divided by ``bag_size`` in mean
mode, multiplied by ``per_sample_weights`` when given).  With ``mode == 0``,
no ``per_sample_weights`` and no ``padding_idx`` every value row is a *copy* of
one ``grad`` row, and all samples of bag ``b`` share the same row -- bag ``b``
occupies the contiguous output rows ``[offsets[b], offsets[b+1])``.

Why the generic implementation is slow on XPU3 (measured,
``harness/perf_ir/embedding_bag_sparse_backward/prof_pre.log``)
-------------------------------------------------------------------------
The generic kernel is "one program per SAMPLES_PER_PROG output rows" with a
masked 512B row load and a masked 512B row store per sample.  On this backend a
512B row DMA is at the *short-DMA cliff* (9 GB/s, ~57-100 ns per descriptor,
`pattern-xpu3-gather-wall-vs-block-dma`), so the cost grows with the output row
count: 0.15 ms (nnz=512) / 0.97 ms (nnz=4096) versus 7.6 / 9.5 us for the vendor
`index_select` gather that ATen uses.  Nothing flattens the store side either:
a 2-D tile store of one broadcast row inside any enclosing ``if`` fails to
legalize (``uni_sram``) and the register-broadcast forms measured 1.3-14x
*worse* than the per-row path.  ``tl.atomic_*`` is not usable and there is no
``tl.gather`` on this backend.

Fast path (``mode == 0``, no per-sample weights, no padding)
-----------------------------------------------------------
One program per bag issues a single 2-D DMA per bag: `ROWS` rows at a time with
**source row stride 0** (the DMA broadcasts one ``grad`` row) and a contiguous
destination run (``tle.dsa`` / SDNN engine).  Requires ``is_sdnn=True`` on the
launch -- without it the same kernel fails to legalize with ``uni_sram``.
Measured on ``[512,256]``/nnz=4096 fp16: target kernel 0.97 ms -> 0.05 ms
(19.5x).  ``ROWS`` is capped by the 32 KiB ``uni_sram`` tile; the tile is loaded
once per bag and the store loop walks the bag in ``ROWS``-row chunks so the
source row is moved once however large the bag is.

Dtypes: fp16/fp32 use their own SRAM type, bf16 is a pure bit move and rides an
``int16`` view (`_DSA_BUF_DTYPE` rejects bf16).  ``D * itemsize`` must be
16-byte aligned (DMA requirement); everything else, plus mean mode,
per-sample-weight mode, ``padding_idx >= 0`` and the empty input, keeps the
generic device kernel unchanged (no aten/CPU path is entered either way).
"""

import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn

try:
    import triton.experimental.tle.language as tle

    _HAS_TLE = all(
        hasattr(tle.dsa, a) for a in ("alloc", "copy", "to_tensor", "UNI_SRAM")
    )
except ImportError:  # triton build without the XPU tile-language extension
    _HAS_TLE = False

logger = logging.getLogger(__name__)

# uni_sram budget per program on the dsa path (see utils/tle_copy.py) and the
# row tile measured best for this op: larger raises the per-bag descriptor cost
# in scaled modes and does not help the plain one, smaller leaves money on the
# table for bags with many samples.
_SRAM_BYTES = 32768
_ROWS_MAX = 64
# (dtype the kernels see, tl dtype of the SRAM buffer).  bf16 has no SRAM type,
# so it moves as the raw 2-byte pattern -- a copy never interprets the value.
_DSA_DTYPES = {
    torch.float16: (torch.float16, tl.float16),
    torch.float32: (torch.float32, tl.float32),
    torch.bfloat16: (torch.int16, tl.int16),
}


@triton.jit(do_not_specialize=["nnz", "num_bags", "s_row"])
def _ebsb_bcast_kernel(
    grad_ptr,
    o2b_ptr,
    offsets_ptr,
    out_ptr,
    nnz,
    num_bags,
    s_row,
    COLS: tl.constexpr,
    ROWS: tl.constexpr,
    DT: tl.constexpr,
):
    """Broadcast one ``grad`` row into the output rows of one bag.

    ``s_row`` is always 0 at the call sites: passing it as a runtime scalar (and
    not a literal) keeps the source tile a genuine 2-D DMA with a zero row
    stride; folding it at compile time changes the layout inference and the
    kernel no longer legalizes.
    """
    b = tl.program_id(0)
    nnz_hi = nnz
    start = tl.load(offsets_ptr + b).to(tl.int32)
    nxt = tl.load(offsets_ptr + tl.minimum(b + 1, num_bags - 1)).to(tl.int32)
    end = tl.where(b + 1 < num_bags, nxt, nnz_hi)
    # Clamp to [0, nnz] so a malformed offsets entry can never drive a DMA past
    # the buffers (an out-of-range DMA is what takes the card down on XPU3).
    start = tl.minimum(tl.maximum(start, 0), nnz_hi)
    end = tl.minimum(tl.maximum(end, 0), nnz_hi)
    nrows = end - start
    # The row every sample of this bag reads; clamped for empty bags.
    src_row = tl.load(o2b_ptr + tl.minimum(start, nnz_hi - 1)).to(tl.int32)
    r = tl.arange(0, ROWS)
    c = tl.arange(0, COLS)
    buf = tle.dsa.alloc([ROWS, COLS], DT, tle.dsa.UNI_SRAM)
    src_tile = grad_ptr + src_row * COLS + r[:, None] * s_row + c[None, :]
    tle.dsa.copy(src_tile, buf, sizes=[tl.minimum(nrows, ROWS), COLS])
    for k in range(0, nrows, ROWS):
        dsz = tl.minimum(nrows - k, ROWS)
        dst = out_ptr + (start + k) * COLS + r[:, None] * COLS + c[None, :]
        tle.dsa.copy(buf, dst, sizes=[dsz, COLS])


def _embedding_bag_sparse_backward(
    grad: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    offset2bag: torch.Tensor,
    bag_size: torch.Tensor,
    num_weights: int,
    scale_grad_by_freq: bool,
    mode: int,
    per_sample_weights: torch.Tensor = None,
    padding_idx: int = -1,
) -> torch.Tensor:
    logger.debug("GEMS KUNLUNXIN _EMBEDDING_BAG_SPARSE_BACKWARD")

    assert indices.dtype in (
        torch.int32,
        torch.int64,
    ), "Indices must be int32 or int64."
    assert (
        grad.device.type == flag_gems.device
        and indices.device.type == flag_gems.device
        and offset2bag.device.type == flag_gems.device
        and bag_size.device.type == flag_gems.device
        and grad.device == indices.device
    ), "All inputs must be on the same device."

    device = grad.device
    D = grad.shape[1]
    num_bags = bag_size.numel()
    nnz = indices.numel()

    dsa_dtypes = _DSA_DTYPES.get(grad.dtype)
    has_padding = padding_idx is not None and padding_idx >= 0
    # ``row_bytes % 16`` is the DMA alignment rule; a zero element size would
    # divide by zero below, and ``nnz == 0`` is cheaper on the generic path.
    use_fast = (
        _HAS_TLE
        and dsa_dtypes is not None
        and mode == 0
        and per_sample_weights is None
        and not has_padding
        and nnz > 0
        and D > 0
        and (D * grad.element_size()) % 16 == 0
        and grad.is_contiguous()
        and offset2bag.is_contiguous()
        and offsets.numel() >= num_bags
        and offsets.numel() <= num_bags + 1
        and offset2bag.numel() == nnz
    )

    if not use_fast:
        from flag_gems.ops._embedding_bag_sparse_backward import (
            _embedding_bag_sparse_backward as _generic,
        )

        return _generic(
            grad,
            indices,
            offsets,
            offset2bag,
            bag_size,
            num_weights,
            scale_grad_by_freq,
            mode,
            per_sample_weights,
            padding_idx,
        )

    values = torch.empty((nnz, D), device=device, dtype=grad.dtype)
    kernel_dtype, sram_dtype = dsa_dtypes
    rows = max(1, min(_ROWS_MAX, _SRAM_BYTES // (D * grad.element_size())))

    grad_v = grad if grad.dtype == kernel_dtype else grad.view(kernel_dtype)
    values_v = values if values.dtype == kernel_dtype else values.view(kernel_dtype)

    with torch_device_fn.device(device):
        _ebsb_bcast_kernel[(num_bags,)](
            grad_v,
            offset2bag,
            offsets,
            values_v,
            nnz,
            num_bags,
            0,
            COLS=D,
            ROWS=rows,
            DT=sram_dtype,
            is_sdnn=True,
        )

    sparse_indices = indices if indices.dtype == torch.int64 else indices.to(torch.int64)
    return torch.sparse_coo_tensor(
        sparse_indices.unsqueeze(0), values, (num_weights, D), device=device
    )
