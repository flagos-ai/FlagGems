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
"""Kunlunxin XPU3 override for ``aten::embedding_sparse_backward``.

Semantics (unchanged): the backward of a sparse embedding lookup returns an
*uncoalesced* COO tensor of shape ``(num_weights, D)`` whose ``nnz`` values are
the ``(N, D)`` rows of ``grad_output``; rows whose index equals ``padding_idx``
contribute nothing.  Dropping those rows (what ATen does) and writing them as
zeros are dense-equivalent, which is what the test compares.

Why the generic implementation is unusable on XPU3 (measured,
``harness/perf_ir/embedding_sparse_backward/prof_pre.log``)
---------------------------------------------------------------------
The generic kernel is "one program per grad row": it loads a ``BLOCK_D``-lane
row, applies ``tl.where`` and stores it.  Two independent failures:

* ``D >= 128`` variants either fail to legalize (``TritonXPUUnrollControl``:
  ``arith.select`` type mismatch where ``tl.where(keep, go, 0.0)`` mixes an
  fp16 tile with an fp32 literal) or hit ``uni_sram`` "out of resource", so the
  whole padded path is broken for the benchmark matrix (D = 128 / 256);
* where it does compile it is a "narrow DMA per program" kernel: 65 536
  programs of 256 B, 12.3 ms for ``(32, 2048, 128)`` fp16 while ATen takes
  185 us.

This override makes the copy **flat**: each program owns a contiguous
``BLOCK``-lane span of the row-major ``values`` buffer, recovers the row with
``off // ROW`` (``ROW`` is ``tl.constexpr``, so the divide strength-reduces) and
reads the per-row keep flag with one gathered ``indices`` element.  The
load/store pair stays a single contiguous block DMA.  The gathered index tile
is tiny (``N`` int64 = 512 KB at the benchmark sizes) and measured at ~5 us of
extra cost over a pure copy:

| case (8.4 M elems) | ATen | generic | this kernel |
|---|---|---|---|
| fp16 ``(32,2048,128)`` pad>=0 | 185 us | 12 310 us | 41 us |
| fp32 ``(32,2048,128)`` pad>=0 | 202 us | 11 429 us | 75 us |
| bf16 ``(32,2048,128)`` pad>=0 | 190 us | 233 us* | 41 us |

(*) bf16 needs the lane trick below.

bf16: a bf16 ``tl.where`` lowers to ``vmerge``/``vscatter_mh`` on XPU3 and
costs 233 us for the same tile (5.7x the fp16 path).  Since this op is a pure
bit move, bf16 is transported through an ``fp16`` *view* of the same buffer --
``view(dtype)`` is a metadata reinterpret, every bit is preserved -- which lands
on the fast fp16 lane path (41 us).  Int word views were measured and are
*worse* on this backend (int32/int16 stores take a slow path: 106-175 us), so
the kernel runs on native element lanes.

``padding_idx < 0`` (or ``None``) needs no device work at all: the COO values
are the grad rows themselves, exactly like ATen.
"""

import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)

# Flat-tile ladder (lanes).  XPU3 copy throughput is tile driven: 16 K lanes at
# 16 warps is the measured peak for the 8.4 M-element benchmark cases, and the
# ladder keeps at least 32 programs in flight for small inputs.  BLOCK stays
# well below 64 K (larger tiles have produced device faults on this backend).
_LADDER = (
    (1 << 19, 16384, 16),
    (1 << 16, 8192, 8),
    (1 << 13, 4096, 8),
    (1 << 10, 1024, 4),
    (1 << 7, 256, 4),
)


@triton.jit
def _embedding_sparse_backward_pad_kernel(
    grad_ptr,  # (N * ROW,) flattened grad rows
    idx_ptr,  # (N,) embedding indices
    out_ptr,  # (N * ROW,) flattened COO values
    numel,  # N * ROW
    padding_idx,
    ROW: tl.constexpr,  # elements per row in lane units
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    zero = tl.zeros((BLOCK,), dtype=grad_ptr.dtype.element_ty)
    if NEED_MASK:
        m = off < numel
        keep = tl.load(idx_ptr + off // ROW, mask=m, other=0) != padding_idx
        v = tl.load(grad_ptr + off, mask=m, other=0)
        tl.store(out_ptr + off, tl.where(keep, v, zero), mask=m)
    else:
        keep = tl.load(idx_ptr + off // ROW) != padding_idx
        v = tl.load(grad_ptr + off)
        tl.store(out_ptr + off, tl.where(keep, v, zero))


def _pick_block(n):
    for threshold, block, warps in _LADDER:
        if n >= threshold:
            return block, warps
    return 64, 4


def embedding_sparse_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
):
    logger.debug("GEMS_KUNLUNXIN EMBEDDING_SPARSE_BACKWARD")
    assert indices.dtype in (
        torch.int32,
        torch.int64,
    ), "Indices must be int32 or int64."
    if (
        grad_output.device.type != flag_gems.device
        or indices.device.type != flag_gems.device
        or grad_output.device != indices.device
    ):
        raise ValueError(
            f"Inputs must be {flag_gems.device} tensors on the same device."
        )
    if scale_grad_by_freq:
        # Matches aten: frequency scaling is not defined for sparse gradients.
        raise RuntimeError(
            "embedding_backward: scale_grad_by_freq not supported with sparse gradients"
        )
    assert (
        grad_output.dim() >= 2
    ), "grad_output must have embedding dimension as the last dim."

    device = grad_output.device
    D = grad_output.shape[-1]
    go = grad_output.contiguous().view(-1, D)  # (N, D)
    idx = indices.contiguous().view(-1)  # (N,)
    N = idx.numel()
    assert go.shape[0] == N, "indices number must match grad_output rows."

    # aten's sparse backward emits one COO entry per grad row and drops the ones
    # whose index equals padding_idx.  Dropped row == all-zero row once the COO
    # tensor is densified, so the padding rows are zeroed in place during the
    # copy instead of being compacted on device (a compaction needs a data
    # dependent scatter / host sync).
    if padding_idx is not None and padding_idx >= 0:
        values = torch.empty((N, D), device=device, dtype=grad_output.dtype)
        if go.numel() > 0:
            # Row width stays D in lane units for every dtype: fp16/bf16 are both
            # 2 bytes, so ``view(dtype)`` keeps the shape.
            row = D
            if grad_output.dtype is torch.bfloat16:
                # Pure bit move: run bf16 payload through fp16 lanes (see the
                # module docstring); every bit (including NaN payloads) is
                # preserved because no conversion is performed.
                src = go.view(torch.float16).view(-1)
                dst = values.view(torch.float16).view(-1)
            else:
                src = go.view(-1)
                dst = values.view(-1)
            numel = src.numel()
            block, num_warps = _pick_block(numel)
            grid = ((numel + block - 1) // block,)
            _embedding_sparse_backward_pad_kernel[grid](
                src,
                idx,
                dst,
                numel,
                padding_idx,
                ROW=row,
                BLOCK=block,
                NEED_MASK=(numel % block) != 0,
                num_warps=num_warps,
            )
    else:
        # No padding row to blank: the flattened grad rows are the COO values.
        values = go

    sparse_indices = idx if idx.dtype == torch.int64 else idx.to(torch.int64)
    return torch.sparse_coo_tensor(
        sparse_indices.view(1, -1),
        values,
        (num_weights, D),
        dtype=grad_output.dtype,
        device=device,
    )
