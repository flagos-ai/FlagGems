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
from flag_gems.utils import triton_lang_extension as ext

from .nonzero import (
    _count_nonzero,
    _dense_result,
    _sparse_result,
    nonzero,
)

logger = logging.getLogger(__name__)

# Block width of the compaction chain below. 8192 is the documented safe
# `tl.sum` / `tl.cumsum` tile on this backend (the value the closed nonzero
# counter and the closed masked_scatter compaction both use).
_NZNP_BLOCK = 8192
_NZNP_WARPS = 16


@libentry()
@triton.jit
def _nznp_count_full_kernel(inp, counts, BLOCK: tl.constexpr):
    # Per-block nonzero count over FULL blocks only: the offsets are affine and
    # every lane is live, so this pass carries no mask at all (a masked tail
    # load feeding a reduction is the documented silent-error pattern here).
    pid = ext.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    w = tl.load(inp + cols)
    tl.store(counts + pid, tl.sum((w != 0).to(tl.int32), axis=0).to(tl.int64))


@libentry()
@triton.jit(do_not_specialize=["n_elements", "n_main", "slot"])
def _nznp_count_tail_kernel(inp, counts, n_elements, n_main, slot, BLOCK: tl.constexpr):
    # Remainder block (< BLOCK elements). Offsets are clamped and the
    # out-of-range lanes are zeroed by an integer ok-multiplier, so the
    # reduction never consumes an untrusted masked `other` value.
    cols = n_main + tl.arange(0, BLOCK)
    last = n_elements - 1
    cclamp = tl.minimum(cols, last)
    ok = (cols < n_elements).to(tl.int32)
    w = tl.load(inp + cclamp)
    tl.store(counts + slot, tl.sum((w != 0).to(tl.int32) * ok, axis=0).to(tl.int64))


@libentry()
@triton.jit(do_not_specialize=["stride_d", "size_d"])
def _nznp_clean_dim_kernel(
    ids,
    bases,
    outd,
    stride_d,
    size_d,
    BLOCK: tl.constexpr,
):
    # CLEAN blocks (no zero at all): the block's BLOCK source elements map onto
    # the contiguous slot run [base, base + BLOCK), so the slot index is
    # `scalar + arange` and the coordinate store is an affine, mask-free int64
    # write. The input is not read at all; the coordinate only depends on the
    # flat source index. Which blocks are clean is decided on the host from the
    # per-block counts, so no data-dependent branch is needed here (an in-kernel
    # `if cnt == BLOCK` around the two store shapes fails to lower on this
    # backend: `OutOfResources: uni_sram`).
    #
    # Two backend-specific constraints shape this kernel:
    #   * ONE LAUNCH PER DIM. A `for d in range(ndim)` loop with the store inside
    #     it does not lower in a kernel of this shape (uni_sram validation
    #     failure at every block size / warp count, whether the dim size comes
    #     from a global load, a list or an if-chain over scalar args).
    #   * SINGLE-TERM STORE ADDRESS. `outd + base + lanes` runs at 229 GB/s, but
    #     adding one more runtime scalar to the address expression
    #     (`out + dim_off + base + lanes`, i.e. indexing the [ndim, N] buffer
    #     inside the kernel) drops it to 5.9 GB/s -- a 39x cliff. The per-dim
    #     row is therefore passed in as its own pointer (`out.select(0, d)`).
    pid = ext.program_id(0)
    blk = tl.load(ids + pid)
    base = tl.load(bases + pid)
    lanes = tl.arange(0, BLOCK)
    idx = blk * BLOCK + lanes
    coord = (idx // stride_d) % size_d
    tl.store(outd + base + lanes.to(tl.int64), coord.to(tl.int64))


@libentry()
@triton.jit(do_not_specialize=["total", "stride_d", "size_d"])
def _nznp_dirty_dim_kernel(
    inp,
    ids,
    bases,
    outd,
    total,
    stride_d,
    size_d,
    BLOCK: tl.constexpr,
):
    # DIRTY full blocks (at least one zero): in-block rank from a 1-D
    # `tl.cumsum` of the nonzero mask on top of the block's host-computed base,
    # then a compacting scatter. No global prefix sum and no `inp != 0` bool
    # copy are materialized. Same one-launch-per-dim / single-term-address
    # discipline as the clean kernel.
    #
    # Inactive lanes are redirected to a per-lane scratch slot in
    # [total, total + BLOCK) instead of relying on masked-store semantics: with
    # `r = base + cumsum - 1` every trailing inactive lane addresses the LAST
    # valid slot, so a store mask that is not honoured (the documented
    # TRITONXPU_STORE_MASK_SIM hazard) silently clobbers it. The scratch slots
    # are unique per lane, so the redirect can never alias a live slot.
    pid = ext.program_id(0)
    blk = tl.load(ids + pid)
    base = tl.load(bases + pid)
    lanes = tl.arange(0, BLOCK)
    cols = blk * BLOCK + lanes
    w = tl.load(inp + cols)
    nz = w != 0
    nzi = nz.to(tl.int32)
    r = (base + tl.cumsum(nzi, axis=0) - 1).to(tl.int64)
    r = tl.where(nz, r, total.to(tl.int64) + lanes.to(tl.int64))
    coord = (cols // stride_d) % size_d
    tl.store(outd + r, coord.to(tl.int64), mask=nz)


@libentry()
@triton.jit(
    do_not_specialize=[
        "total",
        "n_elements",
        "n_main",
        "base",
        "stride_d",
        "size_d",
    ]
)
def _nznp_tail_dim_kernel(
    inp,
    outd,
    total,
    n_elements,
    n_main,
    base,
    stride_d,
    size_d,
    BLOCK: tl.constexpr,
):
    # Remainder block of the compaction (always treated as dirty). Same clamp +
    # ok discipline as the tail counter, same per-lane scratch redirect as the
    # dirty full-block kernel.
    lanes = tl.arange(0, BLOCK)
    cols = n_main + lanes
    last = n_elements - 1
    cclamp = tl.minimum(cols, last)
    ok = cols < n_elements
    w = tl.load(inp + cclamp)
    nz = (w != 0) & ok
    nzi = nz.to(tl.int32)
    r = (base + tl.cumsum(nzi, axis=0) - 1).to(tl.int64)
    r = tl.where(nz, r, total.to(tl.int64) + lanes.to(tl.int64))
    coord = (cols // stride_d) % size_d
    tl.store(outd + r, coord.to(tl.int64), mask=nz)


def _nznp_block_counts(flat, n_elements):
    """Per-block nonzero counts (one read pass) brought back to the host.

    The same pass yields the exact total (so the separate two-phase counter is
    not needed), every block's output base, and the clean/dirty classification
    the compaction kernels are launched from.
    """
    dev = flat.device
    block = _NZNP_BLOCK
    n_full = n_elements // block
    rem = n_elements - n_full * block
    n_blocks = n_full + (1 if rem else 0)
    counts = torch.empty(n_blocks, dtype=torch.int64, device=dev)
    with torch_device_fn.device(dev):
        if n_full > 0:
            _nznp_count_full_kernel[(n_full,)](
                flat, counts, BLOCK=block, num_warps=_NZNP_WARPS
            )
        if rem > 0:
            _nznp_count_tail_kernel[(1,)](
                flat,
                counts,
                n_elements,
                n_full * block,
                n_full,
                BLOCK=block,
                num_warps=_NZNP_WARPS,
            )
    return n_full, rem, counts.cpu()


def _nznp_compact(inp, n_elements, n_full, rem, counts_h, total):
    """Dim-major compaction; returns ndim contiguous 1-D index views."""
    ndim = inp.ndim
    dev = inp.device
    block = _NZNP_BLOCK
    row_stride = total + block
    out = torch.empty(ndim, row_stride, dtype=torch.int64, device=dev)
    flat = inp.view(-1)

    bases_h = torch.empty_like(counts_h)
    bases_h[0] = 0
    if counts_h.numel() > 1:
        torch.cumsum(counts_h[:-1], dim=0, out=bases_h[1:])

    if n_full > 0:
        full_counts = counts_h[:n_full]
        clean_h = (full_counts == block).nonzero().reshape(-1)
        dirty_h = (full_counts != block).nonzero().reshape(-1)
    else:
        clean_h = counts_h.new_empty(0)
        dirty_h = counts_h.new_empty(0)

    # row-major source strides, so coord_d = (i // stride_d) % size_d
    src_strides = [1] * ndim
    for k in range(ndim - 2, -1, -1):
        src_strides[k] = src_strides[k + 1] * inp.shape[k + 1]

    clean_ids = clean_bases = None
    if clean_h.numel() > 0:
        clean_ids = clean_h.to(torch.int32).to(dev)
        clean_bases = bases_h[clean_h].to(dev)
    dirty_ids = dirty_bases = None
    if dirty_h.numel() > 0:
        dirty_ids = dirty_h.to(torch.int32).to(dev)
        dirty_bases = bases_h[dirty_h].to(dev)
    tail_base = int(bases_h[n_full].item()) if rem > 0 else 0

    with torch_device_fn.device(dev):
        for d in range(ndim):
            outd = out.select(0, d)
            if clean_ids is not None:
                _nznp_clean_dim_kernel[(clean_ids.numel(),)](
                    clean_ids,
                    clean_bases,
                    outd,
                    src_strides[d],
                    inp.shape[d],
                    BLOCK=block,
                    num_warps=_NZNP_WARPS,
                )
            if dirty_ids is not None:
                _nznp_dirty_dim_kernel[(dirty_ids.numel(),)](
                    flat,
                    dirty_ids,
                    dirty_bases,
                    outd,
                    total,
                    src_strides[d],
                    inp.shape[d],
                    BLOCK=block,
                    num_warps=_NZNP_WARPS,
                )
            if rem > 0:
                _nznp_tail_dim_kernel[(1,)](
                    flat,
                    outd,
                    total,
                    n_elements,
                    n_full * block,
                    tail_base,
                    src_strides[d],
                    inp.shape[d],
                    BLOCK=block,
                    num_warps=_NZNP_WARPS,
                )
    return list(out[:, :total].unbind(0))


def nonzero_numpy(inp):
    """
    Returns a tuple of 1D tensors, one for each dimension of the input,
    containing the indices of the non-zero elements in that dimension.

    This is equivalent to torch.nonzero(...) / numpy.nonzero() semantics and
    matches the ATen op `nonzero_numpy` (Tensor[] of per-dim index vectors).

    Backend chain (all Kunlunxin/XPU Triton, no fallback):

    * one mask-free read pass produces the per-block nonzero counts; they give
      the exact total, every block's output base and the clean/dirty split.
    * `total == numel` (dense: `randn` in fp32, wide-range `randint`) reuses the
      closed `_dense_result` args kernel -- purely affine int64 stores.
    * NEAR-DENSE inputs, i.e. the zeros dirty at most a quarter of the blocks
      (fp16/bf16 `randn` rounds a handful of samples of a 655M-element tensor to
      exactly 0; full-range int16 `randint` hits 0 about once per 65536) go
      through a dim-major block compaction. A clean block writes its whole slot
      run with mask-free affine stores and never reads the input, so the run
      costs dense-path time instead of full-scatter time.
    * genuinely sparse inputs (a ~50%-True bool mask) would dirty every block,
      where the in-block rank scan does not pay for itself, so they keep the
      previously closed prefix-sum scatter -- with the total taken from the
      counts above, so no second counting pass is needed.
    * 0-dim scalars are special-cased as in ATen: treated as a 1-element 1-D
      tensor (index [0] when nonzero, empty index when zero).

    `ndim > 8` and outputs at/over the int32 index ceiling keep the previously
    closed chain (`_count_nonzero` + `_dense_result` / `_sparse_result`); the
    closed `nonzero` small-input branch is still never entered, because its
    fallback dense kernel (`nonzero_dense_flat_kernel`, per-lane masked
    metadata loads) fails to compile on this backend at BLOCK_SIZE >= 128.
    """
    logger.debug("GEMS_KUNLUNXIN NONZERO_NUMPY")

    if inp.ndim == 0:
        inp = inp.reshape(-1)

    inp_ndim = inp.ndim
    n_elements = inp.numel()

    if n_elements == 0:
        # ATen: empty input -> ndim empty 1-D index tensors.
        out = torch.empty(0, inp_ndim, dtype=torch.int64, device=inp.device)
        return list(out.unbind(dim=1))

    inp = inp.contiguous()

    if inp_ndim <= 8 and n_elements < 2**31:
        n_full, rem, counts_h = _nznp_block_counts(inp.view(-1), n_elements)
        total = int(counts_h.sum().item())
        if total * inp_ndim < 2**31:
            if total == n_elements:
                return list(_dense_result(inp, total, True))
            n_dirty = int((counts_h[:n_full] != _NZNP_BLOCK).sum().item())
            if 4 * n_dirty <= n_full:
                return _nznp_compact(inp, n_elements, n_full, rem, counts_h, total)
            return list(_sparse_result(inp, inp_ndim, n_elements, total, True))

    # Outputs at/over the int32 index ceiling, or ndim > 8: previously closed
    # chain (exact two-phase count + dense args kernel / prefix-sum scatter).
    if n_elements >= 8192 and inp.dtype != torch.bool:
        return list(nonzero(inp, as_tuple=True))
    num_nonzeros = _count_nonzero(inp, n_elements)
    if (
        inp_ndim >= 1
        and num_nonzeros == n_elements
        and num_nonzeros * inp_ndim < 2**31
    ):
        return list(_dense_result(inp, num_nonzeros, True))
    return list(_sparse_result(inp, inp_ndim, n_elements, num_nonzeros, True))
