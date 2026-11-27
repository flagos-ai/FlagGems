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

"""Sparse (COO) x dense matmul with a Triton kernel.

See the kernel and host docstrings below for the algorithm. The kernels are
the generic ones; only the host wrapper and the import-time shims carry the
enflame (gcu400) adaptations.

Enflame (gcu400) adaptations:

- torch_gcu ships no Sparsegcu kernels: ``aten::sspaddmm`` (what ``torch.smm``
  decomposes to) raises NotImplementedError for gcu, and the sparse-COO
  factories silently materialize tensors on CPU (``Tensor.to_sparse`` on a
  device dense tensor and ``torch.sparse_coo_tensor`` from device members
  both copy their inputs D2H and report ``device=cpu``). A GCU workload
  therefore hands this op a CPU-resident COO operand while ``mat`` stays on
  the device, and the generic same-device check rejects the call before any
  kernel runs.
- The import-time shims below tag COO tensors built from device inputs with
  their origin device (``Tensor.to_sparse``; ``Tensor.coalesce`` propagates
  the tag; the public ``torch.sparse_coo_tensor`` factory) and shadow
  ``to_dense`` per instance so the dense materialization lands back on the
  origin device. The tag restores the CUDA device contract: the mixed-device
  check reports the workload's true device (a CPU ``mat`` paired with a
  device-origin sparse operand raises, exactly like upstream), and the
  benchmark's native reference ``(self.to_dense() @ mat)`` computes on the
  device. Every other call shape passes through untouched: CPU inputs,
  non-COO layouts, untagged tensors. The pattern mirrors the CSR accessor
  shims in sparse_sampled_addmm.py and the per-instance accessors in
  sparse_coo_tensor.py; drop all of it once torch_gcu grows a Sparsegcu axis.
- The host wrapper moves the COO member tensors to ``mat``'s device (a plain
  data movement) and launches the Triton kernels there. On this platform the
  members arrive CPU-resident on every call, so the device copies are cached
  keyed on the source members (address, version counter, shape/dtype): repeat
  calls with the same sparse operand skip the pageable H2D round trips. The
  row-boundary ranking composes the FlagGems ``cumsum`` op (a plain FlagGems
  kernel composition), not an aten compute call. The result COO is assembled
  through the platform factory, which materializes the output on CPU exactly
  like every input COO on this platform: the device-side outputs are staged
  through pinned host buffers (the factory's direct D2H of device tensors
  goes through the slow pageable path), synchronized, then handed to the
  factory.
- The main matmul kernel is rewritten for the GCU memory system. The generic
  kernel gathers a ``(BLOCK_NNZ, BLOCK_N)`` tile of ``B`` with one row index
  per nonzeros element, which the GCU compiler lowers to element-wise
  emissions. The rewrite streams the row's COO entries one at a time and
  loads each ``B[k]`` slice as one contiguous ``BLOCK_N`` vector (the affine
  access the tile gather cannot express), with an ``UNROLL`` knob that groups
  the scalar ``k``/``v`` loads of consecutive entries ahead of their dependent
  vector loads to break the load-to-load dependency chain.
"""

import logging
from collections import OrderedDict

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# torch GCU sparse-COO origin shims (torch dispatch gap workaround)
#
# torch_gcu registers no Sparsegcu kernels, so the sparse-COO factories
# materialize their tensors on CPU while the caller works on the device. The
# shims below remember where a COO tensor really came from:
#   1. ``Tensor.to_sparse`` tags device-dense conversions with the source
#      device;
#   2. ``torch.sparse_coo_tensor`` tags constructions whose member tensors
#      live on the device (chaining onto whatever factory shim was installed
#      earlier, e.g. the sparse_coo_tensor.py aliasing fix, in any order);
#   3. ``Tensor.coalesce`` propagates the tag (coalesce returns a fresh
#      tensor for uncoalesced inputs).
# Tagged tensors additionally get a per-instance ``to_dense`` that lands the
# dense materialization back on the origin device, restoring the CUDA
# semantics of ``x.to_sparse().to_dense()`` staying on x's device. The stored
# CPU components keep serving every C++ consumer (to_dense/_nnz/autograd) --
# only Python-level reads observe the origin device. Untagged tensors, CPU
# inputs, and non-COO layouts are passed through untouched, so sibling vendor
# shims compose in either import order.
# ---------------------------------------------------------------------------
_ORIGIN_TAG = "_flaggems_gcu_coo_origin"
_SHIM_MARKER = "_flaggems_gcu_coo_origin_shim"


def _attach_coo_origin(tensor, origin):
    """Tag ``tensor`` with its device origin and shadow ``to_dense``."""
    if not (
        isinstance(tensor, torch.Tensor)
        and tensor.layout is torch.sparse_coo
        and origin is not None
    ):
        return tensor
    if getattr(tensor, _ORIGIN_TAG, None) is not None:
        return tensor
    try:
        setattr(tensor, _ORIGIN_TAG, origin)
        native_to_dense = tensor.to_dense

        def _origin_to_dense(*args, **kwargs):
            return native_to_dense(*args, **kwargs).to(origin)

        tensor.to_dense = _origin_to_dense
    except (AttributeError, TypeError):
        logger.debug("could not attach gcu COO origin tag")
    return tensor


def _install_gcu_coo_origin_shims():
    if getattr(torch.Tensor.to_sparse, _SHIM_MARKER, False):
        return
    device_type = runtime_device.name

    tensor_to_sparse = torch.Tensor.to_sparse

    def to_sparse(self, *args, **kwargs):
        result = tensor_to_sparse(self, *args, **kwargs)
        if self.device.type == device_type:
            return _attach_coo_origin(result, self.device)
        return result

    to_sparse.__doc__ = tensor_to_sparse.__doc__
    setattr(to_sparse, _SHIM_MARKER, True)
    torch.Tensor.to_sparse = to_sparse

    tensor_coalesce = torch.Tensor.coalesce

    def coalesce(self):
        origin = getattr(self, _ORIGIN_TAG, None)
        result = tensor_coalesce(self)
        if origin is not None:
            return _attach_coo_origin(result, origin)
        return result

    coalesce.__doc__ = tensor_coalesce.__doc__
    setattr(coalesce, _SHIM_MARKER, True)
    torch.Tensor.coalesce = coalesce

    factory = torch.sparse_coo_tensor

    def sparse_coo_tensor(*args, **kwargs):
        result = factory(*args, **kwargs)
        if not (isinstance(result, torch.Tensor) and result.layout is torch.sparse_coo):
            return result
        indices = args[0] if len(args) > 0 else kwargs.get("indices")
        values = args[1] if len(args) > 1 else kwargs.get("values")
        origin = None
        for member in (indices, values):
            if isinstance(member, torch.Tensor) and member.device.type != "cpu":
                origin = member.device
                break
        if origin is None:
            return result
        return _attach_coo_origin(result, origin)

    sparse_coo_tensor.__doc__ = factory.__doc__
    setattr(sparse_coo_tensor, _SHIM_MARKER, True)
    torch.sparse_coo_tensor = sparse_coo_tensor


_install_gcu_coo_origin_shims()


# ---------------------------------------------------------------------------
# Cached device copies of the (CPU-materialized) COO members.
#
# On this platform every COO operand arrives with CPU-resident members (no
# Sparsegcu kernels), so each call would otherwise pay the pageable H2D round
# trips for ``indices`` and ``values`` on top of the kernel work. The cache
# keys on the source members' identity: data pointer, version counter, and
# shape/dtype. Holding strong references to the sources keeps their addresses
# from being recycled while an entry is cached, and the version counter
# invalidates the entry on any in-place mutation of the members. LRU eviction
# bounds the retained memory (the device copies plus the pinned sources).
# Misses fall back to the plain per-call copies, so a cache overrun can never
# change behavior, only the copy count.
# ---------------------------------------------------------------------------
_MEMBER_CACHE_LIMIT_BYTES = 256 * 1024 * 1024
_MEMBER_CACHE_LIMIT_ENTRIES = 16
_member_cache = OrderedDict()


def _cached_device_members(indices, values, device):
    """Return device copies of the COO members, cached on the sources."""
    key = (
        str(device),
        indices.data_ptr(),
        values.data_ptr(),
        indices._version,
        values._version,
        tuple(indices.shape),
        indices.dtype,
        tuple(values.shape),
        values.dtype,
    )
    hit = _member_cache.get(key)
    if hit is not None:
        _member_cache.move_to_end(key)
        return hit[2], hit[3]
    indices_dev = indices.to(device).contiguous()
    values_dev = values.to(device).contiguous()
    nbytes = (
        indices.element_size() * indices.numel()
        + values.element_size() * values.numel()
    )
    _member_cache[key] = (indices, values, indices_dev, values_dev, nbytes)
    total = nbytes + sum(entry[4] for entry in _member_cache.values())
    while _member_cache and (
        len(_member_cache) > _MEMBER_CACHE_LIMIT_ENTRIES
        or total > _MEMBER_CACHE_LIMIT_BYTES
    ):
        _, evicted = _member_cache.popitem(last=False)
        total -= evicted[4]
    return indices_dev, values_dev


# Tuning configs for the COO SpMM kernel. Each program owns one structural
# output row (a row present in the sparse input) and a tile of output columns.
# It streams that row's nonzeros straight out of the COO arrays, so the working
# set scales with ``nnz`` rather than with the dense ``M * K``. The COO output
# (indices + values) is written directly, avoiding a re-sparsify pass.
# ``UNROLL`` groups the scalar k/v loads of ``UNROLL`` consecutive entries
# ahead of their dependent B-row vector loads; the grouped form wins for wider
# BLOCK_N tiles where the per-entry load-to-load chain dominates.
SMM_CONFIGS = [
    triton.Config({"BLOCK_N": 64, "UNROLL": 1}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_N": 64, "UNROLL": 2}, num_stages=2, num_warps=2),
    triton.Config({"BLOCK_N": 128, "UNROLL": 1}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_N": 128, "UNROLL": 4}, num_stages=2, num_warps=2),
    triton.Config({"BLOCK_N": 256, "UNROLL": 1}, num_stages=3, num_warps=2),
    triton.Config({"BLOCK_N": 256, "UNROLL": 4}, num_stages=2, num_warps=2),
    triton.Config({"BLOCK_N": 512, "UNROLL": 1}, num_stages=2, num_warps=2),
    triton.Config({"BLOCK_N": 512, "UNROLL": 2}, num_stages=2, num_warps=2),
    triton.Config({"BLOCK_N": 512, "UNROLL": 4}, num_stages=2, num_warps=2),
]


@libentry()
@triton.jit
def smm_mark_row_starts_kernel(
    row_indices,
    is_start,
    nnz,
    BLOCK: tl.constexpr,
):
    """Flag each COO entry that begins a new row.

    A coalesced 2-D COO tensor is sorted by row (then column), so the entries of
    one row occupy a contiguous span and a row boundary is simply an entry whose
    row differs from its predecessor. Entry 0 is always a boundary.
    """
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz

    cur = tl.load(row_indices + offs, mask=mask, other=0)
    prev = tl.load(row_indices + offs - 1, mask=mask & (offs > 0), other=-1)
    tl.store(is_start + offs, tl.where(cur != prev, 1, 0), mask=mask)


@libentry()
@triton.jit
def smm_compact_rows_kernel(
    row_indices,
    is_start,
    rank,
    row_ids,
    row_starts,
    nnz,
    nrows,
    BLOCK: tl.constexpr,
):
    """Compact the flagged boundaries into per-row ids and segment offsets.

    ``rank`` is the inclusive prefix sum of ``is_start``, so a boundary entry at
    COO offset ``o`` is the ``rank[o] - 1``-th distinct row. For each boundary we
    write its row id and its offset, giving ``row_starts[i] .. row_starts[i+1]``
    as the half-open span of row ``i``. The terminating ``row_starts[nrows] =
    nnz`` is written by the single program that owns the last entry.
    """
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz

    flag = tl.load(is_start + offs, mask=mask, other=0)
    sel = mask & (flag != 0)
    r = tl.load(rank + offs, mask=sel, other=1) - 1
    row = tl.load(row_indices + offs, mask=sel, other=0)

    tl.store(row_ids + r, row, mask=sel)
    tl.store(row_starts + r, offs.to(tl.int64), mask=sel)
    # Terminator, so segment ``nrows - 1`` has an end. Scalar store, hence the
    # ``pid`` guard rather than a mask (a block mask needs a block pointer).
    if pid == 0:
        tl.store(row_starts + nrows, nnz)


@libentry()
@libtuner(
    configs=SMM_CONFIGS,
    # Keyed on N only. ``nrows`` is data-dependent (it counts the distinct rows
    # present in the sparse operand), so keying on it would re-tune on nearly
    # every call with new data -- the tuning cost then dominates the kernel.
    # BLOCK_N/BLOCK_NNZ only need N and the per-row nnz distribution, and the
    # latter is not knowable from the launch signature anyway.
    key=["N"],
)
@triton.jit
def smm_coo_kernel(
    col_indices,
    values,
    row_ids,
    row_starts,
    B,
    out_row_idx,
    out_col_idx,
    out_values,
    N,
    stride_bk,
    stride_bn,
    CONTIG_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
):
    """Sparse(COO) x dense matmul, reading the sparse operand as COO.

    Program ``(i, pid_n)`` computes columns ``[pid_n * BLOCK_N, +BLOCK_N)`` of
    structural output row ``i``:

        out[i, n] = sum over the COO entries (r_i, k, v) of row r_i  of
                    v * B[k, n]

    The row's entries are streamed one at a time from
    ``[row_starts[i], row_starts[i + 1])``, so the memory touched is
    proportional to that row's nonzero count -- never to ``M * K``. Each entry
    contributes one multiply-add of its scalar value against the entry's
    ``B[k]`` row slice, loaded as a single ``BLOCK_N`` vector: with
    ``CONTIG_N`` the slice is stride-1 in memory, so the load is an affine
    contiguous vector access (the generic kernel's ``(BLOCK_NNZ, BLOCK_N)``
    tile gather carries one row index per nonzeros element, which the GCU
    compiler lowers to element-wise emissions). The contracted extent is a
    row's nnz (typically small and irregular), not a dense ``K``, so the
    reduction stays a broadcast-multiply-accumulate rather than ``tl.dot``.

    ``UNROLL`` groups the scalar ``k``/``v`` loads of ``UNROLL`` consecutive
    entries ahead of their dependent ``B`` vector loads: the per-entry
    load-to-load chain (k needed before the B slice address is known) is the
    latency floor of the streaming loop, and the grouping lets ``UNROLL``
    independent chains be in flight at once. Masked-off tail entries load
    ``k = 0`` with ``v = 0`` and contribute nothing.

    The COO output is written directly at flat position ``i * N + n``, which is
    the row-major layout ``torch.smm`` produces over ascending structural rows.
    """
    i = ext.program_id(0)
    pid_n = ext.program_id(1)

    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_off < N

    row = tl.load(row_ids + i)
    seg_start = tl.load(row_starts + i)
    seg_end = tl.load(row_starts + i + 1)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if UNROLL == 1:
        for e in range(seg_start, seg_end):
            k = tl.load(col_indices + e)
            v = tl.load(values + e).to(tl.float32)
            if CONTIG_N:
                b = tl.load(B + k * stride_bk + n_off, mask=n_mask, other=0.0)
            else:
                b = tl.load(
                    B + k * stride_bk + n_off * stride_bn, mask=n_mask, other=0.0
                )
            acc += v * b.to(tl.float32)
    elif UNROLL == 2:
        for base in range(seg_start, seg_end, 2):
            e0 = base
            e1 = base + 1
            m0 = e0 < seg_end
            m1 = e1 < seg_end
            k0 = tl.load(col_indices + e0, mask=m0, other=0)
            v0 = tl.load(values + e0, mask=m0, other=0.0).to(tl.float32)
            k1 = tl.load(col_indices + e1, mask=m1, other=0)
            v1 = tl.load(values + e1, mask=m1, other=0.0).to(tl.float32)
            if CONTIG_N:
                b0 = tl.load(B + k0 * stride_bk + n_off, mask=n_mask, other=0.0)
                b1 = tl.load(B + k1 * stride_bk + n_off, mask=n_mask, other=0.0)
            else:
                b0 = tl.load(
                    B + k0 * stride_bk + n_off * stride_bn, mask=n_mask, other=0.0
                )
                b1 = tl.load(
                    B + k1 * stride_bk + n_off * stride_bn, mask=n_mask, other=0.0
                )
            acc += v0 * b0.to(tl.float32) + v1 * b1.to(tl.float32)
    else:
        for base in range(seg_start, seg_end, 4):
            e0 = base
            e1 = base + 1
            e2 = base + 2
            e3 = base + 3
            m0 = e0 < seg_end
            m1 = e1 < seg_end
            m2 = e2 < seg_end
            m3 = e3 < seg_end
            k0 = tl.load(col_indices + e0, mask=m0, other=0)
            v0 = tl.load(values + e0, mask=m0, other=0.0).to(tl.float32)
            k1 = tl.load(col_indices + e1, mask=m1, other=0)
            v1 = tl.load(values + e1, mask=m1, other=0.0).to(tl.float32)
            k2 = tl.load(col_indices + e2, mask=m2, other=0)
            v2 = tl.load(values + e2, mask=m2, other=0.0).to(tl.float32)
            k3 = tl.load(col_indices + e3, mask=m3, other=0)
            v3 = tl.load(values + e3, mask=m3, other=0.0).to(tl.float32)
            if CONTIG_N:
                b0 = tl.load(B + k0 * stride_bk + n_off, mask=n_mask, other=0.0)
                b1 = tl.load(B + k1 * stride_bk + n_off, mask=n_mask, other=0.0)
                b2 = tl.load(B + k2 * stride_bk + n_off, mask=n_mask, other=0.0)
                b3 = tl.load(B + k3 * stride_bk + n_off, mask=n_mask, other=0.0)
            else:
                b0 = tl.load(
                    B + k0 * stride_bk + n_off * stride_bn, mask=n_mask, other=0.0
                )
                b1 = tl.load(
                    B + k1 * stride_bk + n_off * stride_bn, mask=n_mask, other=0.0
                )
                b2 = tl.load(
                    B + k2 * stride_bk + n_off * stride_bn, mask=n_mask, other=0.0
                )
                b3 = tl.load(
                    B + k3 * stride_bk + n_off * stride_bn, mask=n_mask, other=0.0
                )
            acc += (
                v0 * b0.to(tl.float32)
                + v1 * b1.to(tl.float32)
                + v2 * b2.to(tl.float32)
                + v3 * b3.to(tl.float32)
            )

    # ``out_row_idx`` and ``out_col_idx`` are the two rows of the (2, nnz_out)
    # index tensor, passed separately so the flat position needs no ``nrows``
    # arithmetic: Triton specializes an argument equal to 1 to a Python int, and
    # a single structural row would then break tensor-style index math here.
    pos = i.to(tl.int64) * N + n_off
    tl.store(out_values + pos, acc.to(out_values.dtype.element_ty), mask=n_mask)
    tl.store(out_row_idx + pos, row.to(tl.int64), mask=n_mask)
    tl.store(out_col_idx + pos, n_off.to(tl.int64), mask=n_mask)


def smm(self, mat):
    """Sparse (COO) matrix ``self`` multiplied by dense matrix ``mat``.

    Mirrors ``torch.smm``: the result is a sparse COO tensor whose rows are
    exactly the rows that appear in the sparse input's structure, each
    materialized with all ``N`` columns. A fused Triton kernel computes the
    product and writes the COO output (indices + values) directly, avoiding
    any separate dense-to-sparse conversion pass.
    """
    logger.debug("GEMS_ENFLAME SMM")

    if not self.is_sparse:
        raise RuntimeError("tensor.sspaddmm(...) can only be called on sparse tensors")
    if self.ndim != 2:
        raise RuntimeError(
            f"sspaddmm: Argument #2: matrices expected, got {self.ndim}D tensor"
        )
    if mat.is_sparse:
        raise RuntimeError(
            "Cannot access data pointer of Tensor that doesn't have storage"
        )
    if mat.ndim != 2:
        raise RuntimeError(
            f"sspaddmm: Argument #3: matrices expected, got {mat.ndim}D tensor"
        )

    M, K = self.shape
    K2, N = mat.shape
    if K != K2:
        raise RuntimeError(f"sspaddmm: Argument #3: Expected dim 0 size {K}, got {K2}")

    # Both operands must live on the same device: the kernel is launched on the
    # sparse operand's device, so a CPU or foreign-GPU ``mat`` would otherwise
    # be handed to it as a raw pointer and fault. On this platform the sparse
    # operand is CPU-materialized even for device workloads (no Sparsegcu
    # kernels), so the origin tag the module shims attach is the operand's
    # device contract -- exactly what upstream compares on CUDA.
    self_device = getattr(self, _ORIGIN_TAG, None)
    if self_device is None:
        self_device = self.device
    if mat.device != self_device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but got mat2 is on "
            f"{mat.device}, different from other tensors on {self_device}"
        )
    device = mat.device

    # ``torch.smm`` is only implemented for float32 on the CUDA backend, so we
    # restrict the GEMS implementation to the same dtype to stay consistent.
    if self.dtype != torch.float32 or mat.dtype != torch.float32:
        raise RuntimeError(
            "GEMS smm is only implemented for float32 inputs, matching torch.smm"
        )

    # Coalesce the sparse input so its structure is deterministic: duplicate
    # (row, col) entries are summed and the entries end up sorted by row, which
    # is what makes each row a contiguous COO segment below. The origin-tag
    # shim keeps the (possibly fresh) coalesced tensor tagged.
    if not self.is_coalesced():
        self = self.coalesce()

    nnz = self._nnz()
    if nnz == 0 or N == 0 or M == 0:
        # No structural rows (or no output columns) means an empty COO result.
        # The platform factory materializes COO tensors on CPU, so assemble the
        # empty result from CPU components like every other COO here.
        out_indices = torch.empty((2, 0), dtype=torch.int64)
        out_values = torch.empty((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(out_indices, out_values, size=(M, N))

    indices = self._indices()
    values = self._values()
    # The members may sit on the CPU (platform materialization of a device
    # workload) or already on the device (factory alias views); a plain move
    # covers both -- ``to`` is a no-op that returns self on the target device.
    # The whole (2, nnz) index pair moves as one buffer (half the H2D round
    # trips of moving the two rows separately) and both device copies are
    # cached on the source members, so repeat calls with the same sparse
    # operand skip the pageable copies entirely.
    indices_dev, values_dev = _cached_device_members(indices, values, device)
    row_indices = indices_dev[0]
    col_indices = indices_dev[1]

    # Build the per-row COO segments. ``is_start`` flags row boundaries, its
    # inclusive cumsum ranks them, and the compaction writes the row ids plus
    # the segment offsets. The ranking composes the FlagGems cumsum op (an
    # integer scan in this op's own device pipeline); the only thing that
    # leaves the device is the scalar boundary count readback, which the host
    # needs to size the output.
    is_start = torch.empty((nnz,), dtype=torch.int32, device=device)
    grid_nnz = lambda META: (triton.cdiv(nnz, META["BLOCK"]),)
    with torch_device_fn.device(device):
        smm_mark_row_starts_kernel[grid_nnz](row_indices, is_start, nnz, BLOCK=1024)
        import flag_gems

        rank_dtype = torch.int64 if nnz >= 2**31 else torch.int32
        rank = flag_gems.cumsum(is_start, 0, dtype=rank_dtype)
        nrows = int(rank[-1].item())
        row_ids = torch.empty((nrows,), dtype=torch.int64, device=device)
        row_starts = torch.empty((nrows + 1,), dtype=torch.int64, device=device)
        smm_compact_rows_kernel[grid_nnz](
            row_indices,
            is_start,
            rank,
            row_ids,
            row_starts,
            nnz,
            nrows,
            BLOCK=1024,
        )

        # ``torch.smm`` materializes every column of each structural row.
        nnz_out = nrows * N
        out_values = torch.empty((nnz_out,), device=device, dtype=torch.float32)
        out_indices = torch.empty((2, nnz_out), device=device, dtype=torch.int64)

        grid = lambda META: (nrows, triton.cdiv(N, META["BLOCK_N"]))
        smm_coo_kernel[grid](
            col_indices,
            values_dev,
            row_ids,
            row_starts,
            mat,
            out_indices[0],
            out_indices[1],
            out_values,
            N,
            mat.stride(0),
            mat.stride(1),
            CONTIG_N=mat.stride(1) == 1,
        )

        # The platform factory materializes the result on CPU. Handing it the
        # device outputs directly drives its D2H through the slow pageable
        # path, so stage the outputs through pinned host buffers (async DMA),
        # synchronize, and hand the filled buffers to the factory -- it
        # zero-copies them into the COO, so each call needs its own buffers
        # (the previous result keeps its pair alive through its references).
        host_indices = torch.empty((2, nnz_out), dtype=torch.int64, pin_memory=True)
        host_values = torch.empty((nnz_out,), dtype=torch.float32, pin_memory=True)
        host_indices.copy_(out_indices, non_blocking=True)
        host_values.copy_(out_values, non_blocking=True)
        torch_device_fn.synchronize()

    # Rows are ascending and each contributes its columns in ascending order, so
    # the output is already in coalesced order; declaring it avoids a re-sort.
    # The result is a COO on the CPU, which is where every COO tensor on this
    # platform lives.
    return torch.sparse_coo_tensor(
        host_indices, host_values, size=(M, N), is_coalesced=True
    )
