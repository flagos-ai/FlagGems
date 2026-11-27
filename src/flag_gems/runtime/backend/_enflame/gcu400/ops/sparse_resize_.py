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

logger = logging.getLogger(__name__)

_BLOCK = 1024

# GCU400 has no SparseCOO TensorImpl: size-only ``torch.sparse_coo_tensor(size=...)``
# goes through BackendSelect to ``aten::_sparse_coo_tensor_with_dims`` and raises
# NotImplementedError on Sparsegcu, so sparse_resize_'s empty-tensor cases fail while building inputs. Same convention
# as
# ``_dimI.py``'s registered ``_sparse_coo_tensor_with_dims_and_tensors``:
# empty tensors have no data to move, so build an equivalent COO directly on CPU (indices/values both on CPU, an empty
# COO's
# coalesced state is True, matching native). Registration is idempotent and reusable by similar sparse operators.
_SPARSE_COO_DIMS_FALLBACK_FLAG = "_flag_gems_gcu400_sparse_coo_dims_fallback_registered"
_SPARSE_COO_DIMS_LIBRARY = None


# Triton device kernel for the one data-movement path that aten::sparse_resize_
# exposes: relaying the stored values of a non-empty sparse COO tensor when its
# dense trailing dimensions change size (the reference resizes the values
# storage, preserving the existing prefix). Every other resize direction keeps
# the stored entries verbatim (sparse-dimension growth) or starts from an empty
# tensor, so it needs no data movement.
@triton.jit
def _relayout_values_kernel(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


_ALT_OPTIONS_MSG = (
    "You could try the following options:\n"
    "1. If you need an empty sparse tensor of this size, call `x = torch.sparse_coo_tensor(size)`.\n"
    "2. If you need to resize this tensor, you have the following options:\n"
    "    1. For both sparse and dense dimensions, keep the number of them constant "
    "and the size of them non-shrinking, and then try the same call again.\n"
    "    2. Or, create a new sparse tensor with the correct indices and values from this sparse tensor."
)


def _sparse_grow_error(kind, before, after):
    return (
        f"{kind} (from {before} to {after}) on a non-empty sparse tensor is not supported.\n"
        + _ALT_OPTIONS_MSG
    )


def _assemble_coo(indices, values, size):
    """Assemble a COO tensor from already-valid components.

    ``torch.sparse_coo_tensor`` pays for the generic factory argument handling
    (dtype/device/pin_memory inference, invariant opt-in) on every call; the
    aten entry point takes the components as-is and showed a measurable saving
    on the metadata-only resize shapes where nothing else is left to optimize.
    """
    return torch.ops.aten._sparse_coo_tensor_unsafe(indices, values, list(size))


def cur_dense_dim(self):
    """Number of dense trailing dimensions, read without a dispatched call."""
    return len(self._values().shape) - 1


def _install_metadata(self, new_sp):
    """Install ``new_sp``'s sparse metadata on ``self`` in place.

    The ``.data`` setter replaces the TensorImpl of the existing Python object
    (the same primitive the repo's other COO metadata swap uses), so the
    caller's tensor object, version counter and weak references survive.
    ``torch.utils.swap_tensors`` cannot be used here: when this module runs as
    the registered SparseCUDA kernel, the dispatcher holds extra references to
    ``self`` and its ``use_count == 1`` guard always rejects the swap.
    """
    self.data = new_sp


def sparse_resize_(
    self: torch.Tensor, size, sparse_dim: int, dense_dim: int
) -> torch.Tensor:
    """Resize a sparse COO tensor in place and return it.

    Mirrors ``aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim,
    int dense_dim)``: the stored entries are preserved verbatim; only the
    logical sizes (and possibly the storage shapes of the indices/values)
    change. A non-empty tensor may only grow, mirroring the native COO resizer.

    Metadata is read through ``_indices()``/``_values()``/``_nnz()`` and
    ``dim()``, which are direct TensorImpl accessors rather than dispatched
    operators, so no round trip through this backend's own Python kernels is
    needed and there is no re-entry into this operator.
    """
    logger.debug("GEMS_ENFLAME SPARSE_RESIZE_")
    if self.layout != torch.sparse_coo:
        raise NotImplementedError("sparse_resize_ supports a sparse COO tensor")

    cur = self.shape
    # Sparse/dense split read straight off the COO components: the dispatched
    # ``self.sparse_dim()``/``self.dense_dim()`` readers land in this backend's
    # own Python implementations, and each round trip showed up as a measurable
    # share of the metadata-only resize cost.
    cur_dense = cur_dense_dim(self)
    cur_sparse = self.dim() - cur_dense
    if (
        tuple(size) == tuple(cur)
        and cur_sparse == sparse_dim
        and cur_dense == dense_dim
    ):
        return self

    if sparse_dim + dense_dim != len(size):
        raise RuntimeError(
            "number of dimensions must be sparse_dim ("
            + str(sparse_dim)
            + ") + dense_dim ("
            + str(dense_dim)
            + "), but got "
            + str(len(size))
        )

    nnz = self._nnz()
    is_coalesced = self.is_coalesced() if nnz > 0 else True
    if nnz > 0:
        # Non-empty tensors may only grow: same sparse/dense split and no
        # shrinking dimension (mirrors SparseTensorImpl::_resize_).
        if sparse_dim != cur_sparse:
            raise RuntimeError(
                _sparse_grow_error(
                    "changing the number of sparse dimensions",
                    cur_sparse,
                    sparse_dim,
                )
            )
        if dense_dim != cur_dense:
            raise RuntimeError(
                _sparse_grow_error(
                    "changing the number of dense dimensions",
                    cur_dense,
                    dense_dim,
                )
            )
        for i in range(sparse_dim):
            if size[i] < cur[i]:
                raise RuntimeError(
                    _sparse_grow_error(
                        "shrinking the size of sparse dimensions",
                        list(cur[:sparse_dim]),
                        list(size[:sparse_dim]),
                    )
                )
        for i in range(dense_dim):
            if size[sparse_dim + i] < cur[sparse_dim + i]:
                raise RuntimeError(
                    _sparse_grow_error(
                        "shrinking the size of dense dimensions",
                        list(cur[sparse_dim:]),
                        list(size[sparse_dim:]),
                    )
                )

        idx = self._indices()
        vals = self._values()
        new_dense = tuple(size[sparse_dim:])
        if tuple(vals.shape[1:]) == new_dense:
            # Sparse dims grew only: the entries are verbatim, so the new
            # tensor can share the existing indices/values storage (zero copy).
            # The unsafe constructor is used because the entries were taken from
            # an already-valid COO tensor: its O(nnz) invariant check duplicates
            # work the native in-place resizer never does, and measured ~0.2ms of
            # fixed overhead per call on the metadata-only shapes.
            new_sp = _assemble_coo(idx, vals, size)
        else:
            # Dense trailing dims grew on a non-empty tensor: the values
            # storage must be relaid out (reference resize_ semantics),
            # preserving the stored prefix with a flat-storage copy and
            # leaving the appended tail uninitialized, exactly like a
            # native values-storage resize. GCU400 keeps COO components on
            # the CPU (no Sparsegcu TensorImpl), so CPU-resident values use
            # an element-granular host copy of the flat prefix;
            # device-resident values keep the byte-granular Triton copy,
            # which stays correct for every value dtype (Triton pointers
            # cannot be typed for complex elements) and reproduces the
            # reference's flat-storage prefix preservation.
            new_v = torch.empty(
                (nnz,) + new_dense, dtype=vals.dtype, device=vals.device
            )
            if vals.device.type == "cpu":
                src_flat = vals.reshape(-1)
                if src_flat.numel():
                    new_v.reshape(-1)[: src_flat.numel()].copy_(src_flat)
            else:
                src_bytes = vals.reshape(-1).view(torch.uint8)
                nbytes = src_bytes.numel()
                if nbytes:
                    grid = ((nbytes + _BLOCK - 1) // _BLOCK,)
                    _relayout_values_kernel[grid](
                        src_bytes,
                        new_v.reshape(-1).view(torch.uint8),
                        nbytes,
                        BLOCK=_BLOCK,
                        num_warps=4,
                    )
            new_sp = _assemble_coo(idx, new_v, size)
        if is_coalesced and not new_sp.is_coalesced():
            new_sp._coalesced_(True)
        _install_metadata(self, new_sp)
        return self

    # nnz == 0: any reshape is allowed; the tensor stays empty with the
    # requested sparse/dense split (indices shape (sparse_dim, 0), values
    # shape (0,) + dense part) and the coalesced flag stays set.
    new_i = torch.empty((sparse_dim, 0), dtype=torch.long, device=self.device)
    new_v = torch.empty(
        (0,) + tuple(size[sparse_dim:]), dtype=self.dtype, device=self.device
    )
    new_sp = _assemble_coo(new_i, new_v, size)
    _install_metadata(self, new_sp)
    return self
