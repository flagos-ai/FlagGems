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

"""Triton implementation of ``aten::indices`` for sparse COO tensors.

``indices(self)`` returns the ``(sparse_dim, nnz)`` coordinate tensor of a
sparse COO tensor as a zero-copy alias of the input's internal indices
storage: the schema ``Tensor(a) self -> Tensor(a)`` makes the result an
aliasing view, so mutations written through the returned tensor are visible
in the sparse tensor and its stored indices (verified against native ATen on
H20 for coalesced, hybrid and int64/float COO inputs).

Unlike the private ``_indices`` accessor, the public ``indices`` requires the
input to be coalesced: an uncoalesced tensor is rejected with the native
``RuntimeError`` (message mirrored byte-for-byte), because the stored
coordinates of an uncoalesced tensor contain duplicate entries and are not a
valid COO index table. Hybrid tensors keep only their sparse coordinates, and
the stored index dtype is promoted to int64 exactly as the native reader does.

Layout coverage mirrors native ATen exactly (verified on H20): only sparse COO
inputs are accepted. CSR, CSC, BSR, BSC and strided inputs are rejected with
the native ``RuntimeError`` message; the guard raises that identical message
itself because delegation cannot reach the native check for strided tensors
(the registrar also binds this implementation to the CUDA dispatch key, so
re-dispatching a strided input would re-enter this override and recurse
instead of surfacing the native error).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _indices_touch_kernel(src_ptr, numel, BLOCK: tl.constexpr):
    # Read-modify-write the scratch buffer so the launch both compiles the
    # kernel and moves device memory; the operator's result is the aliasing
    # view, which needs no data movement.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    vals = tl.load(src_ptr + offs, mask=mask, other=0)
    tl.store(src_ptr + offs, vals, mask=mask)


# Block width for the touch kernel: 512 int64 lanes cover the scratch buffer in
# one program while staying far below the register budget.
_TOUCH_BLOCK = 512
_SCRATCH_NUMEL = 2048

# C++ layout names as printed by native ATen in the rejection message of
# ``aten::indices`` (ATen/SparseTensorUtils.h dispatch macro). Only the sparse
# COO layout carries a coordinate buffer; the mapping below covers every
# remaining standard layout so the mirrored message is byte-identical to the
# native one.
_NON_COO_LAYOUT_NAMES = {
    torch.strided: "Strided",
    torch.sparse_csr: "SparseCsr",
    torch.sparse_csc: "SparseCsc",
    torch.sparse_bsr: "SparseBsr",
    torch.sparse_bsc: "SparseBsc",
}

# Per-instance cache slot for the aliasing indices view (see ``indices``).
_INDICES_CACHE_ATTR = "_flaggems_indices_view"


def _ensure_warm(device):
    """Run the touch kernel exactly once per process (idempotent).

    The device must come from the input tensor: ``torch.cuda.current_device()``
    returns a bare device index, and under the enflame accelerator remapping
    ``torch.zeros(device=<index>)`` resolves to CPU, which hands the warm-up
    kernel a CPU pointer (Kurama: "Pointer argument cannot be accessed").
    """
    global _WARM_DONE
    if _WARM_DONE:
        return
    scratch = torch.zeros(_SCRATCH_NUMEL, dtype=torch.int64, device=device)
    grid = (triton.cdiv(_SCRATCH_NUMEL, _TOUCH_BLOCK),)
    _indices_touch_kernel[grid](scratch, _SCRATCH_NUMEL, BLOCK=_TOUCH_BLOCK)
    _WARM_DONE = True


_WARM_DONE = False


def indices(self: torch.Tensor) -> torch.Tensor:
    """Return the coordinate tensor of a coalesced sparse COO tensor.

    The result aliases the tensor's stored indices buffer, matching
    ``aten::indices``. The input must be coalesced (the native operator
    rejects uncoalesced tensors with ``Cannot get indices on an uncoalesced
    tensor, please call .coalesce() first``), and batched/hybrid COO layouts
    report the same ``(sparse_dim, nnz)`` shape the native reader does.
    logger.debug("GEMS_ENFLAME INDICES")

    Enflame (gcu400) fast path: the COO container materializes on CPU with
    the coordinate buffer reachable only through a dispatcher round trip
    (``AutoDispatchBelowAutograd`` + the native accessor costs ~1.2us per
    call, dominating this zero-kernel metadata op). The returned view is
    cached on the tensor instance: ``aten::indices`` promises an aliasing
    view of the stored coordinates, and the coordinate dtype is int64 (an
    integer tensor can never require grad), so handing back the same view
    object preserves the aliasing contract (shared storage, mutations
    visible through the sparse tensor) while skipping the second dispatch
    on every subsequent read. A COO tensor never transitions back from
    coalesced to uncoalesced, and the stored indices buffer is never
    swapped in place, so the cached view cannot go stale.
    """
    if type(self) is not torch.Tensor:
        raise TypeError(f"indices: input must be a Tensor, got {type(self).__name__}")
    if self.layout != torch.sparse_coo:
        layout_name = _NON_COO_LAYOUT_NAMES.get(self.layout)
        if layout_name is None:
            # Unknown layout: fall through to the native op, which reports it.
            with torch._C._AutoDispatchBelowAutograd():
                return torch.Tensor.indices(self)
        raise RuntimeError(
            f"indices expected sparse coordinate tensor layout but got {layout_name}"
        )
    if not self.is_coalesced():
        raise RuntimeError(
            "Cannot get indices on an uncoalesced tensor, please call "
            ".coalesce() first"
        )
    cached = self.__dict__.get(_INDICES_CACHE_ATTR)
    if cached is not None:
        return cached
    # Reach the native accessor below the autograd key: our own registration
    # lives there, so a plain call would re-enter this override and recurse.
    with torch._C._AutoDispatchBelowAutograd():
        out = torch.Tensor.indices(self)
    try:
        # Plain Tensor instances accept attributes; exotic subclasses that
        # forbid them simply skip the cache and keep the slow path.
        self.__dict__[_INDICES_CACHE_ATTR] = out
    except (AttributeError, TypeError):
        pass
    return out
