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

from flag_gems.runtime.backend._enflame.gcu400.ops.copy import copy_
from flag_gems.runtime.backend._enflame.gcu400.ops.values_copy import _coo_values_put

logger = logging.getLogger(__name__)


@triton.jit
def _copy_fused(si, di, ni, sv, dv, nv, split, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid < split:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < ni
        tl.store(di + offs, tl.load(si + offs, mask=mask), mask=mask)
    else:
        pid2 = pid - split
        offs = pid2 * BLOCK + tl.arange(0, BLOCK)
        mask = offs < nv
        tl.store(dv + offs, tl.load(sv + offs, mask=mask), mask=mask)


# Unmasked variant: used only when both streams are exactly BLOCK-divisible
# (true for every benchmark workload). Consistently ~0.5-0.7us faster on the
# fp32 indices-heavy large workloads in interleaved A/B screening.
@triton.jit
def _copy_fused_nomask(si, di, ni, sv, dv, nv, split, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid < split:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(di + offs, tl.load(si + offs))
    else:
        pid2 = pid - split
        offs = pid2 * BLOCK + tl.arange(0, BLOCK)
        tl.store(dv + offs, tl.load(sv + offs))


_NW = 4
# A GCU400 program costs much more than the elements it moves, so the fused copy
# wants few wide programs: measured on nnz = 64 / 4 K / 64 K / 256 K / 1 M the
# best tile is 1 K / 1 K / 16 K / 32 K / 32 K elements, i.e. a quarter of the
# stream clamped to [1 K, 32 K]. That is up to 1.8x the ATen kernel and 11x the
# old 1 K-wide tile at nnz = 1 M.
_BLOCK_FUSED_MIN = 1024
_BLOCK_FUSED_MAX = 32768


def _fused_block(ni, nv):
    stream = max(ni, nv)
    block = _BLOCK_FUSED_MIN
    while block < _BLOCK_FUSED_MAX and block * 4 < stream:
        block *= 2
    return block


def copy_sparse_to_sparse_(self, src, non_blocking=False):
    logger.debug("GEMS_ENFLAME COPY_SPARSE_TO_SPARSE_")
    if self.layout != torch.sparse_coo or src.layout != torch.sparse_coo:
        raise RuntimeError("copy_sparse_to_sparse_ expects sparse COO tensors")
    if self.device != src.device:
        raise RuntimeError("source and destination must be on the same device")
    if torch.is_grad_enabled() and self.is_leaf and self.requires_grad:
        raise RuntimeError(
            "a leaf Variable that requires grad is being used in an in-place operation"
        )
    if self is src:
        return self

    s_idx = src._indices()
    s_val = src._values()

    if (
        self._nnz() != src._nnz()
        or self.shape != src.shape
        or self.sparse_dim() != src.sparse_dim()
        or self.dense_dim() != src.dense_dim()
    ):
        # Structure differs: grow / retype ``self`` to match ``src`` and then copy.
        if self._nnz() != 0 and any(
            src.shape[dim] < self.shape[dim] for dim in range(self.dim())
        ):
            raise RuntimeError(
                "shrinking the size of sparse dimensions (from %s to %s) on a "
                "non-empty sparse tensor is not supported."
                % (list(self.shape), list(src.shape))
            )
        # The growth cannot be done in place: torch_gcu drops ``resize_`` on the
        # COO index/value buffers (measured: growing the indices from (2, 2) to
        # (2, 3) leaves them at (2, 2)), so a resize-then-copy sequence silently
        # keeps the old nnz. Copy into fresh buffers and install them through the
        # sparse metadata slot instead, which preserves the destination Python
        # object, its version counter and existing weak references.
        new_idx = torch.empty_like(s_idx)
        new_val = torch.empty(s_val.shape, dtype=self.dtype, device=s_val.device)
        ni = s_idx.numel()
        nv = s_val.numel()
        if not s_idx.is_contiguous() or not s_val.is_contiguous():
            copy_(new_idx, s_idx, non_blocking)
            copy_(new_val, s_val, non_blocking)
        elif ni + nv > 0:
            BLOCK = _fused_block(ni, nv)
            split = (ni + BLOCK - 1) // BLOCK
            _copy_fused[(split + (nv + BLOCK - 1) // BLOCK,)](
                s_idx,
                new_idx,
                ni,
                s_val,
                new_val,
                nv,
                split,
                BLOCK=BLOCK,
                num_warps=_NW,
            )
        # The device is taken from the freshly allocated buffers: on torch_gcu a
        # sparse COO tensor reports cpu through ``.device``.
        # The coalesced flag travels with the new metadata, so it must not be set
        # again afterwards: the extra ``_coalesced_(False)`` call rebuilds the
        # container and re-poisons the value lookup (see below).
        rebuilt = torch.sparse_coo_tensor(
            new_idx,
            new_val,
            tuple(src.shape),
            is_coalesced=src.is_coalesced(),
        )
        self.data = rebuilt
        # GCU400 cannot materialize a sparse-COO TensorImpl: the container is a
        # cpu tensor while the caller-visible values live in the process-wide
        # cache of ``values_copy.py``, keyed by ``id(container)``. Replacing the
        # data slot keeps ``id(self)`` alive, so that entry still points at the
        # pre-grow buffer and ``_values()`` keeps reporting the old nnz. Re-point
        # it at the freshly built device buffer.
        _coo_values_put(self, new_val)
        return self

    d_idx = self._indices()
    d_val = self._values()
    ni = s_idx.numel()
    nv = s_val.numel()
    if not all(t.is_contiguous() for t in (s_idx, s_val, d_idx, d_val)):
        copy_(d_idx, s_idx, non_blocking)
        copy_(d_val, s_val, non_blocking)
    elif ni + nv > 0:
        BLOCK = _fused_block(ni, nv)
        split = (ni + BLOCK - 1) // BLOCK
        grid = (split + (nv + BLOCK - 1) // BLOCK,)
        if BLOCK > _BLOCK_FUSED_MIN and ni % BLOCK == 0 and nv % BLOCK == 0:
            _copy_fused_nomask[grid](
                s_idx,
                d_idx,
                ni,
                s_val,
                d_val,
                nv,
                split,
                BLOCK=BLOCK,
                num_warps=_NW,
            )
        else:
            _copy_fused[grid](
                s_idx,
                d_idx,
                ni,
                s_val,
                d_val,
                nv,
                split,
                BLOCK=BLOCK,
                num_warps=_NW,
            )
    self._coalesced_(src.is_coalesced())
    return self
