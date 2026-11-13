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
#
"""GCU implementation of ``aten::sparse_bsc_tensor`` plus its support shims.

The constructor itself assembles a ``torch.sparse_bsc`` tensor through
``aten::_sparse_compressed_tensor_unsafe`` (the sibling unsafe constructor
pattern): FlagGems registers this very function for
``sparse_bsc_tensor.ccol_row_value(.size)`` when enabled, so re-entering the
dispatched ``torch.sparse_bsc_tensor`` from inside it would recurse.

On the GCU backend the sparse-compressed infrastructure of torch_gcu is
incomplete, which otherwise makes every consumer of a BSC tensor fail before
this operator can even run:

* ``ccol_indices`` / ``row_indices`` / ``values`` have kernels only for the
  ``SparseCsr{CPU,CUDA,Meta}`` keys; a GCU sparse-compressed tensor carries
  ``SparseCsrPrivateUse1`` and falls through to PyTorch's composite defaults,
  which reject every layout ("ccol_indices expected sparse column compressed
  tensor layout but got SparseBsc").
* ``aten::_to_dense`` has no kernel for the ``SparseCsrgcu`` key either.
* ``Tensor.to_sparse_bsc`` produces a native GCU BSC tensor whose member
  tensors cannot be introspected from Python at all (no accessor, no
  ``.cpu()``, no repr) - a verified black box.

This module therefore registers, at import time, GCU-key kernels that close
those gaps for tensors assembled through this file:

- ``aten::to_sparse_bsc`` (PrivateUse1) converts via the CPU native path and
  reassembles the components on the device;
- ``aten::ccol_indices`` / ``aten::row_indices`` / ``aten::values``
  (SparseCsrPrivateUse1) serve the remembered components;
- ``aten::_to_dense`` (SparseCsrPrivateUse1) scatters the block values into a
  dense tensor with a Triton kernel (GCU ``index_put`` rejects the advanced
  indexing form this scatter needs and aborts the device).

Every kernel falls back to the original native error for tensors it does not
know (registry miss), so CSR/BSR/CSC/COO behaviour is unchanged.
"""

import logging
import weakref
from contextlib import contextmanager

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Enflame registers its backend on the PrivateUse1 slot (renamed "gcu"), so
# the sparse-compressed alias key FlagGems spells "SparseCsr" + dispatch key.
_DEVICE_KEY = "PrivateUse1"
_SPARSE_CSR_GCU_KEY = "SparseCsrPrivateUse1"

# C++ layout spellings used by native ATen in its rejection messages.
_LAYOUT_NAMES = {
    torch.strided: "Strided",
    torch.sparse_coo: "Sparse",
    torch.sparse_csr: "SparseCsr",
    torch.sparse_csc: "SparseCsc",
    torch.sparse_bsr: "SparseBsr",
}

# The dispatcher's own message when no kernel exists for the SparseCsrgcu
# key, mirrored verbatim (the values.py sibling mirrors its message the same
# way) so the registry-miss path stays byte-identical to native behaviour.
_TO_DENSE_NATIVE_ERROR = (
    "Could not run 'aten::_to_dense' with arguments from the 'SparseCsrgcu' "
    "backend. This could be because the operator doesn't exist for this "
    "backend, or was omitted during the selective/custom build process (if "
    "using custom build). If you are a Facebook employee using PyTorch on "
    "mobile, please visit https://fburl.com/ptmfixes for possible resolutions."
)

# id(tensor) -> (ccol_indices, row_indices, values) for every BSC tensor this
# module assembled. Entries die with their tensor (weakref finalizer), so an id
# is never served stale.
#
# The backend loader executes this file under several module names
# (flag_gems.runtime.backend._enflame.gcu400.ops..., _enflame.gcu400.ops...,
# gcu400.ops...), and the dispatch kernels may belong to a different copy
# than the function bound into the flag_gems namespace. The registry therefore
# lives on the torch module singleton so every copy shares one dict.
_BSC_STATE_ATTR = "_flaggems_gcu_bsc_components"
_BSC_COMPONENTS = getattr(torch, _BSC_STATE_ATTR, None)
if _BSC_COMPONENTS is None:
    _BSC_COMPONENTS = {}
    setattr(torch, _BSC_STATE_ATTR, _BSC_COMPONENTS)

# Marks that the support kernels below were registered once for this process;
# duplicate module copies skip silently instead of raising.
_BSC_KERNELS_ATTR = "_flaggems_gcu_bsc_kernels_registered"

_CAST_BLOCK = 1024


@triton.jit
def _cast_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(src_ptr + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, x.to(dst_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bsc_to_dense_kernel(
    ri_ptr,
    col_ptr,
    values_ptr,
    out_ptr,
    ncols,
    nrows,
    stride_k,
    stride_r,
    stride_c,
    BS_R: tl.constexpr,
    BS_C: tl.constexpr,
    PAD_R: tl.constexpr,
    PAD_C: tl.constexpr,
):
    # One program per stored block: copy its BS_R x BS_C payload into the
    # dense output at the block position. Padded ranges are masked off, so
    # non-power-of-two block sizes and the matrix edges stay in bounds.
    k = tl.program_id(0)
    row0 = tl.load(ri_ptr + k) * BS_R
    col0 = tl.load(col_ptr + k) * BS_C
    offs_r = tl.arange(0, PAD_R)
    offs_c = tl.arange(0, PAD_C)
    rows = row0 + offs_r
    cols = col0 + offs_c
    mask = (offs_r < BS_R)[:, None] & (offs_c < BS_C)[None, :]
    mask = mask & (rows < nrows)[:, None] & (cols < ncols)[None, :]
    block = tl.load(
        values_ptr
        + k * stride_k
        + offs_r[:, None] * stride_r
        + offs_c[None, :] * stride_c,
        mask=mask,
        other=0.0,
    )
    tl.store(out_ptr + rows[:, None] * ncols + cols[None, :], block, mask=mask)


def _remember(bsc_tensor, ccol_indices, row_indices, values):
    _BSC_COMPONENTS[id(bsc_tensor)] = (ccol_indices, row_indices, values)
    weakref.finalize(bsc_tensor, _BSC_COMPONENTS.pop, id(bsc_tensor), None)


@contextmanager
def _excluded_key(key):
    torch._C._dispatch_tls_set_dispatch_key_excluded(key, True)
    try:
        yield
    finally:
        torch._C._dispatch_tls_set_dispatch_key_excluded(key, False)


def sparse_bsc_tensor(
    ccol_indices,
    row_indices,
    values,
    size=None,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
):
    logger.debug("GEMS SPARSE_BSC_TENSOR")
    if layout is not None and layout != torch.sparse_bsc:
        raise RuntimeError(
            "sparse_bsc_tensor: layout must be SparseBsc but got "
            + _LAYOUT_NAMES.get(layout, str(layout))
        )
    return _assemble(ccol_indices, row_indices, values, size, dtype, device, pin_memory)


def _assemble(ccol_indices, row_indices, values, size, dtype, device, pin_memory):
    out_values = values
    if dtype is not None and out_values.dtype != dtype:
        # The validated native constructor converts the payload to the
        # requested dtype; the unsafe one rejects a mismatch, so cast first
        # (same Triton cast the sibling unsafe operator uses).
        out_values = torch.empty(
            out_values.shape, dtype=dtype, device=out_values.device
        )
        numel = values.numel()
        grid = (triton.cdiv(numel, _CAST_BLOCK),)
        _cast_kernel[grid](values, out_values, numel, BLOCK=_CAST_BLOCK)
    if device is not None and out_values.device != torch.device(device):
        out_values = out_values.to(device)
        ccol_indices = ccol_indices.to(device)
        row_indices = row_indices.to(device)
    if size is None:
        # Native inference rule: just large enough for every stored block.
        block_r, block_c = out_values.shape[-2:]
        nrows = (
            (int(row_indices.max().item()) + 1) * block_r if row_indices.numel() else 0
        )
        size = [nrows, (ccol_indices.numel() - 1) * block_c]
    else:
        size = list(size)
    torch.ops.aten._validate_sparse_bsc_tensor_args(
        ccol_indices, row_indices, out_values, size, None
    )
    bsc = torch.ops.aten._sparse_compressed_tensor_unsafe(
        ccol_indices,
        row_indices,
        out_values,
        size,
        dtype=out_values.dtype,
        layout=torch.sparse_bsc,
        device=out_values.device,
        pin_memory=pin_memory,
    )
    _remember(bsc, ccol_indices, row_indices, out_values)
    return bsc


def _known_bsc_components(self):
    if self.layout != torch.sparse_bsc:
        return None
    return _BSC_COMPONENTS.get(id(self))


def _ccol_indices_gcu(self):
    members = _known_bsc_components(self)
    if members is not None:
        return members[0]
    with _excluded_key(_SPARSE_CSR_GCU_KEY):
        return torch.Tensor.ccol_indices(self)


def _row_indices_gcu(self):
    members = _known_bsc_components(self)
    if members is not None:
        return members[1]
    with _excluded_key(_SPARSE_CSR_GCU_KEY):
        return torch.Tensor.row_indices(self)


def _values_gcu(self):
    members = _known_bsc_components(self)
    if members is not None:
        return members[2]
    with _excluded_key(_SPARSE_CSR_GCU_KEY):
        return torch.Tensor.values(self)


def _to_dense_gcu(self, dtype=None, masked_grad=None):
    members = _known_bsc_components(self)
    if members is None or self.dim() != 2 or members[2].dim() != 3:
        # Unknown tensor (or shape this scatter does not model): keep the
        # native dispatcher error instead of guessing.
        raise NotImplementedError(_TO_DENSE_NATIVE_ERROR)
    ccol_indices, row_indices, values = members
    nrows, ncols = self.shape
    block_r, block_c = values.shape[-2:]
    out = torch.zeros(
        (nrows, ncols),
        dtype=self.dtype if dtype is None else dtype,
        device=self.device,
    )
    nnz = row_indices.numel()
    if nnz:
        n_column_blocks = ccol_indices.numel() - 1
        # Keep the kernel inputs in the int32 safe domain: block indices and
        # matrix extents stay far below 2**31, and gcu400 Triton cannot rely
        # on int64 pointer math unless TORCH_GCU_ENABLE_INT64_AND_UINT64 is
        # enabled for the whole process.
        column_of_entry = torch.repeat_interleave(
            torch.arange(
                n_column_blocks, device=ccol_indices.device, dtype=torch.int32
            ),
            torch.diff(ccol_indices),
        )
        _bsc_to_dense_kernel[(nnz,)](
            row_indices.to(torch.int32),
            column_of_entry,
            values,
            out,
            ncols,
            nrows,
            values.stride(0),
            values.stride(1),
            values.stride(2),
            BS_R=block_r,
            BS_C=block_c,
            PAD_R=triton.next_power_of_2(block_r),
            PAD_C=triton.next_power_of_2(block_c),
        )
    return out


def _to_sparse_bsc_gcu(self, blocksize, dense_dim=None):
    if self.layout != torch.strided:
        with _excluded_key(_DEVICE_KEY):
            return torch.Tensor.to_sparse_bsc(self, blocksize, dense_dim)
    # The native GCU conversion result is not introspectable (see module
    # docstring), so derive the components with the CPU native path and
    # reassemble them on the device; values keep their exact bit pattern.
    dense_cpu = self.detach().to("cpu")
    bsc_cpu = dense_cpu.to_sparse_bsc(blocksize, dense_dim)
    ccol_indices = bsc_cpu.ccol_indices().to(self.device)
    row_indices = bsc_cpu.row_indices().to(self.device)
    values = bsc_cpu.values().to(self.device)
    return _assemble(
        ccol_indices, row_indices, values, list(bsc_cpu.shape), None, None, None
    )


def _register_support_kernels():
    if getattr(torch, _BSC_KERNELS_ATTR, False):
        return None
    lib = torch.library.Library("aten", "IMPL")
    try:
        lib.impl("ccol_indices", _ccol_indices_gcu, _SPARSE_CSR_GCU_KEY)
        lib.impl("row_indices", _row_indices_gcu, _SPARSE_CSR_GCU_KEY)
        lib.impl("values", _values_gcu, _SPARSE_CSR_GCU_KEY)
        lib.impl("_to_dense", _to_dense_gcu, _SPARSE_CSR_GCU_KEY)
        lib.impl("to_sparse_bsc", _to_sparse_bsc_gcu, _DEVICE_KEY)
    except RuntimeError as err:
        # Another module (maybe outside this file) already owns one of these
        # slots; surface that instead of swallowing it silently.
        logger.warning("sparse_bsc_tensor support kernels: %s", err)
        return lib
    setattr(torch, _BSC_KERNELS_ATTR, True)
    return lib


_support_lib = _register_support_kernels()
