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

"""Kunlunxin override of ``aten::_sparse_bsc_tensor_unsafe``.

Identical to the generic implementation (``flag_gems.ops.sparse_bsc_tensor_unsafe``)
except for one platform detail: the dtype passed to the native assembly.

XPyTorch silently downgrades any request for float64 to float32 -- both for
``torch.empty(dtype=torch.float64)`` (which allocates float32) and for
``Tensor.to(torch.float64)`` -- so after the conversion kernel the payload's
actual dtype can differ from the requested one.  The native unsafe constructor
requires ``values.dtype`` to match the requested ``dtype`` exactly and raises
``RuntimeError: dtype of values (Float) must match dtype of sparse tensor
(Double)`` otherwise.  Passing ``out_values.dtype`` (the effective, post-
conversion dtype) instead of the requested one keeps the assembly valid; the
resulting sparse tensor holds the converted payload, which is the closest
representable value on a platform without float64.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK = 1024


@triton.jit
def _cast_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    """Elementwise cast of the payload values into the requested dtype."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(src_ptr + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, x.to(dst_ptr.dtype.element_ty), mask=mask)


def sparse_bsc_tensor_unsafe(
    ccol_indices,
    row_indices,
    values,
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS_KUNLUNXIN SPARSE_BSC_TENSOR_UNSAFE")

    out_values = values
    if dtype is not None and out_values.dtype != dtype:
        out_values = torch.empty(
            out_values.shape, dtype=dtype, device=out_values.device
        )
        n = values.numel()
        grid = (triton.cdiv(n, _BLOCK),)
        _cast_kernel[grid](values, out_values, n, BLOCK=_BLOCK)

    if device is not None and out_values.device != torch.device(device):
        out_values = out_values.to(device)

    # Assemble through the sibling unsafe compressed constructor rather than
    # ``_sparse_bsc_tensor_unsafe`` itself: the FlagGems dispatch table
    # overrides this operator, so re-entering it here would recurse into this
    # wrapper.  Both constructors are the same underlying assembly (verified
    # bit-identical for valid, invalid and empty components).
    #
    # The unsafe constructor infers neither the target device nor the dtype
    # from the values: without an explicit device kwarg it fails the internal
    # values/compressed-instance device check, and without a dtype kwarg it
    # defaults the sparse tensor to float32 and then rejects any other values
    # dtype.  Pass the (possibly converted) values' device and dtype whenever
    # the caller did not specify one.
    #
    # ``dtype`` uses the *effective* values dtype (not the requested one):
    # XPyTorch downgrades a requested float64 to float32, and the native
    # constructor requires an exact match (see the module docstring).
    return torch.ops.aten._sparse_compressed_tensor_unsafe(
        ccol_indices,
        row_indices,
        out_values,
        size,
        dtype=out_values.dtype,
        layout=torch.sparse_bsc if layout is None else layout,
        device=out_values.device if device is None else device,
        pin_memory=pin_memory,
    )
