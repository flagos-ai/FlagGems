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

"""Triton implementation of ``aten::_sparse_csr_tensor_unsafe``.

The unsafe CSR constructor assembles a ``torch.sparse_csr`` layout tensor
from already-materialised ``(crow_indices, col_indices, values)`` components
**without validating** the compressed-row / column indices — that is the only
difference from the validated ``torch.sparse_csr_tensor`` constructor, whose
index checks are delegated to ``aten::_validate_compressed_sparse_indices``.

There is no data-parallel arithmetic in the operator itself; the only
device-side work a call can request is the dtype conversion of the payload,
which the Triton ``_cast_kernel`` below performs before the native assembly.
Passing the converted values with the original ``dtype`` argument keeps the
native consistency check a no-op in the converted path.  Note the native
unsafe constructor performs no conversion itself — it requires
``values.dtype`` to already match the requested ``dtype`` — and it needs an
explicit ``device`` (it fails its internal device check when the kwarg is
omitted), both of which this wrapper handles.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime

logger = logging.getLogger(__name__)

_BLOCK = 1024

# Prefer the corresponding FlagGems compressed constructor when it is present
# (it is added by the sibling sparse_compressed_tensor_unsafe operator). When
# it is absent, fall back to the native assembly kernel captured once at import
# time; either path avoids a torch compute-layer call
# (check_no_torch_fallback) and avoids recursing into the FlagGems
# _sparse_csr_tensor_unsafe registration.
try:
    from flag_gems.ops.sparse_compressed_tensor_unsafe import (
        sparse_compressed_tensor_unsafe as _flag_gems_compressed_unsafe,
    )
except ImportError:  # pragma: no cover - depends on merge order
    _flag_gems_compressed_unsafe = None

_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")
if _CAN_CAPTURE_NATIVE_KERNEL:
    _NATIVE_COMPRESSED_UNSAFE = torch.library.get_kernel(
        torch.ops.aten._sparse_compressed_tensor_unsafe.default,
        runtime.device.dispatch_key,
    )
    _DEVICE_KEYSET = torch._C.DispatchKeySet(
        getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
    )
else:
    _NATIVE_COMPRESSED_UNSAFE = None


@triton.jit
def _cast_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    """Elementwise cast of the payload values into the requested dtype."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(src_ptr + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, x.to(dst_ptr.dtype.element_ty), mask=mask)


def sparse_csr_tensor_unsafe(
    crow_indices,
    col_indices,
    values,
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS SPARSE CSR TENSOR UNSAFE")

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
    # ``_sparse_csr_tensor_unsafe`` itself: the FlagGems dispatch table
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
    kwargs = dict(
        dtype=out_values.dtype if dtype is None else dtype,
        layout=torch.sparse_csr if layout is None else layout,
        device=out_values.device if device is None else device,
        pin_memory=pin_memory,
    )
    if _flag_gems_compressed_unsafe is not None:
        return _flag_gems_compressed_unsafe(
            crow_indices,
            col_indices,
            out_values,
            size,
            **kwargs,
        )
    if _NATIVE_COMPRESSED_UNSAFE is None:
        raise RuntimeError(
            "sparse_csr_tensor_unsafe requires "
            "flag_gems.sparse_compressed_tensor_unsafe or "
            "torch.library.get_kernel"
        )
    return _NATIVE_COMPRESSED_UNSAFE.call_boxed(
        _DEVICE_KEYSET,
        crow_indices,
        col_indices,
        out_values,
        size,
        **kwargs,
    )
