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

"""Triton implementation of ``aten::_sparse_compressed_tensor_unsafe``.

The unsafe compressed constructor assembles a CSR / CSC / BSR / BSC layout
tensor from already-materialised ``(compressed_indices, plain_indices,
values)`` components **without validating** the compressed indices — the only
difference from the validated ``torch.sparse_compressed_tensor`` constructor,
whose index checks are delegated to
``aten::_validate_compressed_sparse_indices``.

There is no data-parallel arithmetic in the operator itself; the only
device-side work a call can request is the dtype conversion of the payload,
which the Triton ``_cast_kernel`` below performs before the native assembly.

The native assembly is captured once at import time (the
``torch.library.get_kernel`` pattern used by ``ops/cholesky.py``): this
operator sits at the bottom of the unsafe-constructor family — the BSR / BSC /
CSR / CSC wrappers assemble through it — so calling it back through the
dispatcher after the FlagGems registration would recurse into this very
wrapper.  When ``torch.library.get_kernel`` is unavailable the plain call is
used instead, matching the behaviour of the sibling wrappers.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime

logger = logging.getLogger(__name__)

_BLOCK = 1024

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


def sparse_compressed_tensor_unsafe(
    compressed_indices,
    plain_indices,
    values,
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS SPARSE COMPRESSED TENSOR UNSAFE")
    if layout is None:
        raise RuntimeError(
            "sparse_compressed_tensor_unsafe(): specified layout must be one of "
            "torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc"
        )

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

    kwargs = dict(
        dtype=out_values.dtype if dtype is None else dtype,
        layout=layout,
        device=out_values.device if device is None else device,
        pin_memory=pin_memory,
    )
    if _NATIVE_COMPRESSED_UNSAFE is not None:
        # The captured native kernel is invoked directly, bypassing the
        # dispatcher so the FlagGems registration of this operator (and of the
        # per-layout unsafe constructors that assemble through it) cannot
        # recurse into this wrapper.
        return _NATIVE_COMPRESSED_UNSAFE.call_boxed(
            _DEVICE_KEYSET,
            compressed_indices,
            plain_indices,
            out_values,
            size,
            **kwargs,
        )
    return torch.ops.aten._sparse_compressed_tensor_unsafe(
        compressed_indices,
        plain_indices,
        out_values,
        size,
        **kwargs,
    )
