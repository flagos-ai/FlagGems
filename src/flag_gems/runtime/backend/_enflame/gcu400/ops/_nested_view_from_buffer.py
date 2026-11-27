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
"""GCU400 ``_nested_view_from_buffer``.

The generic implementation captures the raw native *CUDA* kernel with
``torch.library.get_kernel("aten::_nested_view_from_buffer", "CUDA")`` and calls
it with a CUDA-only key set, because the native kernel dereferences the metadata
on the host. No CUDA kernel exists in this build, so ``get_kernel`` returns
nothing and every call raises ``NotImplementedError``.

On GCU400 the operator is pure metadata: it wraps the buffer and one
``as_strided`` view per component into a strided-layout nested tensor without
moving an element. The native *CPU* kernel implements exactly that and is
device agnostic, so this specialization captures it once at import time and
invokes it with an explicit PrivateUse1 key set (same captured-kernel
``call_boxed`` pattern the generic implementation uses for CUDA).

Calling the public aten entry point instead would dispatch into torch_gcu's
``aten_cpu_fallback`` (registered for PrivateUse1), which rebuilds the nested
tensor from a copy of the buffer. That silently breaks the
``Tensor(a) self -> Tensor(a)`` aliasing contract and, for overlapping
component layouts, raises ``NestedTensor must be contiguous to get buffer``.
No aten registration happens here: the native kernel is only called directly.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_OP_NAME = "aten::_nested_view_from_buffer"

try:
    _NATIVE_KERNEL = torch.library.get_kernel(_OP_NAME, "CPU")
    _NATIVE_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.PrivateUse1)
except Exception:  # pragma: no cover - older torch builds without the CPU kernel
    logger.warning(
        "_nested_view_from_buffer: unable to acquire the native CPU kernel; "
        "falling back to the public aten entry point."
    )
    _NATIVE_KERNEL = None
    _NATIVE_KEYSET = None


def _nested_view_from_buffer(
    self: torch.Tensor,
    nested_size: torch.Tensor,
    nested_strides: torch.Tensor,
    offsets: torch.Tensor,
):
    """Wrap ``self`` into a strided-layout nested tensor using the given metadata.

    Args:
        self: flat buffer backing the components; offsets are element units
            into its storage.
        nested_size: ``(num_components, dim)`` int64 tensor of component sizes.
        nested_strides: ``(num_components, dim)`` int64 tensor of component
            strides.
        offsets: ``(num_components,)`` int64 tensor of component storage
            offsets.

    Returns:
        A nested tensor aliasing ``self``'s storage.
    """
    logger.debug("GEMS_ENFLAME _NESTED_VIEW_FROM_BUFFER")

    # The native kernel reads the metadata on the host, so device resident
    # metadata is staged to the CPU first.
    if not nested_size.is_cpu:
        nested_size = nested_size.cpu()
    if not nested_strides.is_cpu:
        nested_strides = nested_strides.cpu()
    if not offsets.is_cpu:
        offsets = offsets.cpu()

    if _NATIVE_KERNEL is None or self.device.type == "cpu":
        return torch.ops.aten._nested_view_from_buffer.default(
            self, nested_size, nested_strides, offsets
        )

    return _NATIVE_KERNEL.call_boxed(
        _NATIVE_KEYSET, self, nested_size, nested_strides, offsets
    )


__all__ = ["_nested_view_from_buffer"]
