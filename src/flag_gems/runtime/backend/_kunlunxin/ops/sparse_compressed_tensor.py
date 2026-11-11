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

"""Kunlunxin override of ``aten::sparse_compressed_tensor`` (CSR/CSC/BSR/BSC).

The generic implementation (``flag_gems.ops.sparse_compressed_tensor``) infers
the dense extents itself, through ``_plain_max_kernel``, when ``size`` is
omitted.  That kernel reduces with ``tl.atomic_max`` on int64, which the XPU
backend cannot compile: 64-bit atomics exceed the per-core stack budget even at
the minimum ``buffer_size_limit`` and fail with ``Failed to tune buffer size.``
(probed directly on this device; the int32-atomic variant of the same kernel
compiles and runs, so the limitation is specific to the 64-bit atomic).

XPyTorch's native ``sparse_*_tensor`` factories re-derive the same extents when
``size`` is omitted (verified on this platform for all four layouts, both
block and non-block, and multiple shapes) -- and that native inference is
exactly what the CPU reference in the accuracy tests uses, so shape parity is
guaranteed by construction.  The override therefore forwards ``size=None``
straight through to the factory instead of scanning the index buffer on device,
which is also the generic implementation's own fallback for index buffers that
cannot be reduced in Triton.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_SPARSE_CSR = torch.sparse_csr
_SPARSE_CSC = torch.sparse_csc
_SPARSE_BSR = torch.sparse_bsr
_SPARSE_BSC = torch.sparse_bsc

_LAYOUT_NAMES = {
    _SPARSE_CSR: "SparseCsr",
    _SPARSE_CSC: "SparseCsc",
    _SPARSE_BSR: "SparseBsr",
    _SPARSE_BSC: "SparseBsc",
}

# Layout-specialised constructors.  Each keeps the caller's three component
# buffers alive by reference (no copy) and records a differentiable identity on
# ``values``, so the returned tensor's value buffer aliases the input.
_FACTORY = {
    _SPARSE_CSR: torch.sparse_csr_tensor,
    _SPARSE_CSC: torch.sparse_csc_tensor,
    _SPARSE_BSR: torch.sparse_bsr_tensor,
    _SPARSE_BSC: torch.sparse_bsc_tensor,
}

_FLOAT32 = torch.float32


def sparse_compressed_tensor(
    compressed_indices,
    plain_indices,
    values,
    size=None,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
):
    """Build a sparse compressed tensor from its component buffers.

    Mirrors ``aten::sparse_compressed_tensor.comp_plain_value[_size]``:
    ``layout`` is mandatory and must be one of csr/csc/bsr/bsc, ``dtype`` is
    never inferred from ``values`` (the ATen factory defaults it to float32 and
    rejects a mismatch), and ``values`` is retained by reference.
    """
    logger.debug("GEMS_KUNLUNXIN SPARSE_COMPRESSED_TENSOR")
    try:
        factory = _FACTORY[layout]
    except (KeyError, TypeError):
        try:
            got = _LAYOUT_NAMES.get(layout, str(layout).replace("torch.", ""))
        except TypeError:
            got = str(layout)
        raise RuntimeError(
            "sparse_compressed_tensor expected sparse compressed tensor layout"
            " but got " + got
        ) from None

    # dtype is never inferred from values: the factory defaults to float32 and
    # requires values.dtype to match the requested dtype exactly.
    requested = _FLOAT32 if dtype is None else dtype
    if values.dtype is not requested:
        raise RuntimeError(
            "dtype of values (%s) must match dtype of sparse tensor (%s)"
            % (
                str(values.dtype).rsplit(".", 1)[-1],
                str(requested).rsplit(".", 1)[-1],
            )
        )

    if size is not None and type(size) is not list:
        size = list(size)

    # ``size is None`` must be *omitted* from the factory call: XPyTorch's
    # overload resolution rejects an explicit None for the size parameter (it
    # accepts either a tuple of ints or no size argument at all).  Omission
    # makes the factory re-derive the dense extents natively (see the module
    # docstring for why the generic Triton scan cannot be used on XPU).
    kwargs = {"dtype": requested, "device": device}
    if pin_memory is not False:
        kwargs["pin_memory"] = pin_memory
    if size is None:
        return factory(compressed_indices, plain_indices, values, **kwargs)
    return factory(compressed_indices, plain_indices, values, size, **kwargs)
