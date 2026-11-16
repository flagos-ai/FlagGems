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

"""Kunlunxin (TritonXPU) specialization of ``aten::sparse_compressed_tensor``.

Why this file exists
--------------------
The explicit-``size`` path of the generic implementation is pure host dispatch
and works on XPU3.  The ``comp_plain_value`` (no-size) overload infers the dense
extents with ``_plain_max_kernel``: a grid-stride loop that does

    ``tl.load(ptr + offs, mask=mask, other=-1)``

On this backend masked loads are not honoured and ``other=`` is silently
ignored (documented TritonXPU hazard), and that loop form fails to compile at
all (``RuntimeError: Failed to tune buffer size`` inside
``triton/compiler/compiler.py``).  On HEAD this makes all 24
``test_..._infer_size[...]`` cases fail deterministically.

Also note a *plain* unmasked load is not a fix: reading past ``n`` returns
garbage (measured max ``7.23e16`` for a 20-element buffer), so the tail must be
clamped.  The fixed kernel below:
  * uses a flat ``grid = ceil(n / BLOCK)`` (no grid-stride loop: the loop form
    is what fails to lower),
  * loads with a ``tl.where``-clamped index (never a load mask),
  * folds out-of-range lanes with ``tl.where(live, x, -1)`` (-1 is the identity
    for a max over non-negative sparse indices).

The reduction width also matters: an int64 ``tl.max`` over a 1024- or 512-lane
tile does not lower here (1024 hangs the compiler, 512 raises "Failed to tune
buffer size").  The kernel therefore uses a 256-lane tile.

Everything else (layout/dtype contract, factory call shape, host fast path) is
the generic behaviour, reproduced here so the override stays self-contained and
does not mutate the generic module.
"""

import importlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

# Generic module: reused only for its immutable layout constants.  The generic
# host logic is reproduced below because its `_infer_size` is the broken part.
_generic = importlib.import_module("flag_gems.ops.sparse_compressed_tensor")

_FACTORY = _generic._FACTORY
_LAYOUT_NAMES = _generic._LAYOUT_NAMES
_CROW_LAYOUTS = _generic._CROW_LAYOUTS
_BLOCK_LAYOUTS = _generic._BLOCK_LAYOUTS
_FLOAT32 = _generic._FLOAT32

# Tile for the index-max kernel.  The generic 1024-wide int64 reduction does
# not lower here: BLOCK 512 fails with "Failed to tune buffer size" and 1024
# hangs the compiler; 256 (measured) is the largest safe width.
_BLOCK_ELEMS = 256
_MIN_TILE = 64


@libentry()
@triton.jit
def _plain_max_kernel(
    ptr: tl.tensor,
    out_ptr: tl.tensor,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """Max over a non-negative integer index buffer, folded onto ``out_ptr[0]``.

    One program per block (no grid-stride loop), tail lanes clamped to index 0
    on the load and folded to ``-1`` on the value -- no masked load, no
    ``other=``.
    """
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    live = offsets < n
    safe = tl.where(live, offsets, 0)
    x = tl.load(ptr + safe).to(tl.int64)
    x = tl.where(live, x, -1)
    tl.atomic_max(out_ptr, tl.max(x, axis=0))


def _num_warps(block: int) -> int:
    if block >= 2048:
        return 4
    return 1


def _infer_size(compressed_indices, plain_indices, values, layout):
    """Derive ``size`` for the ``comp_plain_value`` (no-size) overload."""
    if layout in _BLOCK_LAYOUTS:
        if values.dim() < 3:
            return None
        bs0 = int(values.shape[-2])
        bs1 = int(values.shape[-1])
        batch = tuple(values.shape[:-3])
    else:
        bs0 = bs1 = 1
        batch = tuple(values.shape[:-1])

    comp_dim = int(compressed_indices.shape[-1]) - 1
    if comp_dim < 0:
        return None

    n = plain_indices.numel()
    if n == 0:
        plain_max = -1
    else:
        if not plain_indices.is_contiguous():
            return None
        flat = plain_indices.reshape(-1)
        acc = torch.full((1,), -1, dtype=torch.int64, device=flat.device)
        block = max(_MIN_TILE, triton.next_power_of_2(int(_BLOCK_ELEMS)))
        grid = (triton.cdiv(n, block),)
        _plain_max_kernel[grid](
            flat, acc, n, BLOCK_SIZE=block, num_warps=_num_warps(block)
        )
        plain_max = int(acc.item())
        del acc

    if layout in _CROW_LAYOUTS:
        rows = comp_dim * bs0
        cols = (plain_max + 1) * bs1
    else:
        rows = (plain_max + 1) * bs0
        cols = comp_dim * bs1
    return list(batch) + [rows, cols]


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

    Mirrors ``aten::sparse_compressed_tensor.comp_plain_value[_size]``: the
    layout must be csr/csc/bsr/bsc, ``dtype`` is never inferred from ``values``
    and must match exactly, and ``values`` is retained by reference.
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

    requested = _FLOAT32 if dtype is None else dtype
    if values.dtype is not requested:
        raise RuntimeError(
            "dtype of values (%s) must match dtype of sparse tensor (%s)"
            % (
                str(values.dtype).rsplit(".", 1)[-1],
                str(requested).rsplit(".", 1)[-1],
            )
        )

    if type(size) is not list:
        if size is None:
            size = _infer_size(compressed_indices, plain_indices, values, layout)
        elif type(size) is not tuple:
            size = list(size)

    if pin_memory is False:
        return factory(
            compressed_indices,
            plain_indices,
            values,
            size,
            dtype=requested,
            device=device,
        )
    return factory(
        compressed_indices,
        plain_indices,
        values,
        size,
        dtype=requested,
        device=device,
        pin_memory=pin_memory,
    )
