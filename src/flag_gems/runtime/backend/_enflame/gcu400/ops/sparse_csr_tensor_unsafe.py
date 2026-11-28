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

"""GCU400 specialization of ``aten::_sparse_csr_tensor_unsafe``.

The operator itself is pure metadata assembly: the unsafe CSR constructor
wraps already-materialised ``(crow_indices, col_indices, values)`` components
without validating the compressed indices, so there is no data-parallel
arithmetic to offload.  The only device-side work a call can request is the
dtype conversion of the payload, which the Triton ``_cast_kernel`` below
performs before the native assembly.

Compared with the portable module this copy

* drives the conversion with a persistent grid capped at two programs per SPC
  instead of ``ceil(nnz / 1024)`` one-shot programs, so the launch geometry is
  independent of ``nnz`` and the per-program strided loop keeps every SPC busy
  on large payloads, and
* uses an exact-tile kernel (no ``offset < nnz`` predicate) whenever ``nnz`` is
  a multiple of the tile width, and skips the launch entirely for empty
  payloads.

Indices are passed through untouched: constructing the compressed tensor needs
the native assembly call, which only reads shape/stride/pointer metadata and is
therefore a metadata operation, not a compute fallback.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.sparse_csr_tensor_unsafe import (
    _DEVICE_KEYSET,
    _NATIVE_COMPRESSED_UNSAFE,
    _flag_gems_compressed_unsafe,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24
GRID_CAP = NUM_SIPS * 2
# Tile width in elements, chosen per payload width so one tile stays inside the
# comfortable register budget of a single SPC (see _launch_cast).
_SMALL_MAX = 65536


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _cast_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    """Elementwise cast of the payload values into the requested dtype."""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (n_elements + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        offs = block_id * BLOCK + arange
        mask = offs < n_elements
        x = tl.load(src_ptr + offs, mask=mask)
        tl.store(dst_ptr + offs, x.to(dst_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _cast_kernel_exact(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    """Exact-tile cast used when ``n_elements`` is a multiple of ``BLOCK``."""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = n_elements // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        offs = block_id * BLOCK + arange
        x = tl.load(src_ptr + offs)
        tl.store(dst_ptr + offs, x.to(dst_ptr.dtype.element_ty))


def _launch_cast(src, dst, n_elements):
    if n_elements <= 0:
        return
    if n_elements <= _SMALL_MAX:
        # Spread a small payload over several SPCs instead of handing the whole
        # tile to a single program.
        tile = triton.next_power_of_2(triton.cdiv(n_elements, NUM_SIPS))
        block = max(1024, min(tile, 8192))
        grid = min(triton.cdiv(n_elements, block), GRID_CAP)
        with torch_device_fn.device(src.device):
            _cast_kernel[(grid,)](src, dst, n_elements, BLOCK=block, num_warps=2)
        return
    # 4-byte payloads move twice the bytes per element, so they use half the
    # tile width of 2-byte payloads for the same tile footprint.
    if src.element_size() <= 2:
        block, num_warps = 65536, 4
    else:
        block, num_warps = 32768, 2
    grid = min(triton.cdiv(n_elements, block), GRID_CAP)
    kernel = _cast_kernel_exact if n_elements % block == 0 else _cast_kernel
    with torch_device_fn.device(src.device):
        kernel[(grid,)](src, dst, n_elements, BLOCK=block, num_warps=num_warps)


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
    logger.debug("GEMS_ENFLAME SPARSE CSR TENSOR UNSAFE")

    out_values = values
    if dtype is not None and out_values.dtype != dtype:
        out_values = torch.empty(
            out_values.shape, dtype=dtype, device=out_values.device
        )
        _launch_cast(values, out_values, values.numel())

    if device is not None and out_values.device != torch.device(device):
        out_values = out_values.to(device)

    # Assemble through the sibling unsafe compressed constructor rather than
    # ``_sparse_csr_tensor_unsafe`` itself: the FlagGems dispatch table
    # overrides this operator, so re-entering it here would recurse into this
    # wrapper.  The unsafe constructor infers neither the target device nor the
    # dtype from the values, so both are forwarded explicitly.
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
