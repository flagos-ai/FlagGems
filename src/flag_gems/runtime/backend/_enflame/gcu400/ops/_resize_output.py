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

# GCU400 override of the KernelGen host (2026-11-17), replacing the copied
# generic implementation:
# 1) P19 grid hard limit: the old host fell through to the flat copy in
#    flag_gems.ops.copy, which launches one CTA per 4096-word tile - the core
#    benchmark shape (2**30 fp32 elements) asks for grid.x = 262144 > 65535
#    and the whole bench failed with OutOfResources (commit 96c56c37d had
#    already re-pointed the import at the enflame copy as a stopgap).  The
#    copy now runs a persistent grid-stride kernel owned by this module,
#    capped at <= 96 CTAs for any size (mse_loss_backward / fw_primal_copy
#    recipe), so grid.x can never reach the hardware limit.
# 2) Same-numel fast path: aten::_resize_output_ keeps the caller's storage
#    when the element count already matches (verified on the Meta backend:
#    torch.ops.aten._resize_output_(x, [2, 4], x.device) returns x itself),
#    so this is a metadata-only reshape with no kernel at all.  A bare device
#    name ('gcu') is normalized the way factory ops do, so a same-numel
#    resize requested for 'gcu' no longer pays a full storage copy.
# 3) The copy is dtype-agnostic: int32 words (uint8 for 1-byte or misaligned
#    layouts) move fp8/int64/fp64/complex bit patterns without any Triton
#    dtype mapping.  Non-contiguous or cross-device operands keep the generic
#    parity path through the enflame FlagGems copy_.
#
# 2026-11-19 perf round (P3): every core benchmark case is a same-numel
# resize, i.e. pure host-side metadata work measured at ~2us.  The host path
# was re-ordered to shave the Python overhead that dominates it:
#   - an identical-shape request is the plain no-op of aten resize_ and
#     returns the input tensor itself (no aten call at all);
#   - a 1-D target of a contiguous tensor takes a bare as_strided alias view
#     (0.97us) instead of reshape (1.4-1.8us); reshape stays as the fallback
#     because Tensor.view dispatches into a slow FlagGems python kernel
#     under use_gems (measured 4.7us) while as_strided/reshape do not;
#   - the device match compares the bare type string directly instead of
#     building a torch.device per call, and the entry-point logger.debug
#     calls (0.25us of a ~2us op) were dropped.
import logging
from typing import List

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Persistent launch: 48 CTAs = 24 SIPs x 2 (gcu400 house recipe).  The deep
# band doubles the CTA count and widens the tile for streaming sizes
# (>= 256 MiB of int32 traffic), matching the measured bands in gcu400
# copy.py.  grid.x is capped for any input size, so the P19 limit is
# unreachable by construction.
_GRID_CAP = 48
_DEEP_GRID_CAP = 96
_DEEP_WORDS = 64 * 1024 * 1024
_BLOCK = 8192
_DEEP_BLOCK = 32768
_INT32_MAX = (1 << 31) - 1


@triton.jit(do_not_specialize=["n_words"])
def _resize_output_copy_kernel(
    src_ptr,
    dst_ptr,
    n_words,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    IDX64: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_tiles = tl.cdiv(n_words, BLOCK)
    for tile_id in tl.range(pid, num_tiles, num_pids):
        if IDX64:
            offsets = tile_id.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        else:
            offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            value = tl.load(src_ptr + offsets)
            tl.store(dst_ptr + offsets, value)
        else:
            mask = offsets < n_words
            value = tl.load(src_ptr + offsets, mask=mask)
            tl.store(dst_ptr + offsets, value, mask=mask)


def _device_matches(inp_device: torch.device, device) -> bool:
    if isinstance(device, torch.device):
        if device.index is None:
            # torch.device('gcu') (no index) is a wildcard for the tensor's
            # device type, exactly like torch.empty(..., device='gcu')
            # allocating on the current device of that type.  cpu tensors
            # compare equal this way too (their device carries no index).
            return device.type == inp_device.type
        return device == inp_device
    if isinstance(device, str):
        # Bare device names ('gcu') are wildcards; indexed names ('gcu:0')
        # must match the tensor device exactly.
        if ":" in device:
            return torch.device(device) == inp_device
        return device == inp_device.type
    return False


def _copy_prefix_fallback(out: torch.Tensor, inp: torch.Tensor, n: int) -> None:
    from flag_gems.runtime.backend._enflame.gcu400.ops.copy import copy_

    copy_(out.reshape(-1)[:n], inp.reshape(-1)[:n])


def _copy_prefix(out: torch.Tensor, inp: torch.Tensor, n: int) -> None:
    """Copy the first ``n`` elements of ``inp`` into ``out`` (same dtype)."""
    if not (inp.is_contiguous() and inp.device == out.device):
        _copy_prefix_fallback(out, inp, n)
        return
    esize = inp.element_size()
    if (n * esize) % 4 == 0 and (inp.storage_offset() * esize) % 4 == 0:
        word_dtype = torch.int32
        n_words = n * esize // 4
    elif esize == 1:
        word_dtype = torch.uint8
        n_words = n
    else:
        # e.g. a 2-byte dtype with an odd element count or a misaligned
        # storage offset: keep the generic parity path.
        _copy_prefix_fallback(out, inp, n)
        return
    try:
        src_view = inp.reshape(-1).view(word_dtype)
        dst_view = out.reshape(-1).view(word_dtype)
    except (RuntimeError, TypeError, NotImplementedError):
        _copy_prefix_fallback(out, inp, n)
        return
    if n_words >= _DEEP_WORDS:
        block, cap, num_warps = _DEEP_BLOCK, _DEEP_GRID_CAP, 8
    else:
        block, cap, num_warps = _BLOCK, _GRID_CAP, 4
    num_tiles = (n_words + block - 1) // block
    grid = (cap if num_tiles > cap else num_tiles,)
    with torch_device_fn.device(inp.device):
        _resize_output_copy_kernel[grid](
            src_view,
            dst_view,
            n_words,
            BLOCK=block,
            EVEN=n_words % block == 0,
            IDX64=n_words > _INT32_MAX - block,
            num_warps=num_warps,
        )


def _resize_output(inp: torch.Tensor, size: List[int], device: torch.device):
    """Resize the input tensor to the specified size (aten::_resize_output).

    The output keeps the overlapping prefix of ``inp``; regions grown beyond
    the old element count stay uninitialized, matching aten resize_.
    """
    if (
        inp.dim() == len(size)
        and tuple(size) == inp.shape
        and _device_matches(inp.device, device)
    ):
        # Identical shape on the matching device: the plain no-op of aten
        # resize_, which returns ``self`` untouched (no kernel, no copy,
        # not even a view).
        return inp

    old_numel = inp.numel()
    if len(size) == 1:
        new_numel = size[0]
    else:
        new_numel = 1
        for dim in size:
            new_numel *= dim

    if new_numel == old_numel and _device_matches(inp.device, device):
        # aten keeps the storage when the element count already matches:
        # metadata-only reshape, no kernel, no copy.  A 1-D target of a
        # contiguous tensor is a bare alias view; as_strided is used
        # instead of Tensor.view because view dispatches into a slow
        # FlagGems python kernel under use_gems, while as_strided does
        # not.  reshape stays as the general (non-contiguous) fallback.
        if len(size) == 1 and inp.is_contiguous():
            return inp.as_strided((size[0],), (1,))
        return inp.reshape(size)

    out = torch.empty(size, dtype=inp.dtype, device=device)
    copy_numel = old_numel if old_numel < new_numel else new_numel
    if copy_numel > 0:
        _copy_prefix(out, inp, copy_numel)
    return out
