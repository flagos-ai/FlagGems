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

"""GCU400-specific ``new_zeros`` host-side dispatch.

The common implementation only treats ``device.type == "cuda"`` as directly writable by a Triton kernel,
so on GCU it takes a detour of ``scratch = torch.empty(..., device="cuda")`` + ``copy_``,
paying an extra allocation, one zeroing pass and one copy. Keep the common implementation's already
compile-cached ``_zero_bytes`` / ``_zero_strided`` Triton kernels and only fix the host
device decision, launching directly on the target buffer; operator semantics and signature are unchanged.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.new_zeros import _zero_bytes, _zero_strided

logger = logging.getLogger(__name__)


@triton.jit
def _zero_words(p, N, BLOCK: tl.constexpr):
    """Zero at 32-bit word granularity: 4x fewer write instructions than uint8 stores."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(p + offs, tl.zeros([BLOCK], p.dtype.element_ty), mask=offs < N)


@triton.jit
def _zero_words_nomask(p, N, BLOCK: tl.constexpr):
    """Used when numel is an exact multiple of BLOCK: drops the predicate, tighter store sequence."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(p + offs, tl.zeros([BLOCK], p.dtype.element_ty))


# --- TUNING PARAMS START ---
DIRECT_DEVICE = 1
# Bucketing rationale: measurements show that for the same shape it is the "number of programs", not just the byte
# count, that determines throughput —
# medium buffers (<=8 MiB) are spread over about 12 programs (after rounding up to a power of 2, landing at 8~12,
# fp16 2 MiB / fp32 4 MiB are both optimal in this bucket), while larger buffers need about 64 programs to be saturated.
TARGET_PROGRAMS_MEDIUM = 12
TARGET_PROGRAMS_LARGE = 64
LARGE_BUFFER_BYTES = 8 * 1024 * 1024
MIN_PROGRAM_BYTES = 16 * 1024
MAX_PROGRAM_BYTES = 1024 * 1024
STORE_BYTES = 8
NUM_WARPS = 2
BLOCK_ELEMS_STRIDED = 1024
STRIDED_WARPS = 4
SKIP_PINNED_CHECK = 1
# --- TUNING PARAMS END ---


def _as_size(size):
    """Normalize ``size`` to an int tuple, semantics identical to the common implementation."""
    if isinstance(size, torch.Size):
        return tuple(int(e) for e in size)
    if isinstance(size, (list, tuple)):
        out = []
        for e in size:
            if isinstance(e, bool):
                out.append(int(e))
            else:
                out.append(e)
        return tuple(out)
    if isinstance(size, torch.Tensor):
        return tuple(int(e) for e in size.tolist())
    raise TypeError(
        "new_zeros(): argument 'size' must be a sequence of integers, "
        f"not {type(size).__name__}"
    )


def _empty(size_t, dtype, layout, device, pin_memory):
    kwargs = {"dtype": dtype, "device": device}
    if layout is not None:
        kwargs["layout"] = layout
    if pin_memory:
        kwargs["pin_memory"] = True
    return torch.empty(size_t, **kwargs)


def _program_bytes(num_bytes):
    """Derives the per-program byte target from the total byte count.

    The rule puts the program count in the measured best tier (8 for medium buffers, 64 for
    large ones) clamped to [MIN_PROGRAM_BYTES, MAX_PROGRAM_BYTES]; tiny buffers therefore land
    on a single program, and larger ones do not slow down from over-long blocks.
    """
    target_programs = (
        TARGET_PROGRAMS_MEDIUM
        if num_bytes <= LARGE_BUFFER_BYTES
        else TARGET_PROGRAMS_LARGE
    )
    program_bytes = num_bytes // target_programs
    return max(MIN_PROGRAM_BYTES, min(MAX_PROGRAM_BYTES, program_bytes))


def _power_of_two_at_most(value):
    """Rounds down to a power of two: ``tl.arange`` block lengths must be powers of two."""
    value = max(1, int(value))
    return 1 << (value.bit_length() - 1)


def _power_of_two_at_least(value):
    """Rounds up to a power of two (block length must be a power of two; rounding up wins full-tile writes)."""
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _fill_contiguous(buf):
    """Contiguous buffer: emit the zeroing store at 64/32/8-bit width, chosen by byte alignment.

    Wider stores move the same bytes with fewer write instructions and longer contiguous runs;
    the int64 view requires the total bytes to be a multiple of 8 (always true for new buffers).
    """
    flat = buf.reshape(-1)
    byte_view = flat.view(torch.uint8)
    num_bytes = byte_view.numel()
    if num_bytes == 0:
        return

    program_bytes = _program_bytes(num_bytes)
    if STORE_BYTES >= 8 and num_bytes % 8 == 0:
        store_view = byte_view.view(torch.int64)
        elem_bytes = 8
    elif num_bytes % 4 == 0:
        store_view = byte_view.view(torch.int32)
        elem_bytes = 4
    else:
        store_view = byte_view
        elem_bytes = 1

    num_elems = store_view.numel()
    block = _power_of_two_at_least(max(1, program_bytes // elem_bytes))
    if num_elems < block:
        # A single program can cover the whole buffer: narrow the block length to the actual number of elements to avoid
        # idle threads.
        block = _power_of_two_at_least(num_elems)
    num_warps = NUM_WARPS
    grid = (triton.cdiv(num_elems, block),)
    if elem_bytes == 1:
        _zero_bytes[grid](store_view, num_elems, block, num_warps=num_warps)
    elif num_elems % block == 0:
        _zero_words_nomask[grid](store_view, num_elems, block, num_warps=num_warps)
    else:
        _zero_words[grid](store_view, num_elems, block, num_warps=num_warps)


def _fill_strided(buf):
    view = buf
    if buf.stride(-1) == 1 and buf.dtype not in (torch.uint8, torch.int8):
        try:
            view = buf.view(torch.uint8)
        except RuntimeError:
            view = buf
    shape = tuple(int(x) for x in view.shape)
    strides = tuple(int(x) for x in view.stride())
    num_elems = view.numel()
    grid = (triton.cdiv(num_elems, BLOCK_ELEMS_STRIDED),)
    _zero_strided[grid](
        view,
        num_elems,
        len(shape),
        shape,
        strides,
        BLOCK_ELEMS_STRIDED,
        num_warps=STRIDED_WARPS,
    )


def _fill_device(buf):
    if buf.is_contiguous():
        _fill_contiguous(buf)
    else:
        _fill_strided(buf)


def _fill_any(t):
    """Zeroes a dense buffer (GCU device / UVA pinned / CPU fallback)."""
    if t.numel() == 0 or t.device.type == "meta":
        return
    if (DIRECT_DEVICE and t.device.type == "gcu") or t.device.type == "cuda":
        # GCU device memory can be addressed directly by the Triton kernel, with no intermediate scratch needed.
        _fill_device(t)
    elif not SKIP_PINNED_CHECK and t.is_pinned():
        _fill_device(t)
    else:
        # Pageable host storage cannot be written directly by a device kernel; zeros are still produced by a device
        # kernel,
        # and here the already-computed zero buffer is simply moved back to the host result.
        scratch = torch.empty(tuple(t.shape), dtype=t.dtype, device="cuda")
        _fill_device(scratch)
        t.copy_(scratch)


def _zero_buffer(buf):
    if buf.layout == torch.sparse_csr or buf.layout == torch.sparse_csc:
        aux = (
            buf.crow_indices() if buf.layout == torch.sparse_csr else buf.ccol_indices()
        )
        _fill_any(aux)
        return
    if buf.layout != torch.strided:
        return
    _fill_any(buf)


def new_zeros(
    self, size, *, dtype=None, layout=None, device=None, pin_memory=None, out=None
):
    logger.debug("GEMS_GCU400 NEW_ZEROS")
    size_t = _as_size(size)

    if dtype is None:
        dtype = self.dtype
    if layout is None:
        layout = self.layout
    if device is None:
        device = self.device
    if pin_memory is None:
        try:
            pin_memory = bool(self.is_pinned())
        except Exception:  # noqa: BLE001 - sparse/meta sources have no pin state
            pin_memory = False

    if out is None:
        buf = _empty(size_t, dtype, layout, device, pin_memory)
    else:
        buf = out
        if tuple(buf.shape) != size_t:
            buf = buf.resize_(size_t)

    _zero_buffer(buf)
    return buf


def new_zeros_out(
    self, size, *, dtype=None, layout=None, device=None, pin_memory=None, out=None
):
    """``aten::new_zeros.out``: zero the caller's buffer and return its alias."""
    logger.debug("GEMS_GCU400 NEW_ZEROS_OUT")
    return new_zeros(
        self,
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        out=out,
    )
