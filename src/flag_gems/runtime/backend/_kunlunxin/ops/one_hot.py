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

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Full-reduction tile for the max inference (the ``num_classes == -1`` path).
# The XPU int64 full reduction is slow, so a single program handles inputs up
# to this many elements directly; larger inputs fall back to a two-stage
# reduction. 8192 was empirically faster than 16384 for the ~10k-element cases.
_MAX_REDUCE_BLOCK = 8192

# num_classes values for which a dense one-hot kernel beats the zero-fill +
# scatter strategy. For these (small, power-of-two) class counts the output is
# written contiguously and NUM_CLASSES becomes a compile-time constant so the
# // and % lower to shifts, approaching the native one_hot_fwd vendor kernel.
_DENSE_NUM_CLASSES = (2, 4, 8, 16)


@triton.jit
def one_hot_set_one_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    # Clamp addresses so the masked-out tail lanes still resolve to a valid
    # element; the one-hot write is idempotent (a "1" that is already set).
    safe_offsets = tl.minimum(offsets, num_elements - 1)
    indices = tl.load(input_ptr + safe_offsets, mask=mask, other=0)
    out_offsets = safe_offsets.to(tl.int64) * num_classes + indices
    tl.store(output_ptr + out_offsets, 1, mask=mask)


@triton.jit
def one_hot_dense_kernel(
    input_ptr,
    output_ptr,
    total_out,
    BLOCK_SIZE: tl.constexpr,
    NUM_CLASSES: tl.constexpr,
):
    # Dense one-hot: each program writes a contiguous slab of the output. For a
    # small power-of-two NUM_CLASSES the integer div/mod lower to shifts and the
    # writes are fully coalesced, which is far faster than the strided scatter
    # (the scatter touches one full cache line per "1"). Note: the index must
    # not depend on a runtime scalar (e.g. clamping i against num_elements),
    # otherwise the address math is not strength-reduced and the kernel slows
    # down several-fold on XPU; the masked load below is sufficient to keep the
    # out-of-bounds tail lanes safe.
    pid = tl.program_id(0)
    out_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < total_out
    i = out_offsets // NUM_CLASSES
    c = out_offsets % NUM_CLASSES
    indices = tl.load(input_ptr + i, mask=mask, other=0)
    val = tl.where(indices == c, 1, 0)
    tl.store(output_ptr + out_offsets, val, mask=mask)


@triton.jit
def one_hot_max_single_kernel(
    input_ptr,
    out_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    safe_offs = tl.minimum(offs, num_elements - 1)
    vals = tl.load(input_ptr + safe_offs)
    tl.store(out_ptr, tl.max(vals, axis=0))


@triton.jit
def one_hot_max_kernel(
    input_ptr,
    partial_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    safe_offsets = tl.minimum(offsets, num_elements - 1)
    vals = tl.load(input_ptr + safe_offsets)
    tl.store(partial_ptr + pid, tl.max(vals, axis=0))


@triton.jit
def one_hot_max_final(
    partial_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    safe_offs = tl.minimum(offs, n - 1)
    vals = tl.load(partial_ptr + safe_offs)
    tl.store(out_ptr, tl.max(vals, axis=0))


def _scatter_block_size(num_elements: int):
    # Small tensors are launch-bound: a narrow tile keeps the grid wide enough
    # without oversubscribing. Large tensors are store-bound: a wide tile
    # amortizes the gather-store setup.
    if num_elements <= 16384:
        return 512, 4
    return 8192, 8


def _dense_block_size(total_out: int):
    # Large outputs are store-bound and want a wide tile; small outputs are
    # launch-bound and prefer a moderate grid.
    if total_out >= 1024 * 1024:
        return 32768, 8
    return 8192, 8


def _infer_num_classes(flat_input: torch.Tensor) -> int:
    # The XPU int64 full reduction is slow, so avoid the flag_gems ``max``
    # override and reduce with a lean Triton kernel instead. A single program
    # handles small inputs directly; larger inputs use a two-stage reduction.
    num_elements = flat_input.numel()
    out = torch.empty((), device=flat_input.device, dtype=torch.int64)
    if num_elements <= _MAX_REDUCE_BLOCK:
        one_hot_max_single_kernel[(1,)](
            flat_input,
            out,
            num_elements,
            BLOCK_SIZE=triton.next_power_of_2(num_elements),
        )
    else:
        nblocks = triton.cdiv(num_elements, _MAX_REDUCE_BLOCK)
        partial = torch.empty(nblocks, device=flat_input.device, dtype=torch.int64)
        one_hot_max_kernel[(nblocks,)](
            flat_input,
            partial,
            num_elements,
            BLOCK_SIZE=_MAX_REDUCE_BLOCK,
        )
        one_hot_max_final[(1,)](
            partial,
            out,
            nblocks,
            BLOCK_SIZE=triton.next_power_of_2(nblocks),
        )
    return int(out.item()) + 1


def one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN ONE_HOT")
    if tensor.dtype != torch.int64:
        raise RuntimeError(
            "one_hot is only applicable to index tensor of type LongTensor."
        )

    if tensor.numel() == 0:
        if num_classes <= 0:
            raise RuntimeError(
                "Can not infer total number of classes from empty tensor."
            )
        return torch.empty(
            (*tensor.shape, num_classes), device=tensor.device, dtype=torch.int64
        )

    flat_input = tensor.contiguous().view(-1)
    num_elements = flat_input.numel()

    if num_classes == -1:
        num_classes = _infer_num_classes(flat_input)

    if num_classes < 1:
        raise RuntimeError("num_classes should be positive")

    if num_classes in _DENSE_NUM_CLASSES:
        out = torch.empty(
            (*tensor.shape, num_classes), device=tensor.device, dtype=torch.int64
        )
        total_out = num_elements * num_classes
        BLOCK_SIZE, num_warps = _dense_block_size(total_out)
        grid = (triton.cdiv(total_out, BLOCK_SIZE),)
        one_hot_dense_kernel[grid](
            flat_input,
            out,
            total_out,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_CLASSES=num_classes,
            num_warps=num_warps,
        )
    else:
        out = torch.zeros(
            (*tensor.shape, num_classes), device=tensor.device, dtype=torch.int64
        )
        BLOCK_SIZE, num_warps = _scatter_block_size(num_elements)
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        one_hot_set_one_kernel[grid](
            flat_input,
            out,
            num_elements,
            num_classes,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    return out
