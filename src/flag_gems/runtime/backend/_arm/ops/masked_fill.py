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

from flag_gems.utils import broadcastable_to

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["value", "n_elements"])
def _masked_fill_kernel(
    inp_ptr,
    mask_ptr,
    value,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(mask_ptr + offsets, mask=mask, other=0).to(tl.int1)
    y = tl.where(m, value, x)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit(do_not_specialize=["value", "n_elements"])
def _masked_fill_single_program_kernel(
    inp_ptr,
    mask_ptr,
    value,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < n_elements
        x = tl.load(inp_ptr + idx, mask=mask, other=0.0)
        m = tl.load(mask_ptr + idx, mask=mask, other=0).to(tl.int1)
        y = tl.where(m, value, x)
        tl.store(out_ptr + idx, y, mask=mask)


@triton.jit(do_not_specialize=["value", "n_elements"])
def _masked_fill_inplace_kernel(
    inp_ptr,
    mask_ptr,
    value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(mask_ptr + offsets, mask=mask, other=0).to(tl.int1)
    y = tl.where(m, value, x)
    tl.store(inp_ptr + offsets, y, mask=mask)


@triton.jit(do_not_specialize=["value", "n_elements"])
def _masked_fill_inplace_single_program_kernel(
    inp_ptr,
    mask_ptr,
    value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < n_elements
        x = tl.load(inp_ptr + idx, mask=mask, other=0.0)
        m = tl.load(mask_ptr + idx, mask=mask, other=0).to(tl.int1)
        y = tl.where(m, value, x)
        tl.store(inp_ptr + idx, y, mask=mask)


def _select_block_size(n_elements):
    if n_elements <= 32:
        return 32
    if n_elements <= 1024:
        return 32
    if n_elements <= 8192:
        return 64
    return 128


def _normalize_scalar_value(value):
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill only supports scalar/0-d tensor value"
    if torch.is_tensor(value):
        return value.item()
    return value


def _prepare_mask(mask, inp_shape):
    if mask.dtype == torch.bool and tuple(mask.shape) == tuple(inp_shape):
        return mask if mask.is_contiguous() else mask.contiguous()
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    if tuple(mask.shape) == tuple(inp_shape):
        return mask if mask.is_contiguous() else mask.contiguous()
    return mask.expand(inp_shape).contiguous()


def _launch_masked_fill(inp, expand_mask, value, out):
    n_elements = inp.numel()
    if n_elements == 0:
        return
    if 1 < n_elements <= 8192:
        single_block = 32 if n_elements <= 4096 else 64
        _masked_fill_single_program_kernel[(1,)](
            inp,
            expand_mask,
            value,
            out,
            n_elements,
            BLOCK_SIZE=single_block,
            num_warps=1,
            num_stages=1,
        )
        return

    block_size = _select_block_size(n_elements)
    grid = (triton.cdiv(n_elements, block_size),)
    _masked_fill_kernel[grid](
        inp,
        expand_mask,
        value,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


def _launch_masked_fill_inplace(inp, expand_mask, value):
    n_elements = inp.numel()
    if n_elements == 0:
        return
    if 1 < n_elements <= 8192:
        single_block = 32 if n_elements <= 4096 else 64
        _masked_fill_inplace_single_program_kernel[(1,)](
            inp,
            expand_mask,
            value,
            n_elements,
            BLOCK_SIZE=single_block,
            num_warps=1,
            num_stages=1,
        )
        return

    block_size = _select_block_size(n_elements)
    grid = (triton.cdiv(n_elements, block_size),)
    _masked_fill_inplace_kernel[grid](
        inp,
        expand_mask,
        value,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


def masked_fill(inp, mask, value):
    logger.debug("GEMS_ARM MASKED_FILL")
    value = _normalize_scalar_value(value)
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "mask shape must be broadcastable to input shape"

    if inp.ndim == 0:
        return (
            torch.tensor(value, dtype=inp.dtype, device=inp.device)
            if mask.item()
            else inp.clone()
        )

    if mask.ndim == 0:
        if bool(mask.item()):
            return torch.full_like(inp, value)
        return inp.clone()

    inp_contig = inp.contiguous() if not inp.is_contiguous() else inp
    expand_mask = _prepare_mask(mask, inp_contig.shape)
    out = torch.empty_like(inp_contig, dtype=inp_contig.dtype, device=inp_contig.device)
    _launch_masked_fill(inp_contig, expand_mask, value, out)
    return out


def masked_fill_(inp, mask, value):
    logger.debug("GEMS_ARM MASKED_FILL_")
    value = _normalize_scalar_value(value)
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "mask shape must be broadcastable to input shape"

    if inp.ndim == 0:
        if mask.item():
            inp[()] = value
        return inp

    if mask.ndim == 0:
        if bool(mask.item()):
            inp.fill_(value)
        return inp

    inp_contig = inp.contiguous() if not inp.is_contiguous() else inp
    expand_mask = _prepare_mask(mask, inp_contig.shape)
    _launch_masked_fill_inplace(inp_contig, expand_mask, value)
    if inp_contig is not inp:
        inp.copy_(inp_contig)
    return inp
