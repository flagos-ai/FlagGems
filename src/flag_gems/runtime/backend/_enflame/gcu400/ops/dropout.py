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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

UNROLL = 4


@libentry()
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def dropout_forward_kernel(
    X,
    Y,
    dropout_mask,
    N,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0_base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid = tl.program_id(0)
    i4 = pid * BLOCK + tl.arange(0, BLOCK)
    c0 = c0_base + i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)

    mask0 = r0 > p
    mask1 = r1 > p
    mask2 = r2 > p
    mask3 = r3 > p

    off_0 = pid * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    x0 = tl.load(X + off_0, mask=off_0 < N, other=0.0).to(X.dtype.element_ty)
    x1 = tl.load(X + off_1, mask=off_1 < N, other=0.0).to(X.dtype.element_ty)
    x2 = tl.load(X + off_2, mask=off_2 < N, other=0.0).to(X.dtype.element_ty)
    x3 = tl.load(X + off_3, mask=off_3 < N, other=0.0).to(X.dtype.element_ty)

    scale = 1.0 / (1.0 - p)
    y0 = x0 * scale * mask0
    y1 = x1 * scale * mask1
    y2 = x2 * scale * mask2
    y3 = x3 * scale * mask3

    tl.store(Y + off_0, y0, mask=off_0 < N)
    tl.store(Y + off_1, y1, mask=off_1 < N)
    tl.store(Y + off_2, y2, mask=off_2 < N)
    tl.store(Y + off_3, y3, mask=off_3 < N)

    tl.store(dropout_mask + off_0, mask0.to(tl.uint8), mask=off_0 < N)
    tl.store(dropout_mask + off_1, mask1.to(tl.uint8), mask=off_1 < N)
    tl.store(dropout_mask + off_2, mask2.to(tl.uint8), mask=off_2 < N)
    tl.store(dropout_mask + off_3, mask3.to(tl.uint8), mask=off_3 < N)


@libentry()
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def dropout_persistent_kernel(
    X,
    Y,
    dropout_mask,
    N,
    p,
    philox_seed,
    philox_offset,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0_base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    for block_id in tl.range(pid, NUM_BLOCKS, num_pids):
        i4 = block_id * BLOCK + arange
        c0 = c0_base + i4
        _O = c0 * 0
        r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
        r0 = uint_to_uniform_float(r0)
        r1 = uint_to_uniform_float(r1)
        r2 = uint_to_uniform_float(r2)
        r3 = uint_to_uniform_float(r3)

        mask0 = r0 > p
        mask1 = r1 > p
        mask2 = r2 > p
        mask3 = r3 > p

        off_0 = block_id * BLOCK * 4 + arange
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK

        x0 = tl.load(X + off_0, mask=off_0 < N, other=0.0).to(X.dtype.element_ty)
        x1 = tl.load(X + off_1, mask=off_1 < N, other=0.0).to(X.dtype.element_ty)
        x2 = tl.load(X + off_2, mask=off_2 < N, other=0.0).to(X.dtype.element_ty)
        x3 = tl.load(X + off_3, mask=off_3 < N, other=0.0).to(X.dtype.element_ty)

        scale = 1.0 / (1.0 - p)
        y0 = x0 * scale * mask0
        y1 = x1 * scale * mask1
        y2 = x2 * scale * mask2
        y3 = x3 * scale * mask3

        tl.store(Y + off_0, y0, mask=off_0 < N)
        tl.store(Y + off_1, y1, mask=off_1 < N)
        tl.store(Y + off_2, y2, mask=off_2 < N)
        tl.store(Y + off_3, y3, mask=off_3 < N)

        tl.store(dropout_mask + off_0, mask0.to(tl.uint8), mask=off_0 < N)
        tl.store(dropout_mask + off_1, mask1.to(tl.uint8), mask=off_1 < N)
        tl.store(dropout_mask + off_2, mask2.to(tl.uint8), mask=off_2 < N)
        tl.store(dropout_mask + off_3, mask3.to(tl.uint8), mask=off_3 < N)


NUM_SIPS = 24


def _choose_block(dtype):
    if dtype.itemsize <= 2:
        return 4096
    return 2048


def dropout(input, p, train=True):
    logger.debug("GEMS_ENFLAME NATIVE_DROPOUT_FORWARD")
    if not train or p == 0:
        out = input.clone()
        mask = torch.ones_like(input, dtype=torch.bool)
        return out, mask
    if p == 1:
        out = torch.zeros_like(input)
        mask = torch.zeros_like(input, dtype=torch.bool)
        return out, mask
    assert p > 0.0 and p < 1.0, "p must be in (0, 1)"
    device = input.device
    input = input.contiguous()
    out = torch.empty_like(input)
    N = input.numel()
    mask_u8 = torch.empty(N, dtype=torch.uint8, device=device)

    BLOCK = _choose_block(input.dtype)
    NUM_BLOCKS = triton.cdiv(N, BLOCK * UNROLL)
    increment = triton.cdiv(N, UNROLL)

    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        if NUM_BLOCKS <= 256:
            dropout_forward_kernel[(NUM_BLOCKS,)](
                input,
                out,
                mask_u8,
                N,
                p,
                philox_seed,
                philox_offset,
                BLOCK=BLOCK,
            )
        else:
            grid_size = min(NUM_BLOCKS, NUM_SIPS)
            dropout_persistent_kernel[(grid_size,)](
                input,
                out,
                mask_u8,
                N,
                p,
                philox_seed,
                philox_offset,
                NUM_BLOCKS=NUM_BLOCKS,
                BLOCK=BLOCK,
            )

    mask = mask_u8.view(input.shape).view(torch.bool)
    return out, mask
