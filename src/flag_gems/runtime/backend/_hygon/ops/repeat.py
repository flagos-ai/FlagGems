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

logger = logging.getLogger(__name__)

# Maximum tensor rank supported by the fixed-signature kernel. PyTorch tensors
# never exceed 8 dimensions here, and padding lower-rank tensors with leading
# size-1 dims keeps the linear-index reconstruction correct.
MAX_RANK = 8


@libentry()
@triton.jit
def repeat_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    # input shape per dim (padded with leading 1s)
    in_s0: tl.int64,
    in_s1: tl.int64,
    in_s2: tl.int64,
    in_s3: tl.int64,
    in_s4: tl.int64,
    in_s5: tl.int64,
    in_s6: tl.int64,
    in_s7: tl.int64,
    # output shape per dim (padded with leading 1s)
    out_s0: tl.int64,
    out_s1: tl.int64,
    out_s2: tl.int64,
    out_s3: tl.int64,
    out_s4: tl.int64,
    out_s5: tl.int64,
    out_s6: tl.int64,
    out_s7: tl.int64,
    # input strides per dim (contiguous input, padded with 0s)
    in_st0: tl.int64,
    in_st1: tl.int64,
    in_st2: tl.int64,
    in_st3: tl.int64,
    in_st4: tl.int64,
    in_st5: tl.int64,
    in_st6: tl.int64,
    in_st7: tl.int64,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    step = num_programs * BLOCK_SIZE

    # Grid-stride loop: each CTA processes tiles [pid, pid + num_programs, ...]
    # so a capped grid still covers arbitrarily large tensors.
    block_start = pid * BLOCK_SIZE
    while block_start < n_elements:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Output is contiguous, so its linear index decomposes into per-dim
        # coordinates from the innermost dimension outward. Each output
        # coordinate maps back to the input via modulo (the essence of repeat
        # / tiling). Slots are padded at the front, so only the last RANK
        # slots hold real dims; the `RANK >= k` guards are compile-time
        # constant, letting the compiler drop arithmetic for padding dims.
        tmp = offsets.to(tl.int64)
        in_idx = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

        if RANK >= 1:
            coord = tmp % out_s7
            tmp = tmp // out_s7
            in_idx += (coord % in_s7) * in_st7
        if RANK >= 2:
            coord = tmp % out_s6
            tmp = tmp // out_s6
            in_idx += (coord % in_s6) * in_st6
        if RANK >= 3:
            coord = tmp % out_s5
            tmp = tmp // out_s5
            in_idx += (coord % in_s5) * in_st5
        if RANK >= 4:
            coord = tmp % out_s4
            tmp = tmp // out_s4
            in_idx += (coord % in_s4) * in_st4
        if RANK >= 5:
            coord = tmp % out_s3
            tmp = tmp // out_s3
            in_idx += (coord % in_s3) * in_st3
        if RANK >= 6:
            coord = tmp % out_s2
            tmp = tmp // out_s2
            in_idx += (coord % in_s2) * in_st2
        if RANK >= 7:
            coord = tmp % out_s1
            tmp = tmp // out_s1
            in_idx += (coord % in_s1) * in_st1
        if RANK >= 8:
            coord = tmp % out_s0
            in_idx += (coord % in_s0) * in_st0

        in_vals = tl.load(in_ptr + in_idx, mask=mask)
        tl.store(out_ptr + offsets, in_vals, mask=mask)

        block_start += step


def repeat(inp: torch.Tensor, sizes) -> torch.Tensor:
    logger.debug("GEMS_HYGON REPEAT")

    in_rank = inp.dim()
    sizes_rank = len(sizes)
    in_shape = list(inp.shape)
    sizes_list = list(sizes)

    assert sizes_rank >= in_rank, (
        "Number of dimensions of repeat dims cannot be smaller than "
        "number of dimensions of tensor"
    )

    # Prepend size-1 dimensions to the input when repeat has more dims.
    if sizes_rank > in_rank:
        diff = sizes_rank - in_rank
        in_shape = [1] * diff + in_shape
        inp = inp.reshape(in_shape)

    # Compute output shape.
    is_empty = False
    out_shape = []
    for i in range(len(in_shape)):
        assert sizes_list[i] >= 0, (
            f"the number of repetitions per dimension out of range "
            f"(expected to >= 0) but got {sizes_list[i]}"
        )
        if in_shape[i] * sizes_list[i] == 0:
            is_empty = True
        out_shape.append(in_shape[i] * sizes_list[i])

    out = torch.empty(out_shape, device=inp.device, dtype=inp.dtype)
    if is_empty:
        return out

    rank = len(in_shape)
    assert rank <= MAX_RANK, f"repeat only supports up to {MAX_RANK} dims, got {rank}"

    inp = inp.contiguous()
    in_strides = list(inp.stride())

    # Pad shapes / strides to MAX_RANK with leading size-1 dims (stride 0),
    # so the fixed-signature kernel handles every rank without recompilation
    # per rank and without any host-to-device metadata transfer.
    pad = MAX_RANK - rank
    in_shape_p = [1] * pad + in_shape
    out_shape_p = [1] * pad + out_shape
    in_strides_p = [0] * pad + in_strides

    n_elements = out.numel()
    BLOCK_SIZE = 1024
    # Cap the grid and let each CTA stride over multiple tiles. This keeps the
    # launch cost bounded on very large tensors instead of spawning a CTA per
    # tile, while still saturating the device for typical sizes.
    max_ctas = 65536
    grid = (min(max_ctas, triton.cdiv(n_elements, BLOCK_SIZE)),)

    with torch_device_fn.device(inp.device):
        repeat_kernel[grid](
            inp,
            out,
            n_elements,
            *in_shape_p,
            *out_shape_p,
            *in_strides_p,
            RANK=rank,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out
