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
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.bucketize import bucketize as default_bucketize
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=1),
    ],
    key=["n_elements", "n_boundaries"],
)
@triton.jit
def bucketize_kernel(
    inp_ptr,
    boundaries_ptr,
    out_ptr,
    n_elements,
    n_boundaries,
    right: tl.constexpr,
    N_BOUNDARY_ITERS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp_val = tl.load(inp_ptr + offsets, mask=mask, other=0)

    # Vectorized binary search: each lane keeps its own [lo, hi) window and
    # narrows it over a fixed iteration count (ceil(log2(n_boundaries + 1))).
    lo = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    hi = tl.full([BLOCK_SIZE], n_boundaries, dtype=tl.int32)

    for _ in range(N_BOUNDARY_ITERS):
        mid = tl.minimum((lo + hi) // 2, n_boundaries - 1)
        mid_val = tl.load(boundaries_ptr + mid)
        # right=True  -> upper_bound (first boundary strictly greater than val)
        # right=False -> lower_bound (first boundary >= val)
        if right:
            cond = mid_val <= inp_val
        else:
            cond = mid_val < inp_val
        lo = tl.where(cond, mid + 1, lo)
        hi = tl.where(cond, hi, mid)

    tl.store(out_ptr + offsets, lo, mask=mask)


# Moore Threads hardware does not support fp64 compute. The specialized kernel
# targets the real floating types; other dtypes / empty boundaries / dtype
# mismatches defer to the generic implementation for correctness.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _use_triton_kernel(inp, boundaries):
    if inp.device.type != "musa":
        return False
    if inp.dtype not in _SUPPORTED_DTYPES:
        return False
    if boundaries.numel() == 0:
        return False
    # Binary search compares input against boundaries; keep them the same
    # element type so the search is exact and no implicit promotion is needed.
    if boundaries.dtype != inp.dtype:
        return False
    return True


def bucketize(input, boundaries, *, out_int32=False, right=False):
    logger.debug("GEMS_MTHREADS BUCKETIZE")

    if not _use_triton_kernel(input, boundaries):
        return default_bucketize(input, boundaries, out_int32=out_int32, right=right)

    output_dtype = torch.int32 if out_int32 else torch.int64

    n_elements = input.numel()
    n_boundaries = boundaries.numel()
    search_iterations = math.ceil(math.log2(n_boundaries + 1))

    # Allocate a contiguous flat output so kernel stores land in the returned
    # tensor regardless of the input's memory layout (empty_like would inherit a
    # non-contiguous layout and flatten() could then copy).
    input_flat = input.contiguous().flatten()
    output_flat = torch.empty(n_elements, dtype=output_dtype, device=input.device)
    boundaries = boundaries.contiguous()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731

    with torch_device_fn.device(input.device):
        bucketize_kernel[grid](
            input_flat,
            boundaries,
            output_flat,
            n_elements,
            n_boundaries,
            right,
            search_iterations,
        )

    return output_flat.reshape(input.shape)
