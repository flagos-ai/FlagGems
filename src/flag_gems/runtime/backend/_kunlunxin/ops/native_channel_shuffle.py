# Copyright 2026, The FlagOS Contributors.
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
#
# Kunlunxin vendor implementation of native_channel_shuffle.
#
# Root cause of the pre-optimization slowness (same fingerprint as
# channel_shuffle, see harness/solution/channel_shuffle/README.md): the generic
# KernelGen implementation used `grid = lambda META: (...)` which under the
# xpubin backend makes the grid participate in the compilation cache key, so
# every single launch triggers a full recompilation (~165-460 ms/call, host
# side, invisible to the profiler) even though the device kernel itself only
# takes ~2.7 us.
#
# Fix: a single 1-D flattened permutation kernel with a constant-tuple grid:
#   * output linear-contiguous store with a pure tail-compare mask
#     (block-DMA friendly);
#   * input side decoded as gather load -- no mask, clamped address and
#     register zeroing via tl.where (other is never a runtime value);
#   * all division/modulo divisors (CHW, HW, groups, cpg) passed as runtime
#     scalar parameters, never tl.constexpr (avoids the platform's
#     deterministic miscompilation band for constexpr division);
#   * BLOCK=1024 fixed, no autotune, num_warps=4.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _native_channel_shuffle_flat_kernel(
    in_ptr,
    out_ptr,
    n_elem,
    C,
    HW,
    CHW,
    groups,
    cpg,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    out_idx = pid * BLOCK + tl.arange(0, BLOCK)

    # Decode linear output index to (n, c_out, spatial) with contiguous NCHW.
    n = out_idx // CHW
    rem = out_idx - n * CHW
    c = rem // HW
    hw = rem - c * HW

    # Channel shuffle (N, C, *spatial) == view(N, groups, cpg, *spatial)
    #   .transpose(1, 2).reshape(N, C, *spatial):
    #   output channel c_out takes input channel
    #   c_in = (c_out % groups) * cpg + (c_out // groups).
    cin = (c % groups) * cpg + c // groups
    in_idx = n * CHW + cin * HW + hw

    # Load without mask (address clamped in-bounds), zero out tail lanes with
    # a value-level tl.where; store uses a pure tail-compare mask.
    oc = tl.minimum(in_idx, n_elem - 1)
    x = tl.where(out_idx < n_elem, tl.load(in_ptr + oc), 0.0)
    tl.store(out_ptr + out_idx, x, mask=out_idx < n_elem)


def native_channel_shuffle(input: torch.Tensor, groups: int) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN NATIVE_CHANNEL_SHUFFLE")
    x = input
    if not x.is_contiguous():
        x = x.contiguous()

    # native_channel_shuffle expects (N, C, *spatial) where C is divisible by groups
    if x.ndim < 2:
        raise ValueError(
            f"Input must have at least 2 dimensions (N, C, ...), got {x.ndim}"
        )

    C = x.shape[1]
    HW = 1
    for d in x.shape[2:]:
        HW *= d

    g = int(groups)
    assert g > 0, "groups must be > 0"
    assert C % g == 0, f"C ({C}) must be divisible by groups ({g})"

    out = torch.empty_like(x)
    numel = x.numel()
    if numel == 0:
        return out

    cpg = C // g
    CHW = C * HW

    # Constant-tuple grid: the lambda form would be baked into the xpubin
    # compilation cache key and trigger a full recompilation on every call.
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    with torch_device_fn.device(x.device):
        _native_channel_shuffle_flat_kernel[grid](
            x,
            out,
            numel,
            C,
            HW,
            CHW,
            g,
            cpg,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out
