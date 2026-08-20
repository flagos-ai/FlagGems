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
from enum import Enum

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def kernel_1(inp, target, mid, M, BLOCK_SIZE: tl.constexpr, reduction: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    target_ptrs = target + offset
    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(tl.float32)
    target_val = tl.load(target_ptrs, mask=mask, other=0).to(tl.float32)
    sub = inp_val - target_val
    pow_val = sub * sub
    # Reduction.MEAN.value: 1 Reduction.SUM.value: 2
    if reduction == 1:
        sum_val = tl.sum(pow_val) / M
    else:
        sum_val = tl.sum(pow_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(tl.float32)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def func(x, y):
    return (x - y) * (x - y)


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


def mse_loss(inp, target, reduction=Reduction.MEAN.value):
    logger.debug("GEMS_KUNLUNXIN MSE_LOSS")
    if reduction == Reduction.NONE.value:
        return func(inp, target)

    inp = inp.contiguous()
    target = target.contiguous()
    M = inp.numel()
    dtype = inp.dtype

    # Block sizing follows the kunlunxin `sum` global-reduction recipe:
    # get_block_size_1d divides work across the 12 clusters and caps the tile at
    # the per-core buffer (paired with buffer_size_limit=2048 at launch) so each
    # program's load is a bounded stride-1 block DMA. mse_loss loads TWO tensors
    # (inp + target) per block, so budget for 2x the element size; sizing for a
    # single tensor (as `sum` does) picks a block wide enough to overrun the
    # per-core buffer and the fp16/bf16 reduction silently truncates to half the
    # true sum. The previous bf16-mean special case instead forced mid_size=12 ->
    # block_size=next_pow2(M/12), a tile of tens of millions of elements for large
    # M; that unbounded tile hung the XPU watchdog ("wait for noc idle timeout" /
    # kl3ChannelCheckErrors 721) under do_bench.
    block_size = get_block_size_1d(M, inp.element_size() * 2)

    # kernel_2 reduces all `mid_size` block partials in ONE tl.sum tile. On XPU,
    # with buffer_size_limit=2048 a 1D tl.sum is only complete up to 32768 lanes;
    # beyond that it silently drops the tail. For fp32 the get_block_size_1d cap
    # is 16384, so at M ~ 6.5e8 (benchmark's [10000, 65536]) mid_size hits 40000
    # -> block_mid = next_pow2(40000) = 65536 and the final reduction drops ~82%
    # of the partials (verified: sum 7232 vs 40000), i.e. the previous result was
    # silently WRONG. Grow the first-stage block just enough to keep
    # mid_size <= 32768 so the second stage stays inside the correctness ceiling.
    # This only ever enlarges the block for that huge-M/fp32 case; every other
    # benchmark shape already satisfies the bound, so their (small, fast to
    # compile) blocks are untouched.
    MAX_MID = 32768
    if triton.cdiv(M, block_size) > MAX_MID:
        block_size = triton.next_power_of_2(triton.cdiv(M, MAX_MID))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    # Always accumulate the block partials in fp32: summing O(M/block) partials in
    # a low-precision dtype (bf16 especially) loses accuracy, and fp32 mid matches
    # the generic implementation.
    mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    import os

    os.environ["TRITONXPU_OTHER_SIM"] = "1"

    with torch_device_fn.device(inp.device):
        kernel_1[(mid_size, 1, 1)](
            inp, target, mid, M, block_size, reduction, buffer_size_limit=2048
        )
        if mid_size == 1:
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            return mid.reshape([]).to(dtype)
        kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid, buffer_size_limit=2048)

    if "TRITONXPU_OTHER_SIM" in os.environ:
        del os.environ["TRITONXPU_OTHER_SIM"]

    return out
