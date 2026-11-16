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

logger = logging.getLogger(__name__)
# One reduction program handles moderate inputs; large inputs stream tiles
# into an int32 atomic OR accumulator, whose low byte is the scalar bool.
# Threshold set to 16384: a larger single-program BLOCK degrades elementwise gather of non-contiguous views,
# so above the threshold split into multiple programs for a chunked reduction, lowering latency and raising parallelism.
_SINGLE_MAX = 16384
# Streaming tiles align with the threshold, balancing single-pass bandwidth against per-program register usage.
_BLOCK = 16384


@triton.jit
def _any_kernel(
    src,
    dst,
    n,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    CONTIG: tl.constexpr,
    ATOMIC: tl.constexpr,
    BLOCK: tl.constexpr,
    ONE_DIM: tl.constexpr,
    ZERO_STRIDE: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    if CONTIG:
        offset = idx
    elif ZERO_STRIDE:
        # All elements of a scalar / expanded view share one address, avoiding repeated gather of a whole block.
        offset = tl.full((BLOCK,), 0, tl.int64)
    elif ONE_DIM:
        # For a 1D strided view, scale linearly by the first dimension's stride, avoiding per-element int64 division /
        # modulo.
        offset = idx * STRIDES[0]
    else:
        rem = idx
        offset = tl.full((BLOCK,), 0, tl.int64)
        for d in tl.static_range(len(SHAPE) - 1, -1, -1):
            offset += (rem % SHAPE[d]) * STRIDES[d]
            rem = rem // SHAPE[d]
    # Bool tensors may contain noncanonical nonzero bytes; interpret every
    # nonzero byte as true, as the native reduction does.
    value = tl.load(src.to(tl.pointer_type(tl.int8)) + offset, idx < n, other=0)
    result = tl.max((value != 0).to(tl.int32), 0)
    if ATOMIC:
        tl.atomic_or(dst, result)
    else:
        tl.store(dst, result.to(tl.int8))


def _is_any_true(self):
    """Reduce every logical boolean element, including arbitrary-rank views."""
    logger.debug("GEMS_ENFLAME_GCU400 _IS_ANY_TRUE")
    if not isinstance(self, torch.Tensor):
        raise TypeError("_is_any_true: self must be a Tensor")
    if self.dtype != torch.bool:
        raise RuntimeError("_is_any_true: input must be a bool tensor")
    n = self.numel()
    with torch_device_fn.device(self.device):
        atomic = n > _SINGLE_MAX
        if atomic:
            accumulator = torch.zeros((1,), dtype=torch.int32, device=self.device)
            output = accumulator.view(torch.bool)[0]
            target = accumulator
            block, warps = _BLOCK, 4
        else:
            output = torch.empty((), dtype=torch.bool, device=self.device)
            target = output
            block = max(16, triton.next_power_of_2(n))
            warps = max(1, min(8, block // 1024))
        _any_kernel[(max(1, triton.cdiv(n, block)),)](
            self,
            target,
            n,
            SHAPE=tuple(self.shape),
            STRIDES=self.stride(),
            CONTIG=self.is_contiguous(),
            ATOMIC=atomic,
            BLOCK=block,
            ONE_DIM=len(self.shape) == 1,
            ZERO_STRIDE=all(stride == 0 for stride in self.stride()),
            num_warps=warps,
        )
    return output
