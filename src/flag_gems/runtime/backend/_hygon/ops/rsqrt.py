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

from flag_gems.ops.rsqrt import rsqrt as generic_rsqrt
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _rsqrt_kernel(input_ptr, output_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(input_ptr + offsets, mask=offsets < N, other=1.0)
    bits = x.to(tl.int32, bitcast=True)
    magnitude = bits & 0x7FFFFFFF
    mantissa = bits & 0x007FFFFF
    subnormal = (magnitude < 0x00800000) & (mantissa != 0)
    # Integer mantissa conversion avoids hardware flushing the subnormal input.
    # x = mantissa * 2**-149, hence rsqrt(x) = rsqrt(mantissa) * 2**74.5.
    argument = tl.where(subnormal, mantissa.to(tl.float32), x.to(tl.float32))
    result = tl.rsqrt(argument)
    result = tl.where(subnormal, result * 2.6713738906281536e22, result)
    result = tl.where(subnormal & (bits < 0), float("nan"), result)
    result = tl.where(
        magnitude == 0, tl.where(bits < 0, -float("inf"), float("inf")), result
    )
    tl.store(output_ptr + offsets, result, mask=offsets < N)


def rsqrt(A):
    logger.debug("GEMS_HYGON RSQRT")
    n = A.numel()
    if (
        A.dtype != torch.float32
        or not A.is_contiguous()
        or not 65536 <= n <= (1 << 31) - 1
        or A.is_conj()
        or A.is_neg()
        or A.requires_grad
    ):
        return generic_rsqrt(A)
    output = torch.empty_like(A)
    with torch_device_fn.device(A.device):
        _rsqrt_kernel[(triton.cdiv(n, 1024),)](
            A,
            output,
            n,
            BLOCK=1024,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=True,
            allow_flush_denorm=False,
        )
    return output
