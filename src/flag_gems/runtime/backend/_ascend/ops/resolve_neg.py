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
import triton.runtime.driver as driver

logger = logging.getLogger(__name__)


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


@triton.jit
def resolve_neg_kernel(
    inp,
    out,
    data_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_DATA_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    iter_num = tl.cdiv(BLOCK_SIZE, MAX_DATA_SIZE)

    for idx in tl.range(0, iter_num):
        offsets = pid * BLOCK_SIZE + idx * MAX_DATA_SIZE + tl.arange(0, MAX_DATA_SIZE)
        mask = offsets < data_len
        inp_val = tl.load(inp + offsets, mask=mask)
        out_val = -inp_val
        tl.store(out + offsets, out_val, mask=mask)


def resolve_neg(A: torch.Tensor):
    logger.debug("GEMS_ASCEND RESOLVE_NEG")

    if A.is_neg():
        data_len = A.numel()
        out = torch.empty(A.numel(), dtype=A.dtype, device=A.device)

        CORE_NUM = get_npu_properties()["num_vectorcore"]
        BLOCK_SIZE = math.ceil(data_len / CORE_NUM)
        MAX_DATA_SIZE = 20 * 1024

        grid = lambda meta: (triton.cdiv(data_len, meta["BLOCK_SIZE"]),)
        resolve_neg_kernel[grid](
            A,
            out,
            data_len,
            BLOCK_SIZE,
            MAX_DATA_SIZE,
        )
        out = out.view(A.shape)
        return out
    else:
        return A
