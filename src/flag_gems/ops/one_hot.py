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


@triton.jit
def one_hot_kernel(
    index_ptr,
    out_ptr,
    num_classes,
    numel,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    row_start = pid * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < numel

    target_classes = tl.load(index_ptr + row_offsets, mask=row_mask, other=0)

    for col_st in range(0, num_classes, BLOCK_N):
        col_offsets = col_st + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < num_classes
        result = target_classes[:, None] == col_offsets[None, :]
        result = result.to(tl.int64)
        offs_2d = row_offsets[:, None] * num_classes + col_offsets[None, :]
        tl.store(out_ptr + offs_2d, result, mask=row_mask[:, None] & col_mask[None, :])


def one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    logger.debug("GEMS ONE_HOT")
    if num_classes == -1:
        if tensor.numel() == 0:
            # torch's composite reports this itself; torch_npu's max() on an
            # empty tensor does not, so state it explicitly.
            raise RuntimeError(
                "Can not infer total number of classes from empty tensor."
            )
        num_classes = int(tensor.max().item()) + 1

    if not tensor.is_cuda:
        # Expand the composite here rather than delegating to
        # torch.nn.functional.one_hot. That function is
        # CompositeImplicitAutograd, so calling it from this kernel re-enters
        # flag_gems' own registration as soon as the Autograd key is excluded
        # (torch.inference_mode) and recurses until the stack overflows.
        # zeros + scatter_ is the decomposition torch itself applies.
        #
        # The class-value bounds are checked here explicitly because scatter_
        # does not report them on every backend (torch_npu's one_hot silently
        # accepts negative classes), while the CPU/aten composite does.
        if tensor.numel() > 0:
            if int(tensor.min().item()) < 0:
                raise RuntimeError("Class values must be non-negative.")
            if int(tensor.max().item()) >= num_classes:
                raise RuntimeError("Class values must be smaller than num_classes.")
        out = torch.zeros(
            (*tensor.shape, num_classes), dtype=torch.int64, device=tensor.device
        )
        return out.scatter_(-1, tensor.unsqueeze(-1), 1)

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    numel = tensor.numel()

    out = torch.empty(
        (*tensor.shape, num_classes), device=tensor.device, dtype=torch.int64
    )
    BLOCK_N = triton.next_power_of_2(num_classes)
    BLOCK_N = min(BLOCK_N, 128)
    BLOCK_M = 32

    grid = (triton.cdiv(numel, BLOCK_M),)

    one_hot_kernel[grid](
        tensor,
        out,
        num_classes,
        numel,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out
