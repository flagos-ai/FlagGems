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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from .copy import _pick_flat_block

logger = logging.getLogger(__name__)


@triton.jit
def _nested_select_backward_seg_kernel(
    grad_ptr,
    out_ptr,
    blocks_head,
    blocks_copy,
    lo,
    hi,
    n_head,
    n_copy,
    n_tail,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    if pid < blocks_head:
        o = pid * BLOCK_SIZE + offsets
        tl.store(out_ptr + o, 0.0, mask=o < n_head)
    elif pid < blocks_head + blocks_copy:
        o = (pid - blocks_head) * BLOCK_SIZE + offsets
        m = o < n_copy
        tl.store(out_ptr + lo + o, tl.load(grad_ptr + o, mask=m), mask=m)
    else:
        o = (pid - blocks_head - blocks_copy) * BLOCK_SIZE + offsets
        tl.store(out_ptr + hi + o, 0.0, mask=o < n_tail)


def _nested_select_backward(grad, self, dim, index):
    logger.debug("GEMS_KUNLUNXIN _NESTED_SELECT_BACKWARD")
    if not self.is_nested:
        raise ValueError(
            "_nested_select_backward expects `self` to be a nested tensor."
        )

    ndim = self.dim()
    if dim < 0:
        dim += ndim
    if dim != 0:
        raise NotImplementedError(
            f"_nested_select_backward only supports dim == 0, got dim == {dim}."
        )

    buffer = self.values()
    if not buffer.is_contiguous():
        buffer = buffer.contiguous()
    offsets = self.offsets()
    if not offsets.is_contiguous():
        offsets = offsets.contiguous()
    num_components = offsets.numel() - 1
    if index < 0:
        index += num_components
    start = int(offsets[index].item())
    end = int(offsets[index + 1].item())

    if not grad.is_contiguous():
        grad = grad.contiguous()
    n_cols = buffer.shape[-1] if buffer.dim() > 1 else 1

    # torch.empty: output allocation only. Every element of `out` is written by
    # the kernel below (head zeros + grad + tail zeros cover [0, total)), so no
    # uninitialized element can escape and no aten op produces any output value.
    out = torch.empty(buffer.shape, dtype=grad.dtype, device=grad.device)
    total = out.numel()

    n_copy = min(grad.numel(), max(end - start, 0) * n_cols)
    lo = start * n_cols
    hi = lo + n_copy
    n_head = lo
    n_tail = total - hi

    block = _pick_flat_block(total) if total > 0 else 1024
    blocks_head = triton.cdiv(n_head, block)
    blocks_copy = triton.cdiv(n_copy, block)
    blocks_tail = triton.cdiv(n_tail, block)
    n_programs = blocks_head + blocks_copy + blocks_tail
    # Empty buffer / degenerate component: emit no launch at all. An all-false
    # mask aborts the program on this backend, which would leave the output
    # buffer unwritten.
    if n_programs > 0:
        with torch_device_fn.device(out.device):
            _nested_select_backward_seg_kernel[(n_programs,)](
                grad.reshape(-1),
                out.reshape(-1),
                blocks_head,
                blocks_copy,
                lo,
                hi,
                n_head,
                n_copy,
                n_tail,
                BLOCK_SIZE=block,
                num_warps=32,
                buffer_size_limit=1024,
            )

    return torch.nested.nested_tensor_from_jagged(out, offsets=offsets)
