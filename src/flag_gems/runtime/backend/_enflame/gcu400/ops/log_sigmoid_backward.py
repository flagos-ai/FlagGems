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
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# GCU caps grid.x at 65535 (CUDA allows 2^31-1): cap the launch and let each
# program stride over the remaining blocks so any input size stays launchable.
_MAX_GRID_X = 65535
_INT32_MAX = torch.iinfo(torch.int32).max

# torch_gcu dispatch adaptation (P21: native log_sigmoid_forward returns an
# uninitialized buffer). The native kernel writes the output correctly but
# leaves the buffer (contract: exp(-|x|)) as unallocated garbage, so any
# consumer of torch.ops.aten.log_sigmoid_forward on a GCU tensor gets poisoned
# values — most notably the autograd reference chain in
# tests/test_log_sigmoid_backward.py::test_log_sigmoid_backward_via_autograd,
# whose .backward() feeds the garbage buffer into the native
# log_sigmoid_backward and then aborts in torch.testing's mismatch printer
# (P21 max.dim defect). Route the aten default overload for the vendor
# dispatch key to the FlagGems kernel, which computes both the output and the
# buffer correctly; CPU references are unaffected (different dispatch key).
# Registration is idempotent (torch attribute flag); enable()/use_gems() later
# re-registering the same op for the same key is captured as a warning by the
# registrar, behavior unchanged (resize_as_sparse_ / sparse_resize_ pattern).
_FORWARD_RESCUE_FLAG = "_flag_gems_gcu400_log_sigmoid_forward_rescue_registered"
_FORWARD_RESCUE_LIBRARY = None


def _ensure_log_sigmoid_forward_rescue():
    global _FORWARD_RESCUE_LIBRARY
    if getattr(torch, _FORWARD_RESCUE_FLAG, False):
        return
    from .log_sigmoid_forward import log_sigmoid_forward as _forward_impl

    _FORWARD_RESCUE_LIBRARY = torch.library.Library("aten", "IMPL")
    try:
        _FORWARD_RESCUE_LIBRARY.impl(
            "log_sigmoid_forward", _forward_impl, "PrivateUse1"
        )
    except RuntimeError as error:
        logger.warning(
            "log_sigmoid_forward PrivateUse1 rescue registration skipped: %s", error
        )
    setattr(torch, _FORWARD_RESCUE_FLAG, True)


_ensure_log_sigmoid_forward_rescue()


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def log_sigmoid_backward_kernel(grad_output, self):
    self_fp32 = self.to(tl.float32)
    z = tl.exp(-tl.abs(self_fp32))
    derivative = tl.where(self_fp32 < 0.0, 1.0 / (1.0 + z), z / (1.0 + z))
    return grad_output * derivative


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def log_sigmoid_backward_buffer_kernel(grad_output, self, buffer):
    self_fp32 = self.to(tl.float32)
    z = buffer.to(tl.float32)
    derivative = tl.where(self_fp32 < 0.0, 1.0, z) / (1.0 + z)
    return grad_output * derivative


@triton.jit(do_not_specialize=["n_elements"])
def log_sigmoid_backward_contiguous_kernel(
    grad_output,
    self,
    buffer,
    grad_input,
    n_elements,
    HAS_BUFFER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    offsets_in_block = tl.arange(0, BLOCK_SIZE)
    for block_id in tl.range(pid, num_blocks, num_pids):
        offsets = block_id * BLOCK_SIZE + offsets_in_block
        mask = offsets < n_elements
        grad = tl.load(grad_output + offsets, mask=mask)
        inp = tl.load(self + offsets, mask=mask).to(tl.float32)
        if HAS_BUFFER:
            z = tl.load(buffer + offsets, mask=mask).to(tl.float32)
            derivative = tl.where(inp < 0.0, 1.0, z) / (1.0 + z)
        else:
            derivative = tl.sigmoid(-inp)
        result = grad * derivative
        tl.store(grad_input + offsets, result, mask=mask)


def _can_use_contiguous_kernel(grad_output, self, grad_input=None):
    return (
        grad_output.shape == self.shape
        and grad_output.dtype == self.dtype
        and grad_output.is_contiguous()
        and self.is_contiguous()
        and (grad_input is None or grad_input.is_contiguous())
    )


def _launch_contiguous_kernel(grad_output, self, buffer, grad_input):
    n_elements = self.numel()
    if n_elements == 0:
        return grad_input

    block_size = 512
    # Flat-offset addressing is int32 inside the kernel; past that domain (or
    # when rounding the block count up would cross it) defer to the
    # pointwise_dynamic kernel, which handles arbitrary sizes.
    if triton.cdiv(n_elements, block_size) * block_size > _INT32_MAX:
        return log_sigmoid_backward_kernel(grad_output, self, out0=grad_input)
    grid = (min(triton.cdiv(n_elements, block_size), _MAX_GRID_X),)
    # For FP32 the extra buffer read costs more than fused sigmoid recomputation.
    has_buffer = (
        buffer.numel() == n_elements
        and buffer.dtype == self.dtype
        and buffer.is_contiguous()
        and self.dtype != torch.float32
    )
    with torch_device_fn.device(self.device):
        log_sigmoid_backward_contiguous_kernel[grid](
            grad_output,
            self,
            buffer,
            grad_input,
            n_elements,
            HAS_BUFFER=has_buffer,
            BLOCK_SIZE=block_size,
        )
    return grad_input


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS LOG_SIGMOID_BACKWARD")

    if (
        buffer.shape == self.shape
        and buffer.dtype == self.dtype
        and buffer.is_contiguous()
        and self.dtype != torch.float32
    ):
        return log_sigmoid_backward_buffer_kernel(grad_output, self, buffer)
    return log_sigmoid_backward_kernel(grad_output, self)


def log_sigmoid_backward_out(grad_output, self, buffer, *, grad_input):
    logger.debug("GEMS LOG_SIGMOID_BACKWARD_OUT")

    if _can_use_contiguous_kernel(grad_output, self, grad_input):
        return _launch_contiguous_kernel(grad_output, self, buffer, grad_input)
    return log_sigmoid_backward_kernel(grad_output, self, out0=grad_input)
