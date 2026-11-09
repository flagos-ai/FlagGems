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
from flag_gems.utils import libentry, tl_extra_shim

_erf = tl_extra_shim.erf
_exp = tl_extra_shim.exp
_tanh = tl_extra_shim.tanh

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N"])
def _gelu_kernel(X, Out, N, APPROX: tl.constexpr, BLOCK: tl.constexpr):
    inv_sqrt2: tl.constexpr = 0.7071067811865476
    sqrt_2_over_pi: tl.constexpr = 0.7978845608028654
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        if APPROX == 0:
            y = 0.5 * x * (1.0 + _erf(x * inv_sqrt2))
        else:
            z = sqrt_2_over_pi * x * (1.0 + 0.044715 * x * x)
            y = 0.5 * x * (1.0 + _tanh(z))
        tl.store(Out + off, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _gelu_backward_kernel(DY, X, Out, N, APPROX: tl.constexpr, BLOCK: tl.constexpr):
    inv_sqrt2: tl.constexpr = 0.7071067811865476
    inv_sqrt_2pi: tl.constexpr = 0.3989422804014327
    sqrt_2_over_pi: tl.constexpr = 0.7978845608028654
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        dy = tl.load(DY + off, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        if APPROX == 0:
            dydx = (
                inv_sqrt_2pi * x * _exp(-0.5 * x * x) + 0.5 * _erf(x * inv_sqrt2) + 0.5
            )
        else:
            t = _tanh(sqrt_2_over_pi * x * (1.0 + 0.044715 * x * x))
            dydx = 0.5 * x * (1.0 - t * t) * (
                sqrt_2_over_pi + 0.1070322243 * x * x
            ) + 0.5 * (1.0 + t)
        tl.store(Out + off, dydx * dy, mask=mask)


def _choose_block(N, is_fp32):
    if N <= 1024:
        return 1024
    if N <= 32768:
        return triton.next_power_of_2(N)
    if is_fp32:
        return 65536
    return 4096


def _approx_code(approximate):
    return 1 if approximate == "tanh" else 0


def _launch_forward(inp, out, approximate):
    N = inp.numel()
    is_fp32 = inp.dtype == torch.float32
    BLOCK = _choose_block(N, is_fp32)
    grid_size = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(inp.device):
        _gelu_kernel[(grid_size,)](
            inp,
            out,
            N,
            _approx_code(approximate),
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out


def gelu(self, *, approximate="none"):
    logger.debug("GEMS_ENFLAME GELU")
    inp = self if self.is_contiguous() else self.contiguous()
    out = torch.empty_like(inp)
    return _launch_forward(inp, out, approximate)


def gelu_(A, *, approximate="none"):
    logger.debug("GEMS_ENFLAME GELU_")
    if not A.is_contiguous():
        A.copy_(gelu(A, approximate=approximate))
        return A
    return _launch_forward(A, A, approximate)


def gelu_backward(grad_output, self, *, approximate="none"):
    logger.debug("GEMS_ENFLAME GELU_BACKWARD")
    dy = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    inp = self if self.is_contiguous() else self.contiguous()
    out = torch.empty_like(dy)
    N = dy.numel()
    BLOCK = _choose_block(N, False)
    grid_size = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(dy.device):
        _gelu_backward_kernel[(grid_size,)](
            dy,
            inp,
            out,
            N,
            _approx_code(approximate),
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out
