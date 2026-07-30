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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic, tl_extra_shim
from flag_gems.utils.triton_version_utils import HAS_TLE

logger = logging.getLogger(__name__)
exp = tl_extra_shim.exp

if HAS_TLE:
    import triton.experimental.tle.language as tle
else:
    tle = None


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 2 ** math.ceil(math.log2(x))


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def glu_kernel(a, b):
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    result = a * sigmoid_b
    return result


# ============================================================================
# TLE kernel (static extract_tile)
# From: FlagTree python/tutorials/tle/05-glu.py
# ============================================================================

if HAS_TLE:

    @triton.jit
    def glu_kernel_tle(
        x_ptr,
        out_ptr,
        D,
        stride_xn,
        stride_outn,
        D_P2: tl.constexpr,
        D2_P2: tl.constexpr,
    ):
        pid_n = tl.program_id(0)

        offs = tl.arange(0, D2_P2)
        mask = offs < (D * 2)
        halo = tl.load(x_ptr + pid_n * stride_xn + offs, mask=mask, other=0.0)

        a_tile = tle.extract_tile(halo, index=[0], tile_shape=[D_P2])
        b_tile = tle.extract_tile(halo, index=[1], tile_shape=[D_P2])

        a_f32 = a_tile.to(tl.float32)
        b_f32 = b_tile.to(tl.float32)
        sigmoid_b = 1.0 / (1.0 + tl.exp(-b_f32))
        result = a_f32 * sigmoid_b

        offs_d = tl.arange(0, D_P2)
        mask_d = offs_d < D
        tl.store(
            out_ptr + pid_n * stride_outn + offs_d,
            result.to(out_ptr.dtype.element_ty),
            mask=mask_d,
        )


@pointwise_dynamic(
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ]
)
@triton.jit
def glu_backward_kernel(grad_output, a, b):
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    da = grad_output * sigmoid_b
    db = grad_output.to(tl.float32) * a * sigmoid_b * (1.0 - sigmoid_b)
    return da, db


def glu(self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logger.debug("GEMS GLU FORWARD")
    D2 = self.shape[-1]
    D = D2 // 2
    if HAS_TLE and dim == -1 and D < 8192:
        logger.debug("GEMS GLU FORWARD (TLE extract_tile path)")
        N = 1
        for d in self.shape[:-1]:
            N *= d

        x = self.reshape(N, D2)
        out = torch.empty((N, D), device=self.device, dtype=self.dtype)
        d_p2 = _next_pow2(D)
        d2_p2 = _next_pow2(D2)

        with torch_device_fn.device(self.device):
            glu_kernel_tle[(N,)](
                x,
                out,
                D,
                x.stride(0),
                out.stride(0),
                D_P2=d_p2,
                D2_P2=d2_p2,
                num_warps=4,
                num_stages=2,
            )
        return out.reshape(self.shape[:-1] + (D,))

    # Split into a and b
    a, b = torch.chunk(self, 2, dim=dim)
    out = glu_kernel(a, b)
    return out


def glu_backward(grad_output, self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logger.debug("GEMS GLU BACKWARD")
    a, b = torch.chunk(self, 2, dim=dim)
    grad_input = torch.empty_like(self, memory_format=torch.contiguous_format)
    grad_a, grad_b = torch.chunk(grad_input, 2, dim=dim)
    glu_backward_kernel(grad_output, a, b, out0=grad_a, out1=grad_b)
    return grad_input
