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

"""t8 4/8 addcmul candidate v1: hand-written 1D kernel for small shapes, common codegen otherwise.

Production uses pointwise_dynamic (gcu400 config max_tile=65536 / max_grid=(48,1,1)), where
small-shape latency is dominated by the codegen Python wrapper. The fast path requires three inputs
with same shape / same dtype / contiguous / scalar value / n<=65536; everything else falls back,
guaranteeing unchanged semantics. HP=True uses fp32 intermediates, bit-identical to production.
"""

import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def addcmul_forward(x, t1, t2, value):
    return x + value * t1 * t2


@triton.jit
def _addcmul_flat(
    X, T1, T2, Y, N, VALUE: tl.constexpr, BLOCK: tl.constexpr, HP: tl.constexpr
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(X + off, mask=m, other=0)
    t1 = tl.load(T1 + off, mask=m, other=0)
    t2 = tl.load(T2 + off, mask=m, other=0)
    if HP:
        y = (x.to(tl.float32) + VALUE * t1.to(tl.float32) * t2.to(tl.float32)).to(
            Y.dtype.element_ty
        )
    else:
        y = x + VALUE * t1 * t2
    tl.store(Y + off, y, mask=m)


_FAST_MAX = 65536
_BLOCK = 4096


def addcmul(inp, tensor1, tensor2, *, value=1.0, out=None):
    logger.debug("GEMS_ENFLAME ADDCMUL")
    # Inline checks: put the cheapest out/numel checks first so the large-shape fallback has
    # almost zero extra cost (v1's _fast_ok degraded float16/1M to 0.930x).
    n = inp.numel()
    if out is None and n <= _FAST_MAX and isinstance(value, (int, float)):
        if (
            inp.dtype == tensor1.dtype == tensor2.dtype
            and inp.shape == tensor1.shape == tensor2.shape
            and inp.is_contiguous()
            and tensor1.is_contiguous()
            and tensor2.is_contiguous()
        ):
            o = torch.empty_like(inp)
            block = min(_BLOCK, max(1024, triton.next_power_of_2(n)))
            hp = inp.dtype in (torch.float16, torch.bfloat16)
            _addcmul_flat[(triton.cdiv(n, block),)](
                inp, tensor1, tensor2, o, n, value, block, hp, num_warps=8
            )
            return o
    if out is not None:
        broadcast_shape = torch.broadcast_shapes(
            inp.shape, tensor1.shape, tensor2.shape
        )
        if list(out.shape) != list(broadcast_shape):
            out.resize_(broadcast_shape)
        addcmul_forward(inp, tensor1, tensor2, value, out0=out)
        return out
    return addcmul_forward(inp, tensor1, tensor2, value)


def addcmul_out(inp, tensor1, tensor2, *, value=1.0, out):
    logger.debug("GEMS_ENFLAME_GCU400 ADDCMUL_OUT")
    return addcmul(inp, tensor1, tensor2, value=value, out=out)
