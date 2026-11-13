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

"""t8 4/8 addcmul 候选 v1：小 shape 手写 1D kernel，其余回退公共 codegen。

生产走 pointwise_dynamic（gcu400 配置 max_tile=65536 / max_grid=(48,1,1)），
小 shape 的延迟主要是 codegen Python wrapper 开销。快速路径要求三个输入
同 shape / 同 dtype / contiguous / value 为标量 / n<=65536，其余全部回退，
保证语义不变。HP=True 用 fp32 中间计算，与生产逐位一致。
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
    # 内联检查：把最便宜的 out/numel 判断放最前，让大 shape 回退路径
    # 几乎零额外开销（v1 的 _fast_ok 让 float16/1M 劣化到 0.930x）。
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
    return addcmul(inp, tensor1, tensor2, value=value, out=out)
