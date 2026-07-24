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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# addcmul is a memory-bound elementwise op (reads 3 tensors, writes 1) whose
# only compute is two multiplies + an add. The default pointwise_dynamic codegen
# emits a tiny 256-element tile with no unrolling, which underutilizes the XPU
# badly (baseline ~0.001-0.15x torch; see ir_addcmul_out.log tile<256xf16>).
# Fix: cover addcmul_out and use div.py's tuned config (larger buffer + unroll8),
# but keep vectorization OPEN (isCloseVectorization=False). Unlike addcdiv (whose
# division favors closing vectorization), addcmul's pure multiply-add vectorizes
# well: measured 4096^2 fp16 0.25->0.80, fp32 0.60->0.94 with vec open vs closed.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, True, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=config_,
)
@triton.jit
def addcmul_forward(x, t1, t2, value):
    return x + value * t1 * t2


def addcmul(inp, tensor1, tensor2, *, value=1.0, out=None):
    logger.debug("GEMS_KUNLUNXIN ADDCMUL")
    if out is None:
        broadcast_shape = torch.broadcast_shapes(
            inp.shape, tensor1.shape, tensor2.shape
        )
        dtype = torch.promote_types(
            inp.dtype, torch.promote_types(tensor1.dtype, tensor2.dtype)
        )
        out = torch.empty(broadcast_shape, device=inp.device, dtype=dtype)
    addcmul_forward(inp, tensor1, tensor2, value, out0=out)
    return out


def addcmul_out(inp, tensor1, tensor2, *, value=1.0, out):
    logger.debug("GEMS_KUNLUNXIN ADDCMUL_OUT")
    broadcast_shape = torch.broadcast_shapes(inp.shape, tensor1.shape, tensor2.shape)
    if list(out.shape) != list(broadcast_shape):
        out.resize_(broadcast_shape)
    addcmul_forward(inp, tensor1, tensor2, value, out0=out)
    return out


def addcmul_(inp, tensor1, tensor2, *, value=1.0):
    # In-place variant: reuse the same tuned kernel writing back into inp.
    # Without this override addcmul_ falls back to the generic untuned kernel
    # (tile<256>, no unroll) -> ~0.001-0.002x torch on large shapes.
    logger.debug("GEMS_KUNLUNXIN ADDCMUL_")
    addcmul_forward(inp, tensor1, tensor2, value, out0=inp)
    return inp
