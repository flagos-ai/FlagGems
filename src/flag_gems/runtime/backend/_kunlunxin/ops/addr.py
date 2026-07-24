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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# addr = beta*input + alpha*(vec1 ⊗ vec2), i.e. out[m,n] = beta*input[m,n]
# + alpha*vec1[m]*vec2[n]. The original hand-written 2D-tile kernel (fixed
# 32x32 BLOCK + 2D offset math) is judged discrete by XPU OffsetAnalysis
# (1800× offsetState=-1 in harness/perf_ir_3/ir-addr-dev0.log, 1590 gm2lm /
# 525 lm2gm, no vector<>) → per-element access, catastrophic.
# Reformulate as a broadcasted pointwise op (exactly like addcmul): view vec1
# as (M,1) and vec2 as (1,N) and let pointwise_dynamic broadcast them against
# input (M,N). The codegen then emits contiguous block-DMA tiles on the
# stride-1 inner dim. Reuse addcmul's tuned config (buffer4096 + unroll8) but
# keep vectorization CLOSED (isCloseVectorization=True): the vectorized
# broadcast path (0-stride vec1 (M,1) / vec2 (1,N)) produces WRONG results for
# 16-bit dtypes on XPU (fp16/bf16 (5333,497) ~2.6% elems mismatch, rel-diff
# thousands); fp32 was fine. Closing vectorization keeps all dtypes correct.
# Compute in fp32 to preserve the accuracy the old kernel got via its explicit
# .to(tl.float32).
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, True, False, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=config_,
)
@triton.jit
def addr_forward(inp, v1, v2, beta, alpha):
    return beta * inp.to(tl.float32) + alpha * (v1.to(tl.float32) * v2.to(tl.float32))


def addr(input, vec1, vec2, *, beta=1, alpha=1):
    logger.debug("GEMS_KUNLUNXIN ADDR")
    if vec1.dim() != 1 or vec2.dim() != 1:
        raise ValueError("addr: expected 1-D vectors")

    M = vec1.shape[0]
    N = vec2.shape[0]
    output_shape = (M, N)

    try:
        input_broadcasted = torch.broadcast_to(input, output_shape)
    except RuntimeError:
        raise ValueError(
            f"addr: input tensor of shape {input.shape} cannot be broadcast "
            f"to output shape {output_shape}"
        )
    out = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    addr_forward(
        input_broadcasted,
        vec1.reshape(M, 1),
        vec2.reshape(1, N),
        beta,
        alpha,
        out0=out,
    )
    return out
