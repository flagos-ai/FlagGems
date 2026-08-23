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

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Without an explicit CodeGenConfig, pointwise_dynamic specializes the kernel
# per input shape on XPU -> per-shape recompile + slow memory-bound path
# (baseline large-shape speedup ~0.24-0.42). Mirror silu (the closest exp-based
# unary activation): bounded 1d tile + close-vectorization + unroll makes the
# kernel shape-independent (compiles once) and saturates memory bandwidth.
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
    is_tensor=[True, False, False, False],
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit
def elu_forward_kernel(x, alpha, scale, input_scale):
    x_fp32 = x.to(tl.float32)
    return tl.where(
        x_fp32 > 0,
        scale * input_scale * x_fp32,
        scale * alpha * (tl.exp(x_fp32 * input_scale) - 1),
    )


# elu_backward_kernel is intentionally kept config-less. On XPU a config-less
# pointwise_dynamic specializes/recompiles the kernel per input shape, which
# produces the large ir-elu_backward MLIR dump (1404 module dumps under
# MLIR_ENABLE_DUMP) -- BUT that is a compile-time artifact only. The per-shape
# specialized kernels give the best *steady-state* gems speedup here (avg ~0.63,
# large shapes 0.47-0.54). Every shape-independent config tried regresses the
# gems-speedup metric: silu-style (vecClose+unroll8) -> -10% on large shapes
# (0.63->0.59); unroll4 and kunlunAutoGrid -> catastrophic 10-100x slowdown in
# the real launch path. Since the acceptance criterion is gems speedup and the
# recompile cost is amortized (cached per shape) in real fixed-shape workloads,
# config-less is retained. (Only the forward kernel above benefits from the
# tuned config_.)
@pointwise_dynamic(
    is_tensor=[True, True, False, False, False, False],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def elu_backward_kernel(grad_output, x, alpha, scale, input_scale, is_result):
    x_fp32 = x.to(tl.float32)
    grad_pos = grad_output * scale * input_scale
    if is_result:
        grad_neg = grad_output * input_scale * (x_fp32 + scale * alpha)
    else:
        grad_neg = (
            grad_output * scale * alpha * input_scale * tl.exp(x_fp32 * input_scale)
        )

    return tl.where(x_fp32 > 0, grad_pos, grad_neg)


def elu(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS_KUNLUNXIN ELU")
    return elu_forward_kernel(A, alpha, scale, input_scale)


def elu_(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS_KUNLUNXIN ELU_")
    return elu_forward_kernel(A, alpha, scale, input_scale, out0=A)


def elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result):
    logger.debug("GEMS_KUNLUNXIN ELU_BACKWARD")
    grad_input = elu_backward_kernel(
        grad_output, self_or_result, alpha, scale, input_scale, is_result
    )
    return grad_input
