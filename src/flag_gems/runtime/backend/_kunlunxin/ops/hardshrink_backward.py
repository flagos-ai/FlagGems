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

# Kunlunxin (XPU) override of hardshrink_backward / hardshrink_backward_grad_input.
#
# The generic `flag_gems.ops.hardshrink_backward` uses a raw @triton.jit kernel
# with `grid = lambda meta: (...)` (per-launch recompilation on the xpubin
# backend, ~130-200ms per call regardless of size) and a
# `tl.where(tl.abs(x) <= lambd, 0, grad_out)` body (fcmp-derived select = the
# known slow path, ~4-8x penalty vs pure copy). Following the established
# Kunlunxin pointwise recipe (hardsigmoid_backward / mish_backward): the same
# semantics are expressed with pure FP arithmetic (no i1/select) and compiled
# with a bounded 1D-tile CodeGenConfig (kunlunAutoGrid + prefer_1d_tile +
# unroll_num=8 + buffer_size_limit=4096).
#
# Semantics (ATen): grad = 0 where |self| <= lambd, else grad_out. Ordered
# comparison means NaN/+-inf inputs fail "abs(x) <= lambd" and pass the
# gradient through. The kernel first clamps x to +-1e30: minnum/maxnum min/max
# map NaN -> 1e30 and +-inf -> +-1e30, so |xc| > lambd holds for every
# non-finite input (lambd in the real-world/test range is <= 1.0). It then uses
# the saturation step t = min(1, max(0, (|xc| - lambd) * 1e30 * 1e30)) which is
# exactly {1 : |xc| > lambd, 0 : |xc| <= lambd} for finite x (any non-zero f32
# difference saturates: min positive diff 2^-150), avoiding the vcmpf/vselect
# path entirely. Known deviation (out-of-matrix, not covered by torch tests):
# lambd > 1e30 (test lambds are 0.5 / 1.0 / 0.0) would misroute NaN; the
# equality case |x| == lambd and -0.0 are handled exactly.
import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def hardshrink_backward_func(grad_output, self, lambd):
    g32 = grad_output.to(tl.float32)
    x32 = self.to(tl.float32)
    # Clamp to +-1e30 first: minnum/maxnum min/max map NaN -> 1e30 and
    # +-inf -> +-1e30, so every non-finite (and |x| > 1e30) input satisfies
    # |xc| > lambd and the gradient passes through, matching ATen.
    xc = tl.minimum(tl.maximum(x32, -1.0e30), 1.0e30)
    # Saturation step: t = min(1, max(0, (|xc|-lambd)*1e30*1e30)) is exactly
    # {1 : |xc| > lambd, 0 : |xc| <= lambd} for finite x (1e60 saturates any
    # non-zero f32 difference). The x*t form avoids the vcmpf/vselect path
    # which measures 4-8x slower than pure copy on this backend.
    t = tl.minimum(tl.maximum((tl.abs(xc) - lambd) * 1e30 * 1e30, 0.0), 1.0)
    return (g32 * t).to(grad_output.dtype)


def hardshrink_backward(grad_out, self_t, lambd=0.5):
    logger.debug("GEMS_KUNLUNXIN HARDSHRINK_BACKWARD")
    return hardshrink_backward_func(grad_out, self_t, lambd)


def hardshrink_backward_grad_input(grad_out, self_t, lambd=0.5, *, grad_input=None):
    logger.debug("GEMS_KUNLUNXIN HARDSHRINK_BACKWARD_GRAD_INPUT")
    if grad_input is None:
        raise ValueError(
            "Argument 'grad_input' must be provided for hardshrink_backward.grad_input."
        )
    hardshrink_backward_func(grad_out, self_t, lambd, out0=grad_input)
    return grad_input
