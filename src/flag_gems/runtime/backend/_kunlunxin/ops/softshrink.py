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

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def softshrink_kernel(x, lambd):
    # softshrink(x) = x > l ? x - l : (x < -l ? x + l : 0)
    #              == x - clamp(x, -l, l)
    # Select-free min/max form: XPU favours min/max over tl.where (single
    # instruction vs. compare+select), and the expression is exact for finite
    # x (|x| <= l -> x - x = +-0; x > l -> x - l; x < -l -> x + l).  NaN
    # propagates through min/max on this backend, matching F.softshrink(NaN)
    # = NaN (verified against torch on CPU).
    x32 = x.to(tl.float32)
    return (x32 - tl.minimum(tl.maximum(x32, -lambd), lambd)).to(x.dtype)


def _check_supported_dtype(t: torch.Tensor):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"Unsupported dtype {t.dtype}. Supported dtypes are float16, bfloat16, and float32."
        )


def softshrink(input: torch.Tensor, lambd: float = 0.5):
    logger.debug("GEMS_KUNLUNXIN SOFTSHRINK")
    _check_supported_dtype(input)
    return softshrink_kernel(input, lambd)


def softshrink_out(input: torch.Tensor, lambd: float = 0.5, out: torch.Tensor = None):
    logger.debug("GEMS_KUNLUNXIN SOFTSHRINK_OUT")
    if out is None:
        raise ValueError("Argument 'out' must be provided for softshrink_out.")
    if input.shape != out.shape:
        raise ValueError(
            f"Shape mismatch: input.shape={input.shape}, out.shape={out.shape}"
        )
    if input.dtype != out.dtype:
        raise TypeError(
            f"Dtype mismatch: input.dtype={input.dtype}, out.dtype={out.dtype}"
        )
    _check_supported_dtype(input)
    softshrink_kernel(input, lambd, out0=out)
    return out
