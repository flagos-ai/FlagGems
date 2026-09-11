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

from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# zero / zero_ / zero.out are a pure write-only constant memset (fill with 0).
# The previous kunlunxin kernel launched grid=(CLUSTER_NUM=12,) with
# BLOCK_SIZE = next_pow2(cdiv(n, 12)), which grows UNBOUNDEDLY with n
# (4096^2 -> 2M-elem tile, 268M -> 33M-elem tile). Giant tiles trigger
# compile/execution catastrophes (baseline 250-380ms, speedup 0.000 on many
# shapes; bf16 10000x65536 measured 333ms). Fix: route through the same tuned
# pointwise_dynamic memset path as fill.py (bounded 65536 tile, grid scales
# with n). NOTE: unlike the read+write view_copy, this write-only constant
# memset must NOT use vec-OPEN+unroll8 -- that config is catastrophic here
# (fp16 268M -> 222ms, fp32 4096^2 -> 30ms measured). The plain fill config
# (default vec, isCloseDtypeConvert) tracks torch across all shapes.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseDtypeConvert=True,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, "DEFAULT")],
    num_outputs=1,
    config=config_,
)
@triton.jit
def _zero_fill(inp, value_scalar):
    return tl.full(inp.shape, value_scalar, dtype=inp.dtype)


def _launch_zero_kernel(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor
    with torch_device_fn.device(tensor.device):
        _zero_fill(tensor, 0, out0=tensor)
    return tensor


def zero(self: torch.Tensor) -> torch.Tensor:
    """aten::zero(Tensor self) -> Tensor  — in-place zero-fill, returns self."""
    logger.debug("GEMS_KUNLUNXIN ZERO")
    return _launch_zero_kernel(self)


def zero_(self: torch.Tensor) -> torch.Tensor:
    """aten::zero_(Tensor(a!) self) -> Tensor(a!)  — in-place zero-fill.

    Without this kunlunxin override, aten::zero_ falls back to the generic
    ops/zeros.py:zero_ (bare BLOCK_SIZE=1024 @triton.jit, no libentry/config),
    which on XPU launches a huge serial grid (655M elems -> 640k blocks) with
    no vectorization -> the 250-380ms catastrophes seen in ir-zero_-dev6.log.
    """
    logger.debug("GEMS_KUNLUNXIN ZERO_")
    return _launch_zero_kernel(self)


def zero_out(self: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """aten::zero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)  — writes zeros to out."""
    logger.debug("GEMS_KUNLUNXIN ZERO_OUT")
    return _launch_zero_kernel(out)
