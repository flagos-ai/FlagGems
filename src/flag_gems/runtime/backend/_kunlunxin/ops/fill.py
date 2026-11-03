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
from flag_gems.utils.shape_utils import heuristics_for_num_warps, volume

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseDtypeConvert=True,
)


fill_scalar_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseDtypeConvert=True,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, "DEFAULT")],
    num_outputs=1,
    config=fill_scalar_config,
)
@triton.jit
def fill_scalar_func(inp, value_scalar):
    return tl.full(inp.shape, value_scalar, dtype=inp.dtype)


def fill_scalar(input, value):
    logger.debug("GEMS_KUNLUNXIN FILL")
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_scalar_func(input, value, out0=out)


def fill_scalar_out(input, value, *, out=None):
    # The generic ops/fill.py fill_scalar_out routes through a NO-config
    # pointwise kernel whose store is judged discrete (lm2gm offsetState=-1) on
    # XPU -> ~0.002-0.003 speedup on large shapes. Reuse the kunlunxin-tuned
    # fill_scalar_func (prefer_1d_tile) so the write is a contiguous block DMA.
    logger.debug("GEMS_KUNLUNXIN FILL_SCALAR_OUT")
    if out is None:
        return fill_scalar(input, value)
    with torch_device_fn.device(input.device):
        fill_scalar_func(input, value, out0=out)
    return out


def _check_value_0d(value):
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )


# NOTE(KUNLUNXIN, 2026-09-19): the 0-d tensor `value` is materialized on the host
# (`value.item()`) and the fill is done by `fill_scalar_func`, whose `value` is a
# *scalar* argument (no per-lane tensor load of `value`).  This matches ATen's own
# semantics -- native `fill_.Tensor` is literally `self.fill_(value.item())` -- and
# avoids a reproducible device fault: the old `pointwise_dynamic(is_tensor=[True,
# True])` tensor-value path emitted a per-lane 0-stride tensor load of `value` at
# the full `tile_size`, which on this backend raises `KL_XID_KERNEL_EXCEPTION` /
# `status 700` (`reason[4] load/store operation exceed memory size`) for 1-byte
# dtypes.  Deterministic at `int8 (1024, 1024)`: gems faulted, while both the
# native op and the scalar path (`aten.fill.Scalar`) were verified fine on the very
# same shape/dtype.  Evidence: harness/solution/fill_tensor/README.md (section 2).
def fill_tensor(input, value):
    logger.debug("GEMS_KUNLUNXIN FILL")
    _check_value_0d(value)
    return fill_scalar(input, value.item())


@triton.jit
def _fill_tensor_out_kernel(out_ptr, n_elements, value, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(
        out_ptr + offs,
        tl.full([BLOCK_SIZE], value, dtype=out_ptr.dtype.element_ty),
        mask=mask,
    )


def fill_tensor_out(input, value, *, out=None):
    logger.debug("GEMS_KUNLUNXIN FILL_TENSOR_OUT")
    if out is None:
        return fill_tensor(input, value)
    _check_value_0d(value)
    N = volume(input.shape)
    grid_fn = (12, 1, 1)
    block_size = triton.next_power_of_2(triton.cdiv(N, 12))
    num_warps = heuristics_for_num_warps(block_size)
    with torch_device_fn.device(input.device):
        _fill_tensor_out_kernel[grid_fn](
            out,
            N,
            value.item(),
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            isCloseDtypeConvert=True,
        )
    return out


def fill_tensor_(self, value):
    logger.debug("GEMS_KUNLUNXIN FILL_TENSOR_")
    _check_value_0d(value)
    return fill_scalar_(self, value.item())


def fill_scalar_(self, value):
    logger.debug("GEMS_KUNLUNXIN FILL_SCALAR_")
    with torch_device_fn.device(self.device):
        fill_scalar_func(self, value, out0=self)
    return self
