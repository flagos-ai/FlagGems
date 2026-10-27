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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger(__name__)


config_ = CodeGenConfig(
    760,
    (48, 1, 1),
    32,
    False,
    prefer_1d_tile=int(triton.__version__[0]) < 3,
)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(1, 2, "NO_OPMATH")],
    config=config_,
)
@triton.jit
def where_inner(condition, self, other):
    return tl.where(condition, self, other)


@triton.jit
def _where_broadcast_offset(linear, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    offset = tl.full(linear.shape, 0, tl.int64)
    remaining = linear
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        if axis == 0:
            coordinate = remaining
        else:
            coordinate = remaining % SHAPE[axis]
            remaining = remaining // SHAPE[axis]
        offset += coordinate * STRIDES[axis]
    return offset


@libentry()
@triton.jit
def _where_uniform_condition_kernel(
    condition,
    self,
    other,
    out,
    numel,
    SHAPE: tl.constexpr,
    SELF_STRIDES: tl.constexpr,
    OTHER_STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    SCALAR_SELF: tl.constexpr,
    SCALAR_OTHER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # A masked vector load from a broadcast bool pointer fails Ascend lowering.
    # Read the live device condition once, so graph replays can change its value.
    predicate = tl.load(condition) != 0
    for start in range(tle.program_id(0) * BLOCK, numel, tl.num_programs(0) * BLOCK):
        linear = start + tl.arange(0, BLOCK)
        mask = linear < numel
        if SCALAR_SELF:
            left = tl.load(self)
        else:
            self_offset = _where_broadcast_offset(linear, SHAPE, SELF_STRIDES)
            left = tl.load(self + self_offset, mask=mask, other=0)
        if SCALAR_OTHER:
            right = tl.load(other)
        else:
            other_offset = _where_broadcast_offset(linear, SHAPE, OTHER_STRIDES)
            right = tl.load(other + other_offset, mask=mask, other=0)
        out_offset = _where_broadcast_offset(linear, SHAPE, OUT_STRIDES)
        tl.store(out + out_offset, tl.where(predicate, left, right), mask=mask)


def _has_uniform_storage(tensor):
    return tensor.numel() > 0 and all(
        size == 1 or stride == 0 for size, stride in zip(tensor.shape, tensor.stride())
    )


def _where_uniform_condition(condition, self, other, out):
    if out.numel() == 0:
        return out
    shape = tuple(out.shape)

    def broadcast_strides(tensor):
        strides = tuple(
            0 if size == 1 else stride
            for size, stride in zip(tensor.shape, tensor.stride())
        )
        return (0,) * (len(shape) - tensor.ndim) + strides

    block = min(256, triton.next_power_of_2(out.numel()))
    grid = (min(config_.max_grid_size[0], triton.cdiv(out.numel(), block)),)
    with torch_device_fn.device(out.device):
        _where_uniform_condition_kernel[grid](
            condition,
            self,
            other,
            out,
            out.numel(),
            SHAPE=shape,
            SELF_STRIDES=broadcast_strides(self),
            OTHER_STRIDES=broadcast_strides(other),
            OUT_STRIDES=tuple(out.stride()),
            SCALAR_SELF=_has_uniform_storage(self),
            SCALAR_OTHER=_has_uniform_storage(other),
            BLOCK=block,
        )
    return out


def where_self_out(condition, self, other, out=None):
    logger.debug("GEMS_ASCEND WHERE_SELF_OUT")
    result_type = torch.result_type(self, other)
    if out is not None:
        assert (
            out.dtype == result_type
        ), f"Expected out type to be {result_type}, but got {out.dtype}."

    c, a, b = list(
        map(
            lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x),
            (condition, self, other),
        )
    )

    if a.dtype != result_type:
        a = a.to(result_type)
    if b.dtype != result_type:
        b = b.to(result_type)

    devices = map(lambda x: x.device, (c, a, b))
    devices = list(filter(lambda k: k.type != "cpu", devices))

    assert len(devices), "CPU only. There seems a mistake to dispatch to here."

    device = devices[0]
    if c.device != device and c.ndim == 0:
        c = c.to(device)
    if a.device != device and a.ndim == 0:
        a = a.to(device)
    if b.device != device and b.ndim == 0:
        b = b.to(device)

    assert (
        len(set(devices)) == 1
    ), f"Expected all tensors to be on the same device, but found at least two devices, {devices}"
    assert (
        c.dtype == torch.bool
    ), f"where expected condition to be a boolean tensor, but got a tensor with dtype {condition.dtype}"

    if out is None:
        out_shape = torch.broadcast_shapes(c.shape, a.shape, b.shape)
        out = torch.empty(out_shape, dtype=result_type, device=device)

    if _has_uniform_storage(c) and not out.is_complex():
        if out.shape == torch.broadcast_shapes(c.shape, a.shape, b.shape):
            return _where_uniform_condition(c, a, b, out)

    ndim = max(c.ndim, a.ndim, b.ndim)
    where_inner.instantiate(ndim)
    where_inner(c, a, b, out0=out)
    return out


def where_self(condition, self, other):
    logger.debug("GEMS_ASCEND WHERE_SELF")
    return where_self_out(condition, self, other)


def where_scalar_self(condition, self, other):
    logger.debug("GEMS_ASCEND WHERE_SCALAR_SELF")
    return where_self_out(condition, self, other)


def where_scalar_other(condition, self, other):
    logger.debug("GEMS_ASCEND WHERE_SCALAR_OTHER")
    return where_self_out(condition, self, other)
