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

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(1, 2, "NO_OPMATH")],
)
@triton.jit
def where_inner(condition, self, other):
    return tl.where(condition, self, other)


def _materialize_full(t, out_shape, device, dtype=None):
    if dtype is not None and t.dtype != dtype:
        if t.device.type == "cpu" or t.numel() <= 1:
            t = t.cpu().to(dtype)
        else:
            t = t.to(dtype)

    if tuple(t.shape) == tuple(out_shape) and t.device == device:
        return t if t.is_contiguous() else t.contiguous()

    # Expand on CPU (correct cast + no TX81 0-stride), then one H2D copy.
    cpu_t = t.detach().cpu()
    if dtype is not None and cpu_t.dtype != dtype:
        cpu_t = cpu_t.to(dtype)
    cpu_t = cpu_t.expand(out_shape).contiguous()
    out_t = torch.empty(out_shape, dtype=cpu_t.dtype, device=device)
    out_t.copy_(cpu_t)
    return out_t


def where_self_out(condition, self, other, out=None):
    logger.debug("GEMS_TSINGMICRO WHERE_SELF_OUT")
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

    devices = [x.device for x in (c, a, b) if x.device.type != "cpu"]
    assert len(devices), "CPU only. There seems a mistake to dispatch to here."
    assert (
        len(set(devices)) == 1
    ), f"Expected all tensors to be on the same device, but found at least two devices, {devices}"
    device = devices[0]

    assert (
        c.dtype == torch.bool
    ), f"where expected condition to be a boolean tensor, but got a tensor with dtype {condition.dtype}"

    out_shape = torch.broadcast_shapes(c.shape, a.shape, b.shape)
    if out is None:
        out = torch.empty(out_shape, dtype=result_type, device=device)

    c = _materialize_full(c, out_shape, device)
    a = _materialize_full(a, out_shape, device, dtype=result_type)
    b = _materialize_full(b, out_shape, device, dtype=result_type)

    ndim = len(out_shape)
    where_inner.instantiate(ndim)
    where_inner(c, a, b, out0=out)
    return out


def where_self(condition, self, other):
    logger.debug("GEMS_TSINGMICRO WHERE_SELF")
    return where_self_out(condition, self, other)


def _scalar_to_full_tensor(scalar, like):
    cpu = torch.full(like.shape, scalar, dtype=like.dtype, device="cpu")
    out = torch.empty(like.shape, dtype=like.dtype, device=like.device)
    out.copy_(cpu)
    return out


def where_scalar_self(condition, self, other):
    logger.debug("GEMS_TSINGMICRO WHERE_SCALAR_SELF")
    result_type = torch.result_type(self, other)
    if other.dtype != result_type:
        other = other.to(result_type)
    self_tensor = _scalar_to_full_tensor(self, other)
    return where_self(condition, self_tensor, other)


def where_scalar_other(condition, self, other):
    logger.debug("GEMS_TSINGMICRO WHERE_SCALAR_OTHER")
    result_type = torch.result_type(self, other)
    if self.dtype != result_type:
        self = self.to(result_type)
    other_tensor = _scalar_to_full_tensor(other, self)
    return where_self(condition, self, other_tensor)
