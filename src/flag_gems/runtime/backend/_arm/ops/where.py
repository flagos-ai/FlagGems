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

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(1, 2, "NO_OPMATH")],
)
@triton.jit
def where_inner(condition, self, other):
    return tl.where(condition, self, other)


@triton.jit(do_not_specialize=["scalar", "n_elements"])
def _where_scalar_self_kernel(
    condition_ptr,
    other_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    cond = tl.load(condition_ptr + offs, mask=mask, other=0).to(tl.int1)
    other = tl.load(other_ptr + offs, mask=mask, other=0.0)
    out = tl.where(cond, scalar, other)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit(do_not_specialize=["scalar", "n_elements"])
def _where_scalar_other_kernel(
    condition_ptr,
    self_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    cond = tl.load(condition_ptr + offs, mask=mask, other=0).to(tl.int1)
    self_tensor = tl.load(self_ptr + offs, mask=mask, other=0.0)
    out = tl.where(cond, self_tensor, scalar)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit(do_not_specialize=["scalar", "n_elements"])
def _where_scalar_self_single_program_kernel(
    condition_ptr,
    other_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < n_elements
        cond = tl.load(condition_ptr + idx, mask=mask, other=0).to(tl.int1)
        other = tl.load(other_ptr + idx, mask=mask, other=0.0)
        out = tl.where(cond, scalar, other)
        tl.store(out_ptr + idx, out, mask=mask)


@triton.jit(do_not_specialize=["scalar", "n_elements"])
def _where_scalar_other_single_program_kernel(
    condition_ptr,
    self_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < n_elements
        cond = tl.load(condition_ptr + idx, mask=mask, other=0).to(tl.int1)
        self_tensor = tl.load(self_ptr + idx, mask=mask, other=0.0)
        out = tl.where(cond, self_tensor, scalar)
        tl.store(out_ptr + idx, out, mask=mask)


def _as_scalar(v):
    if isinstance(v, torch.Tensor):
        if v.numel() != 1:
            return None
        return v.item()
    if isinstance(v, (int, float, bool)):
        return v
    return None


def _where_scalar_tensor_fastpath(condition, self, other, out):
    if not isinstance(condition, torch.Tensor) or condition.dtype is not torch.bool:
        return False
    if condition.device.type != "cpu":
        return False
    if not condition.is_contiguous() or not out.is_contiguous():
        return False

    self_scalar = _as_scalar(self)
    other_scalar = _as_scalar(other)
    self_tensor = self if isinstance(self, torch.Tensor) else None
    other_tensor = other if isinstance(other, torch.Tensor) else None

    # Only specialize one-scalar + one-tensor, contiguous, same flattened size.
    if (
        self_scalar is not None
        and other_tensor is not None
        and other_tensor.is_contiguous()
    ):
        if other_tensor.numel() != condition.numel():
            return False
        if other_tensor.dtype != out.dtype:
            return False
        cond_flat = condition.view(-1)
        other_flat = other_tensor.view(-1)
        out_flat = out.view(-1)
        n = cond_flat.numel()
        if n <= 262144:
            _where_scalar_self_single_program_kernel[(1,)](
                cond_flat,
                other_flat,
                out_flat,
                float(self_scalar),
                n,
                BLOCK_SIZE=256,
                num_warps=1,
                num_stages=1,
            )
        else:
            grid = (triton.cdiv(n, 256),)
            _where_scalar_self_kernel[grid](
                cond_flat,
                other_flat,
                out_flat,
                float(self_scalar),
                n,
                BLOCK_SIZE=256,
                num_warps=1,
                num_stages=1,
            )
        return True

    if (
        other_scalar is not None
        and self_tensor is not None
        and self_tensor.is_contiguous()
    ):
        if self_tensor.numel() != condition.numel():
            return False
        if self_tensor.dtype != out.dtype:
            return False
        cond_flat = condition.view(-1)
        self_flat = self_tensor.view(-1)
        out_flat = out.view(-1)
        n = cond_flat.numel()
        if n <= 262144:
            _where_scalar_other_single_program_kernel[(1,)](
                cond_flat,
                self_flat,
                out_flat,
                float(other_scalar),
                n,
                BLOCK_SIZE=256,
                num_warps=1,
                num_stages=1,
            )
        else:
            grid = (triton.cdiv(n, 256),)
            _where_scalar_other_kernel[grid](
                cond_flat,
                self_flat,
                out_flat,
                float(other_scalar),
                n,
                BLOCK_SIZE=256,
                num_warps=1,
                num_stages=1,
            )
        return True

    return False


def where_self_out(condition, self, other, out=None):
    logger.debug("GEMS_ARM WHERE_SELF_OUT")
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

    # assert len(devices), "CPU only. There seems a mistake to dispatch to here."

    # device = devices[0]
    # if c.device != device and c.ndim == 0:
    #     c = c.to(device)
    # if a.device != device and a.ndim == 0:
    #     a = a.to(device)
    # if b.device != device and b.ndim == 0:
    #     b = b.to(device)

    # assert (
    #     len(set(devices)) == 1
    # ), f"Expected all tensors to be on the same device, but found at least two devices, {devices}"
    assert (
        c.dtype == torch.bool
    ), f"where expected condition to be a boolean tensor, but got a tensor with dtype {condition.dtype}"

    if out is None:
        out_shape = torch.broadcast_shapes(c.shape, a.shape, b.shape)
        out = torch.empty(out_shape, dtype=result_type, device=c.device)

    if _where_scalar_tensor_fastpath(c, a, b, out):
        return out

    ndim = max(c.ndim, a.ndim, b.ndim)
    where_inner.instantiate(ndim)
    where_inner(c, a, b, out0=out)
    return out


def where_self(condition, self, other):
    logger.debug("GEMS_ARM WHERE_SELF")
    return where_self_out(condition, self, other)


def where_scalar_self(condition, self, other):
    logger.debug("GEMS_ARM WHERE_SCALAR_SELF")
    return where_self_out(condition, self, other)


def where_scalar_other(condition, self, other):
    logger.debug("GEMS_ARM WHERE_SCALAR_OTHER")
    return where_self_out(condition, self, other)
