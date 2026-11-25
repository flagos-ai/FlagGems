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

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _special_modified_bessel_i1_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    if NEED_MASK:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offsets)

    x_f32 = x.to(tl.float32)
    ax = tl.abs(x_f32)

    # Small region: |x| <= 3.75, I1(x) = x * P(x^2)
    y = x_f32 / 3.75
    y2 = y * y
    p = 0.00032411
    p = 0.00301532 + y2 * p
    p = 0.02658733 + y2 * p
    p = 0.15084934 + y2 * p
    p = 0.51498869 + y2 * p
    p = 0.87890594 + y2 * p
    p = 0.5 + y2 * p
    ans_small = x_f32 * p

    # Large region: |x| > 3.75, asymptotic expansion (Cephes)
    t = 3.75 / tl.maximum(ax, 1e-20)
    q = -0.00420059
    q = 0.01787654 + t * q
    q = -0.02895312 + t * q
    q = 0.02282967 + t * q
    q = -0.01031555 + t * q
    q = 0.00163801 + t * q
    q = -0.00362018 + t * q
    q = -0.03988024 + t * q
    q = 0.39894228 + t * q
    pref = tl.exp(ax) / tl.sqrt(tl.maximum(ax, 1e-20))
    ans_large = pref * q
    ans_large = tl.where(x_f32 < 0, -ans_large, ans_large)

    is_small = ax <= 3.75
    ans = tl.where(is_small, ans_small, ans_large)

    if NEED_MASK:
        tl.store(out_ptr + offsets, ans.to(x.dtype), mask=mask)
    else:
        tl.store(out_ptr + offsets, ans.to(x.dtype))


def _launch(x: torch.Tensor, out: torch.Tensor):
    n_elements = x.numel()
    if n_elements == 0:
        return
    BLOCK_SIZE = 1024
    need_mask = n_elements % BLOCK_SIZE != 0
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _special_modified_bessel_i1_kernel[grid](
        x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, NEED_MASK=need_mask
    )


def special_modified_bessel_i1(self: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_MODIFIED_BESSEL_I1")
    x_c = self.contiguous()
    out = torch.empty_like(x_c)
    _launch(x_c, out)
    if self.is_contiguous():
        return out
    return out.view_as(self)


def special_modified_bessel_i1_out(self: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_MODIFIED_BESSEL_I1_OUT")
    if out.dtype != self.dtype:
        raise TypeError("out dtype must match input dtype")
    if out.device != self.device:
        raise TypeError("out device must match input device")
    x_c = self.contiguous()
    out_c = out.contiguous()
    _launch(x_c, out_c)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out
