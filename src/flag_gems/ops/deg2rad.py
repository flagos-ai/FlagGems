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
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_SCALE = math.pi / 180.0


@triton.jit
def deg2rad_kernel(x_ptr, y_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * scale
    tl.store(y_ptr + offsets, y, mask=mask)


def _launch(x_contig: torch.Tensor, out_contig: torch.Tensor):
    n_elements = out_contig.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    deg2rad_kernel[grid](x_contig, out_contig, n_elements, _SCALE, BLOCK_SIZE=1024)


def deg2rad(x: torch.Tensor):
    logger.debug("GEMS DEG2RAD")
    if x.is_floating_point():
        result_dtype = x.dtype
    else:
        result_dtype = torch.float32

    x_contig = x.to(result_dtype).contiguous()
    out = torch.empty_like(x_contig)
    _launch(x_contig.view(-1), out.view(-1))
    return out.view_as(x)


def deg2rad_(x: torch.Tensor):
    logger.debug("GEMS DEG2RAD_")
    if not x.is_floating_point():
        x.mul_(_SCALE)
        return x

    if x.is_contiguous():
        _launch(x.view(-1), x.view(-1))
    else:
        x_contig = x.contiguous()
        _launch(x_contig.view(-1), x_contig.view(-1))
        x.copy_(x_contig)
    return x


def deg2rad_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS DEG2RAD_OUT")
    x_contig = x.to(out.dtype).contiguous()

    if out.is_contiguous():
        _launch(x_contig.view(-1), out.view(-1))
    else:
        out_contig = torch.empty_like(out, memory_format=torch.contiguous_format)
        _launch(x_contig.view(-1), out_contig.view(-1))
        out.copy_(out_contig)
    return out
