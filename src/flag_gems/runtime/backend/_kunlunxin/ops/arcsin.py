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
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.xpu.libdevice as xpu

from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

os.environ.setdefault("TRITONXPU_BF16_FAST", "1")

#
MIN_BLOCK = 2048
UNROLL_NUM = 8
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    if n_elements >= 1048576 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 8192 and n_elements % 8192 == 0:
        return 8192, 8, False
    if n_elements <= 65536:
        return 2048, 4, True
    return 8192, 8, True


@triton.jit
def arcsin_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    t = 0.5 - 0.5 * tl.abs(x)
    s = t * xpu.rsqrt(t + 1e-30)
    p = 0.0962260290980339
    p = p * t + 0.008193825371563435
    p = p * t + 0.08233591169118881
    p = p * t + 0.1661728471517563
    p = p * t + 1.0000052452087402
    v = (s * p) * 2.0
    w = 1.5707964 - v
    m = tl.minimum(1.0, tl.maximum(0.0, -x * 8.50705917e37))
    r = w * (1.0 - 2.0 * m)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def arcsin_kernel_unmasked(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    t = 0.5 - 0.5 * tl.abs(x)
    s = t * xpu.rsqrt(t + 1e-30)
    p = 0.0962260290980339
    p = p * t + 0.008193825371563435
    p = p * t + 0.08233591169118881
    p = p * t + 0.1661728471517563
    p = p * t + 1.0000052452087402
    v = (s * p) * 2.0
    w = 1.5707964 - v
    m = tl.minimum(1.0, tl.maximum(0.0, -x * 8.50705917e37))
    r = w * (1.0 - 2.0 * m)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty))


def _launch(x, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        arcsin_kernel[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        arcsin_kernel_unmasked[grid](
            x,
            out,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def arcsin(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ARCSIN")
    xc = x.contiguous()
    if out is None:
        out = torch.empty_like(xc)
        _launch(xc, out)
        return out
    oc = out.contiguous()
    _launch(xc, oc)
    if oc.data_ptr() != out.data_ptr():
        out.copy_(oc.view(out.shape))
    return out


def arcsin_(x):
    logger.debug("GEMS_KUNLUNXIN ARCSIN_")
    xc = x.contiguous()
    _launch(xc, xc)
    if xc.data_ptr() != x.data_ptr():
        x.copy_(xc.view(x.shape))
    return x


def arcsin_out(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ARCSIN OUT")
    return arcsin(x, out=out)
