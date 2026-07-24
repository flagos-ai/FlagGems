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

from flag_gems.utils import broadcastable_to

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# masked_fill(inp, mask, value) == where(mask, value, inp): a memory-bound
# select/copy. The old hand-written kernel launched grid=12 with
# BLOCK_SIZE=next_power_of_2(cdiv(N, 12)); for the 1G-element benchmark shape
# that becomes a single 128M-wide tile -> IR explosion (the perf_ir_2 dumps are
# 1.97-4.0GB and the benchmark stalls). It also paid an extra full-tensor
# expand_mask.to(torch.int) copy (4x bytes of the bool mask). Route through the
# tuned pointwise_dynamic path instead (bounded tiles + autoGrid + unroll8 +
# libentry caching), and feed the bool mask straight into tl.where.
#
# isCloseVectorization=True (NOT the neg/view_copy OPEN recipe): unlike a pure
# unary copy, tl.where mixes an i1/i8 mask with f16/f32 data + a scalar. With
# vectorization OPEN the compiler cannot cleanly vectorize the mixed-type
# where and the large-shape path collapses to ~195 GB/s. Closing vectorization
# lets it emit efficient block DMA -> measured 1.8-2.1x on all large shapes
# ([4096,4096] fp16 0.415->0.232ms, [10000,65536] 15.6->8.0ms, bf16 256M
# 7.02->3.40ms) with bit-identical output and no regression on small shapes.
_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, "NO_OPMATH")],
    config=_config,
)
@triton.jit
def masked_fill_kernel(inp, expand_mask, value):
    return tl.where(expand_mask, value, inp)


def masked_fill(inp, mask, value):
    logger.debug("GEMS_KUNLUNXIN MASKED_FILL")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        # Value can be a tensor or a scalar
        value = value.item()
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        # inp is a single-value
        return (
            torch.tensor(value, dtype=inp.dtype, device=inp.device)
            if mask.item()
            else inp.clone()
        )

    out = torch.empty_like(inp, dtype=inp.dtype, device=inp.device)
    if inp.numel() == 0:
        return out

    if inp.is_contiguous() and tuple(mask.shape) == tuple(inp.shape):
        # Common case (mask already matches inp): one flat stride-1 pass, which
        # is what the tuned 1D config accelerates.
        mask = mask.contiguous()
        masked_fill_kernel(inp.view(-1), mask.view(-1), value, out0=out.view(-1))
    else:
        expand_mask = mask.expand(inp.shape)
        masked_fill_kernel.instantiate(inp.ndim)
        masked_fill_kernel(inp, expand_mask, value, out0=out)
    return out


def masked_fill_(inp, mask, value):
    logger.debug("GEMS_KUNLUNXIN MASKED_FILL_")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        # Value can be a tensor or a scalar
        value = value.item()
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        # inp is a single-value
        if mask.item():
            inp[()] = value
        return inp

    if inp.numel() == 0:
        return inp

    if inp.is_contiguous() and tuple(mask.shape) == tuple(inp.shape):
        mask = mask.contiguous()
        masked_fill_kernel(inp.view(-1), mask.view(-1), value, out0=inp.view(-1))
    else:
        expand_mask = mask.expand(inp.shape)
        masked_fill_kernel.instantiate(inp.ndim)
        masked_fill_kernel(inp, expand_mask, value, out0=inp)
    return inp
