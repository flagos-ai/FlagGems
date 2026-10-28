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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _soft_margin_loss_elementwise(x, y):
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    z = -xf * yf
    absz = tl.abs(z)
    return tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-absz))


@libentry()
@triton.jit
def kernel_1(
    x_ptr,
    y_ptr,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    reduction: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offset < M
        xf = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
        yf = tl.load(y_ptr + offset, mask=mask, other=0).to(tl.float32)
    else:
        xf = tl.load(x_ptr + offset).to(tl.float32)
        yf = tl.load(y_ptr + offset).to(tl.float32)

    z = -xf * yf
    absz = tl.abs(z)
    vals = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-absz))
    if NEED_MASK:
        # Zero out contributions from out-of-bounds elements
        # (soft_margin_loss(0,0) = log(2) != 0, so masking is required)
        vals = tl.where(mask, vals, 0.0)

    # Reduction.MEAN.value: 1, Reduction.SUM.value: 2
    if reduction == 1:
        sum_val = tl.sum(vals) / M
    else:
        sum_val = tl.sum(vals)

    tl.store(mid + pid, sum_val)


@libentry()
@triton.jit
def kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    # Loop-accumulate into a [BLOCK_MID] fp32 tile, then a SINGLE tl.sum.
    # BLOCK_MID is capped at 8192 (XPU tl.sum only reduces the first 8192
    # lanes correctly), so when mid_size > 8192 we must stride over it in
    # chunks; the element-wise `acc +=` accumulation across iterations is
    # correct on XPU (verified) and a single final reduce stays within 8192.
    acc = tl.zeros([BLOCK_MID], dtype=tl.float32)
    n_iter = tl.cdiv(mid_size, BLOCK_MID)
    for i in range(n_iter):
        offset = i * BLOCK_MID + tl.arange(0, BLOCK_MID)
        mask = offset < mid_size
        acc += tl.load(mid + offset, mask=mask, other=0).to(tl.float32)
    tl.store(out, tl.sum(acc))


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == "none":
            return 0
        if r == "mean":
            return 1
        if r == "sum":
            return 2
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def soft_margin_loss(input: torch.Tensor, target: torch.Tensor, reduction="mean"):
    logger.debug("GEMS_KUNLUNXIN SOFT_MARGIN_LOSS")
    red = _normalize_reduction(reduction)

    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()

    n_elements = input.numel()

    if red == 0:
        # reduction = 'none': use pointwise kernel (no atomic_add, no masked load issues)
        if n_elements == 0:
            return torch.empty_like(input)
        return _soft_margin_loss_elementwise(input, target)

    # reduction = 'sum' (red==2) or 'mean' (red==1)
    if n_elements == 0:
        if red == 2:
            return torch.zeros((), device=input.device, dtype=input.dtype)
        else:
            return torch.full((), float("nan"), device=input.device, dtype=input.dtype)

    # XPU tl.sum/large-tile rules (HARNESS_SUMMARY 2.5):
    #  - without buffer_size_limit, tl.sum is only complete for BLOCK <= 8192;
    #  - with buffer_size_limit=2048, BLOCK == 32768 is complete.
    # So for N % 32768 == 0 we use BLOCK=32768 + buffer_size_limit=2048 (fewer
    # programs, smaller `mid`, less kernel_2 work; measured e.g. [10000,65536]
    # fp32 24.8 -> 17.6ms, 2**28 fp16 10.1 -> 7.0ms vs the 8192 config) and
    # for everything else BLOCK = next_pow2(N) capped at 8192 with a masked
    # tail (same semantics as the original kernel). For n <= 8192 a single
    # block (mid_size==1) skips kernel_2 entirely (single kernel, best on
    # tiny shapes).
    # Use empty_strided (NOT torch.empty) for the scratch tensors: flag_gems
    # registers `empty.memory_format`, and on XPU that gems empty kernel costs
    # ~96ms per call (one zero-write launch + vendor allocator), which used to
    # dominate the whole op. empty_strided is not registered -> native
    # allocator, ~10us. This was the single biggest perf blocker (measured:
    # benchmark harness showed ~190ms/call due to two torch.empty calls).
    if n_elements >= 32768 and n_elements % 32768 == 0:
        block_size = 32768
        buffer_limit = 2048
    else:
        block_size = min(triton.next_power_of_2(n_elements), 8192)
        buffer_limit = None
    if n_elements % block_size == 0:
        need_mask = False
    else:
        need_mask = True
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = min(triton.next_power_of_2(mid_size), 8192)

    mid = torch.empty_strided(
        (mid_size,), (1,), dtype=torch.float32, device=input.device
    )
    out = torch.empty_strided((), (), dtype=torch.float32, device=input.device)

    with torch_device_fn.device(input.device):
        kw1 = {}
        kw2 = {}
        if buffer_limit is not None:
            kw1["buffer_size_limit"] = buffer_limit
            kw2["buffer_size_limit"] = buffer_limit
        kernel_1[(mid_size, 1, 1)](
            input,
            target,
            mid,
            n_elements,
            block_size,
            red,
            need_mask,
            **kw1,
        )
        if mid_size == 1:
            result = mid.reshape([]).to(dtype=input.dtype)
            return result
        kernel_2[(1, 1, 1)](
            mid,
            out,
            mid_size,
            block_mid,
            **kw2,
        )

    return out.to(dtype=input.dtype)


def soft_margin_loss_out(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction="mean",
    out: torch.Tensor = None,
):
    logger.debug("GEMS_KUNLUNXIN SOFT_MARGIN_LOSS_OUT")
    result = soft_margin_loss(input, target, reduction)
    if out is None:
        return result
    out.copy_(result)
    return out
