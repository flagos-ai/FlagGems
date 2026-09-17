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

_BIG_PATH_MIN_ELEMS = 131072

_config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, "DEFAULT")],
    config=_config_,
)
@triton.jit
def _softshrink_big_func(x, lambd):
    x32 = x.to(tl.float32)

    gt = x32 > lambd
    lt = x32 < -lambd
    res32 = tl.where(gt, x32 - lambd, tl.where(lt, x32 + lambd, 0.0))

    # Propagate NaN: if x is NaN, keep it (matches torch.nn.functional.softshrink)
    x_bits = x32.to(tl.int32, bitcast=True)
    is_nan = (x_bits & 0x7FFFFFFF) > 0x7F800000
    res32 = tl.where(is_nan, x32, res32)

    return res32.to(x.dtype)


@triton.jit
def _softshrink_small_kernel(
    x_ptr, out_ptr, n_elements, lambd, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x32 = x.to(tl.float32)

    threshold = lambd  # scalar float32

    gt = x32 > threshold
    lt = x32 < -threshold
    res32 = tl.where(gt, x32 - threshold, tl.where(lt, x32 + threshold, 0.0))

    # Propagate NaN: if x is NaN, keep it
    x_bits = x32.to(tl.int32, bitcast=True)
    is_nan = (x_bits & 0x7FFFFFFF) > 0x7F800000
    res32 = tl.where(is_nan, x32, res32)

    res = res32.to(x.dtype)
    tl.store(out_ptr + offsets, res, mask=mask)


def _softshrink_small(x: torch.Tensor, out: torch.Tensor, lambd: float):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(x.device):
        _softshrink_small_kernel[grid](
            x,
            out,
            n_elements,
            float(lambd),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )


def _check_supported_dtype(t: torch.Tensor):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"Unsupported dtype {t.dtype}. Supported dtypes are float16, bfloat16, and float32."
        )


def softshrink(input: torch.Tensor, lambd: float = 0.5):
    logger.debug("GEMS_KUNLUNXIN SOFTSHRINK")
    _check_supported_dtype(input)
    if input.numel() == 0:
        return torch.empty_like(input)
    x = input.contiguous()
    if x.numel() < _BIG_PATH_MIN_ELEMS:
        out = torch.empty_like(x)
        _softshrink_small(x, out, lambd)
        return out.reshape_as(input)
    return _softshrink_big_func(x, float(lambd)).reshape_as(input)


def softshrink_out(input: torch.Tensor, lambd: float = 0.5, out: torch.Tensor = None):
    logger.debug("GEMS_KUNLUNXIN SOFTSHRINK_OUT")
    if out is None:
        raise ValueError("Argument 'out' must be provided for softshrink_out.")
    if input.shape != out.shape:
        raise ValueError(
            f"Shape mismatch: input.shape={input.shape}, out.shape={out.shape}"
        )
    if input.dtype != out.dtype:
        raise TypeError(
            f"Dtype mismatch: input.dtype={input.dtype}, out.dtype={out.dtype}"
        )
    _check_supported_dtype(input)
    if input.numel() == 0:
        return out

    x = input.contiguous()
    if out.is_contiguous():
        out_buf = out
    else:
        out_buf = torch.empty_like(out, memory_format=torch.contiguous_format)

    if out_buf.numel() < _BIG_PATH_MIN_ELEMS:
        _softshrink_small(x, out_buf, lambd)
    else:
        _softshrink_big_func(x, float(lambd), out0=out_buf)

    if out_buf.data_ptr() != out.data_ptr():
        out.copy_(out_buf)
    return out
