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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

NUM_SIPS = 24
BLOCK = 8192


@libentry()
@triton.jit(do_not_specialize=["N_total", "lambd"])
def softshrink_kernel(x_ptr, out_ptr, N_total, lambd, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    threshold = lambd
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask, other=0)
        x32 = x.to(tl.float32)
        gt = x32 > threshold
        lt = x32 < -threshold
        res32 = tl.where(gt, x32 - threshold, tl.where(lt, x32 + threshold, 0.0))
        res32 = tl.where(x32 != x32, x32, res32)
        tl.store(out_ptr + off, res32.to(x.dtype), mask=mask)


def _check_supported_dtype(t: torch.Tensor):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"Unsupported dtype {t.dtype}. Supported dtypes are float16, bfloat16, and float32."
        )


def _grid(n_elements):
    return min(triton.cdiv(n_elements, BLOCK), NUM_SIPS * 2)


def _launch_softshrink_kernel(x: torch.Tensor, out: torch.Tensor, lambd: float):
    n_elements = x.numel()
    if n_elements == 0:
        return
    with torch_device_fn.device(x.device):
        softshrink_kernel[(_grid(n_elements),)](
            x, out, n_elements, float(lambd), BLOCK_SIZE=BLOCK, num_warps=4
        )


def softshrink(input: torch.Tensor, lambd: float = 0.5):
    _check_supported_dtype(input)
    x = input.contiguous()
    out = torch.empty_like(x)
    _launch_softshrink_kernel(x, out, lambd)
    return out.reshape_as(input)


def softshrink_out(input: torch.Tensor, lambd: float = 0.5, out: torch.Tensor = None):
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

    x = input.contiguous()
    if out.is_contiguous():
        out_buf = out
    else:
        out_buf = torch.empty_like(out, memory_format=torch.contiguous_format)

    _launch_softshrink_kernel(x, out_buf, lambd)

    if out_buf.data_ptr() != out.data_ptr():
        out.copy_(out_buf)
    return out
