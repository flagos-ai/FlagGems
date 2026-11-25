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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24
GRID_CAP = NUM_SIPS * 2


def _torch_dtype_to_triton(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float64:
        return tl.float64
    raise ValueError(f"Unsupported dtype for Triton conversion: {dtype}")


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _hypot_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pids = tl.num_programs(axis=0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        offsets = block_id * BLOCK_SIZE + arange
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        xf = x.to(COMPUTE_DTYPE)
        yf = y.to(COMPUTE_DTYPE)
        res = tl.sqrt(xf * xf + yf * yf)
        tl.store(out_ptr + offsets, res.to(OUT_DTYPE), mask=mask)


def _hypot_launch_config(n_elements, out_dtype):
    # Small sizes: sqrt-E law block B = np2(32 * sqrt(n)) keeps one mono wave.
    # Large sizes: 2R1W stream sweet spots measured on L600 (gcu400, 48 SP):
    #   n <= 2^24  -> B=32768/nw2 (all dtypes);  n > 2^24 half-precision ->
    #   B=65536/nw2; fp32 stays at B=32768/nw2 (65536 pushes per-program DSM
    #   past the 448KB dual-program line). fp64 keeps a DSM-safe small block.
    if n_elements <= 65536:
        block = triton.next_power_of_2(32 * math.isqrt(n_elements))
        return max(block, 32), 4
    if out_dtype == torch.float64:
        return 8192, 2
    if out_dtype in (torch.float16, torch.bfloat16) and n_elements > 16777216:
        return 65536, 2
    return 32768, 2


def _infer_hypot_out_dtype(a: torch.Tensor, b: torch.Tensor) -> torch.dtype:
    if a.is_complex() or b.is_complex():
        raise NotImplementedError(
            "Complex dtypes are not supported for hypot in this implementation."
        )
    if a.is_floating_point() or b.is_floating_point():
        return torch.result_type(a, b)
    return torch.get_default_dtype()


def _launch_hypot_kernel(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor):
    n_elements = out.numel()
    if n_elements == 0:
        return

    out_dtype = out.dtype
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"Unsupported output dtype for hypot: {out_dtype}")

    block_size, num_warps = _hypot_launch_config(n_elements, out_dtype)
    grid = (min(triton.cdiv(n_elements, block_size), GRID_CAP),)

    with torch_device_fn.device(out.device):
        _hypot_kernel[grid](
            x,
            y,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            OUT_DTYPE=_torch_dtype_to_triton(out_dtype),
            COMPUTE_DTYPE=tl.float64 if out_dtype == torch.float64 else tl.float32,
            num_warps=num_warps,
        )


def hypot(a: torch.Tensor, b: torch.Tensor):
    logger.debug("GEMS_ENFLAME HYPOT")
    if (
        a.dtype == b.dtype
        and a.is_floating_point()
        and a.device == b.device
        and a.shape == b.shape
        and a.is_contiguous()
        and b.is_contiguous()
    ):
        out = torch.empty_like(a)
        _launch_hypot_kernel(a, b, out)
        return out

    out_dtype = _infer_hypot_out_dtype(a, b)
    device = a.device
    if b.device != device:
        raise ValueError("Input tensors must be on the same device")

    out_shape = torch.broadcast_shapes(a.shape, b.shape)
    out = torch.empty(out_shape, dtype=out_dtype, device=device)

    x = a.expand(out_shape).contiguous()
    y = b.expand(out_shape).contiguous()

    _launch_hypot_kernel(x, y, out)
    return out


def hypot_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME HYPOT_OUT")
    if out.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"Unsupported out dtype for hypot_out: {out.dtype}")

    if (
        a.dtype == b.dtype == out.dtype
        and a.shape == b.shape == out.shape
        and a.is_contiguous()
        and b.is_contiguous()
        and out.is_contiguous()
    ):
        _launch_hypot_kernel(a, b, out)
        return out

    target_shape = out.shape
    x = a.expand(target_shape).contiguous()
    y = b.expand(target_shape).contiguous()

    if out.is_contiguous():
        _launch_hypot_kernel(x, y, out)
    else:
        out_tmp = torch.empty(target_shape, dtype=out.dtype, device=out.device)
        _launch_hypot_kernel(x, y, out_tmp)
        out.copy_(out_tmp)
    return out
