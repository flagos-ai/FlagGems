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

logger = logging.getLogger(__name__)

# Fixed Grid on GCU400 (48 CTAs); see ops/deg2rad.py for the grid.x limit
# rationale.
_GRID_SIZE = 48


@libentry()
@triton.jit
def fmin_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # tl.minimum lowers to minnumf on this backend (verified against
    # torch.fmin: one-NaN -> other operand, both-NaN -> NaN), so it already
    # carries fmin NaN semantics. The former .to(tl.float32) upcast was both
    # a measurable tax on fp16 (kernel-only 16M/1G probes: +4-6% over the
    # 2R1W floor) and silently lossy for int64 inputs beyond 2^24; native
    # dtype compute sits at the 2R1W memory floor for every float dtype
    # (fp16/bf16/fp32, 16.7M/1G, interleaved A/B).
    pid = tl.program_id(axis=0)
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, GRID_SIZE, num_stages=NUM_STAGES):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = tl.minimum(x, y)
        tl.store(out_ptr + offsets, out, mask=mask)


def _to_tensor(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        t = x
        if device is not None and t.device != device:
            t = t.to(device)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t
    return torch.tensor(x, device=device, dtype=dtype)


def _prepare_inputs(a, b, out=None):
    dev = None
    if isinstance(out, torch.Tensor):
        dev = out.device
    else:
        if isinstance(a, torch.Tensor):
            dev = a.device
        if isinstance(b, torch.Tensor):
            dev = b.device if dev is None else dev
    if dev is None:
        dev = torch.device(f"gcu:{torch.gcu.current_device()}")
    a = _to_tensor(a, device=dev)
    b = _to_tensor(b, device=dev)
    a_b, b_b = torch.broadcast_tensors(a, b)
    out_dtype = torch.result_type(a_b, b_b)
    if out_dtype.is_complex:
        raise TypeError("fmin does not support complex dtypes.")
    compute_dtype = torch.int8 if out_dtype == torch.bool else out_dtype
    a_c = a_b.to(compute_dtype).contiguous()
    b_c = b_b.to(compute_dtype).contiguous()
    return a_c, b_c, out_dtype, compute_dtype


def _launch_fmin(a_c, b_c, out_c):
    n_elements = out_c.numel()
    if n_elements == 0:
        return
    # Tuned on GCU400 (minnum kernel body, official bench kernel mode
    # do_bench w1000/r100 median, interleaved A/B probes):
    # - 16-bit peak at BLOCK=65536 (1G fp16/bf16 2.8-2.9 TB/s vs 2.4-2.5 at
    #   8192), fp32 keeps 32768 (256KB DSM per 2R stream);
    # - num_warps=2 (4 is +10-16% everywhere); grid capped at 48 CTAs
    #   (96/192 neutral-to-worse, fp16 1G grid96 +8%);
    # - n<=4096 (e.g. core (64,64)): 16-bit favors a single wide CTA
    #   (BS=max(1024, npo2(n)): fp16 3.44 vs 3.75us for npo2(n//2)),
    #   4-byte favors two (BS=max(512, npo2(n//2)): fp32 3.50 vs 3.57us).
    if n_elements <= 4096:
        if a_c.element_size() <= 2:
            block_size = max(1024, triton.next_power_of_2(n_elements))
        else:
            block_size = max(512, triton.next_power_of_2(n_elements // 2))
        num_stages = 1
    else:
        block_size = 65536 if a_c.element_size() <= 2 else 32768
        if n_elements < block_size:
            block_size = max(1024, triton.next_power_of_2(n_elements))
        num_stages = 4
    num_tile = triton.cdiv(n_elements, block_size)
    grid = (min(_GRID_SIZE, num_tile), 1, 1)
    with torch_device_fn.device(out_c.device.index):
        fmin_kernel[grid](
            a_c,
            b_c,
            out_c,
            n_elements,
            BLOCK_SIZE=block_size,
            GRID_SIZE=min(_GRID_SIZE, num_tile),
            NUM_STAGES=num_stages,
            num_warps=2,
        )


def fmin(a, b):
    logger.debug("GEMS_ENFLAME FMIN")
    a_c, b_c, out_dtype, compute_dtype = _prepare_inputs(a, b, out=None)
    out_shape = a_c.shape
    if compute_dtype == out_dtype:
        out = torch.empty(out_shape, dtype=out_dtype, device=a_c.device)
        out_c = out
    else:
        out = torch.empty(out_shape, dtype=out_dtype, device=a_c.device)
        out_c = torch.empty(out_shape, dtype=compute_dtype, device=a_c.device)
    _launch_fmin(a_c, b_c, out_c)
    if out_c.dtype != out.dtype:
        out.copy_(out_c.to(out_dtype))
    return out


def fmin_out(a, b, out):
    logger.debug("GEMS_ENFLAME FMIN_OUT")
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a Tensor")
    a_c, b_c, out_dtype, compute_dtype = _prepare_inputs(a, b, out=out)
    expected_shape = a_c.shape
    if out.device != a_c.device:
        raise ValueError("out tensor must be on the same device as inputs.")
    if out.dtype != out_dtype:
        raise TypeError(f"out tensor has dtype {out.dtype}, expected {out_dtype}.")
    if tuple(out.shape) != tuple(expected_shape):
        raise ValueError(
            f"out tensor has shape {tuple(out.shape)}, expected {tuple(expected_shape)} after broadcasting."
        )
    if compute_dtype == out_dtype and out.is_contiguous():
        out_c = out
    else:
        out_c = torch.empty(expected_shape, dtype=compute_dtype, device=out.device)
    _launch_fmin(a_c, b_c, out_c)
    if out_c is not out:
        if out_c.dtype != out.dtype:
            out.copy_(out_c.to(out.dtype))
        else:
            if out.is_contiguous():
                out.copy_(out_c)
            else:
                out.view_as(out.contiguous()).copy_(out_c)
    return out
