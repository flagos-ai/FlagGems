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
def fmax_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, GRID_SIZE, num_stages=NUM_STAGES):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = tl.maximum(x, y)
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
        raise TypeError("fmax does not support complex dtypes.")
    compute_dtype = torch.int8 if out_dtype == torch.bool else out_dtype
    a_c = a_b.to(compute_dtype).contiguous()
    b_c = b_b.to(compute_dtype).contiguous()
    return a_c, b_c, out_dtype, compute_dtype


def _launch(a_c, b_c, out_c):
    n_elements = out_c.numel()
    if n_elements == 0:
        return
    # Tuned on GCU400 (maxnum kernel body, official bench kernel mode
    # do_bench w1000/r100 median, interleaved A/B probes):
    # - 16-bit peak at BLOCK=65536 (1G bf16 2.96 TB/s vs 2.10 at 8192;
    #   131072 ties but its 2x131072x2B=512KB DSM budget crosses the 448KB
    #   48-way concurrency line), fp32 keeps 32768 (256KB);
    # - num_warps=2 (4 is +8-13% everywhere); NUM_STAGES 1/2/4 within noise;
    # - n<=4096 (e.g. core (64,64)) uses BS=max(512, npo2(n//2)) so several
    #   CTAs share the launch-latency-bound work: fp32 4096 7.39->3.34us.
    if n_elements <= 4096:
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
        fmax_kernel[grid](
            a_c,
            b_c,
            out_c,
            n_elements,
            BLOCK_SIZE=block_size,
            GRID_SIZE=min(_GRID_SIZE, num_tile),
            NUM_STAGES=num_stages,
            num_warps=2,
        )


def fmax(a, b):
    logger.debug("GEMS_ENFLAME FMAX")
    a_c, b_c, out_dtype, compute_dtype = _prepare_inputs(a, b, out=None)
    out_shape = a_c.shape
    if compute_dtype == out_dtype:
        out = torch.empty(out_shape, dtype=out_dtype, device=a_c.device)
        out_c = out
    else:
        out = torch.empty(out_shape, dtype=out_dtype, device=a_c.device)
        out_c = torch.empty(out_shape, dtype=compute_dtype, device=a_c.device)
    _launch(a_c, b_c, out_c)
    if out_c.dtype != out.dtype:
        out.copy_(out_c.to(out_dtype))
    return out


def fmax_out(a, b, out):
    logger.debug("GEMS_ENFLAME FMAX_OUT")
    a_c, b_c, out_dtype, compute_dtype = _prepare_inputs(a, b, out=out)
    out_shape = a_c.shape
    if tuple(out.shape) != tuple(out_shape):
        raise ValueError("fmax_out: output shape mismatch")
    if compute_dtype == out_dtype and out.is_contiguous():
        _launch(a_c, b_c, out)
        return out
    out_c = torch.empty(out_shape, dtype=compute_dtype, device=a_c.device)
    _launch(a_c, b_c, out_c)
    if out_c.dtype != out.dtype:
        out.copy_(out_c.to(out_dtype))
    else:
        out.copy_(out_c)
    return out
