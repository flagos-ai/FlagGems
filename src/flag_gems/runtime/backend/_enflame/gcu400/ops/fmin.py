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
NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def fmin_kernel(x_ptr, y_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        y = tl.load(y_ptr + off, mask=mask).to(tl.float32)
        out = tl.minimum(x, y)
        tl.store(out_ptr + off, out, mask=mask)


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
        dev = torch.device("cuda")
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
    N = out_c.numel()
    if N == 0:
        return
    BLOCK = 8192
    grid = min((N + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(a_c.device):
        fmin_kernel[(grid,)](a_c, b_c, out_c, N, BLOCK=BLOCK, num_warps=4)


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
