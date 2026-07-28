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
from flag_gems.utils import libentry, libtuner

TOTAL_CORE_NUM = torch_device_fn.get_device_properties().multi_processor_count

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@libtuner(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_stages=1, num_warps=1),
    ],
    key=["N"],
    strategy=["log"],
)
@triton.jit(do_not_specialize=["value_scalar"])
def fill_scalar_kernel(
    out_ptr,
    N,
    value_scalar,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    block_start = block_start.to(tl.int64)
    for block_start_offset in range(block_start, N, step):
        offset = block_start_offset + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + offset, value_scalar, mask=offset < N)


@libentry()
@libtuner(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 65536}, num_stages=1, num_warps=1),
    ],
    key=["N"],
)
@triton.jit
def fill_tensor_kernel(
    out_ptr,
    N,
    value_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    block_start = block_start.to(tl.int64)
    for block_start_offset in range(block_start, N, step):
        offset = block_start_offset + tl.arange(0, BLOCK_SIZE)
        value_scalar = tl.load(value_ptr)  # load the value from the tensor.
        tl.store(out_ptr + offset, value_scalar, mask=offset < N)


# Complex tensors are filled through their real view (shape [..., 2]) so that
# triton only ever sees real dtypes. The buffer is laid out as the interleaved
# pattern [real, imag, real, imag, ...]; we select the channel by offset parity.
@libentry()
@libtuner(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_stages=1, num_warps=1),
    ],
    key=["N"],
    strategy=["log"],
)
@triton.jit(do_not_specialize=["real_scalar", "imag_scalar"])
def fill_complex_scalar_kernel(
    out_ptr,
    N,
    real_scalar,
    imag_scalar,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    block_start = block_start.to(tl.int64)
    for block_start_offset in range(block_start, N, step):
        offset = block_start_offset + tl.arange(0, BLOCK_SIZE)
        is_imag = (offset % 2) == 1
        value_scalar = tl.where(is_imag, imag_scalar, real_scalar)
        tl.store(out_ptr + offset, value_scalar, mask=offset < N)


@libentry()
@libtuner(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 65536}, num_stages=1, num_warps=1),
    ],
    key=["N"],
)
@triton.jit
def fill_complex_tensor_kernel(
    out_ptr,
    N,
    value_ptr,  # real view of the 0-dim complex value, layout [real, imag]
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    block_start = block_start.to(tl.int64)
    real_scalar = tl.load(value_ptr)
    imag_scalar = tl.load(value_ptr + 1)
    for block_start_offset in range(block_start, N, step):
        offset = block_start_offset + tl.arange(0, BLOCK_SIZE)
        is_imag = (offset % 2) == 1
        value_scalar = tl.where(is_imag, imag_scalar, real_scalar)
        tl.store(out_ptr + offset, value_scalar, mask=offset < N)


def _fill_complex_scalar(out, value):
    # `out` is a complex tensor filled in place via its real view.
    cval = complex(value)
    out_real = torch.view_as_real(out)
    N = out_real.numel()  # == 2 * out.numel()
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)
    with torch_device_fn.device(out.device):
        fill_complex_scalar_kernel[grid_fn](
            out_real, N, float(cval.real), float(cval.imag)
        )
    return out


def _fill_complex_tensor(out, value):
    # `out` is a complex tensor filled in place from a 0-dim complex value tensor.
    value = value.to(device=out.device, dtype=out.dtype)
    out_real = torch.view_as_real(out)
    value_real = torch.view_as_real(value)
    N = out_real.numel()  # == 2 * out.numel()
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)
    with torch_device_fn.device(out.device):
        fill_complex_tensor_kernel[grid_fn](out_real, N, value_real)
    return out


def fill_tensor(input, value):
    logger.debug("GEMS_TSINGMICRO FILL TENSOR")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    if 0 in out.shape:
        return out
    if out.is_complex():
        return _fill_complex_tensor(out, value)
    N = out.numel()
    # grid = triton.cdiv(N, BLOCK_SIZE)
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)

    with torch_device_fn.device(input.device):
        fill_tensor_kernel[grid_fn](out, N, value)
    return out


def fill_scalar(input, value):
    logger.debug("GEMS_TSINGMICRO FILL SCALAR")
    if 0 in input.shape:
        return input
    out = torch.empty_like(input)
    if out.is_complex():
        return _fill_complex_scalar(out, value)
    N = out.numel()
    # grid = triton.cdiv(N, BLOCK_SIZE)
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)

    with torch_device_fn.device(input.device):
        fill_scalar_kernel[grid_fn](out, N, value)
    return out


def fill_scalar_out(input, value, *, out=None):
    logger.debug("GEMS_TSINGMICRO FILL SCALAR_OUT")
    if out is None:
        return fill_scalar(input, value)
    if 0 in out.shape:
        return out
    if out.is_complex():
        return _fill_complex_scalar(out, value)
    N = out.numel()
    # grid = triton.cdiv(N, BLOCK_SIZE)
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)

    with torch_device_fn.device(input.device):
        fill_scalar_kernel[grid_fn](out, N, value)
    return out


def fill_tensor_out(input, value, *, out=None):
    logger.debug("GEMS_TSINGMICRO FILL_TENSOR_OUT")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    if out is None:
        return fill_tensor(input, value)
    if 0 in out.shape:
        return out
    if out.is_complex():
        return _fill_complex_tensor(out, value)
    N = out.numel()
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)

    with torch_device_fn.device(input.device):
        fill_tensor_kernel[grid_fn](out, N, value)
    return out


def fill_tensor_(self, value):
    logger.debug("GEMS_TSINGMICRO FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    if 0 in self.shape:
        return self
    if self.is_complex():
        return _fill_complex_tensor(self, value)
    N = self.numel()
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)

    with torch_device_fn.device(self.device):
        fill_tensor_kernel[grid_fn](self, N, value)
    return self


def fill_scalar_(self, value):
    logger.debug("GEMS_TSINGMICRO FILL_SCALAR_")
    if 0 in self.shape:
        return self
    if self.is_complex():
        return _fill_complex_scalar(self, value)
    N = self.numel()
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)

    with torch_device_fn.device(self.device):
        fill_scalar_kernel[grid_fn](self, N, value)
    return self
