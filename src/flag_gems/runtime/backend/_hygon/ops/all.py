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

from flag_gems import runtime
from flag_gems.ops.all import all_dim as generic_all_dim
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# torch.all: Tests if all elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if all elements in input evaluate to non-zero value
# In triton function, test if all elements in input evaluate to non-zero value is ok.


@triton.jit
def reduce_all(a, b):
    return a and b


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("all"), key=["M", "N"])
@triton.jit
def all_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=1.0)
        _all = _all and (a != 0)
    all = tl.reduce(_all, axis=1, combine_fn=reduce_all)
    tl.store(out, all[:, None], row_mask)


@libentry()
@triton.jit
def all_kernel_1(
    inp,
    mid,
    n_elements,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < n_elements
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0)
    all_val = tl.reduce(inp_val != 0, axis=0, combine_fn=reduce_all)
    mid_ptr = mid + pid
    tl.store(mid_ptr, all_val)


@libentry()
@triton.jit
def all_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=1).to(tl.int1)
    all_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_all)
    tl.store(out, all_val)


def all(inp):
    logger.debug("GEMS_HYGON ALL")
    n_elements = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)

    with torch_device_fn.device(inp.device):
        all_kernel_1[(mid_size, 1)](inp, mid, n_elements, mid_size, block_size)
        all_kernel_2[(1, 1)](mid, out, mid_size, block_mid)

    return out


@libentry()
@triton.jit
def _all_f32_rows(
    inp,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    values = tl.load(inp + offsets, mask=mask, other=1.0)
    # IEEE FP32: only +/-0 are false. Integer truth testing also preserves
    # subnormal truth on targets that flush floating-point comparisons to zero.
    truth = (values.to(tl.int32, bitcast=True) & 0x7FFFFFFF) != 0
    result = tl.reduce(truth, axis=1, combine_fn=reduce_all)
    tl.store(out + rows, result, mask=rows < M)


def _can_use_all_dim_fast_path(inp, dim, keepdim):
    # Vendor selection is handled by the normal backend export mechanism.
    # Inspect only metadata; never sample values or infer the reduction result.
    if type(dim) is not int or type(keepdim) is not bool:
        return False
    if inp.dtype != torch.float32 or inp.layout != torch.strided:
        return False
    if not 1 <= inp.ndim <= 8 or dim not in (-1, inp.ndim - 1):
        return False
    if not inp.is_contiguous() or inp.is_conj() or inp.is_neg():
        return False
    n = inp.shape[-1]
    count = inp.numel()
    if type(n) is not int or type(count) is not int:
        return False
    return 128 <= n <= 1024 and 1 <= count // n <= 65536 and count <= 2147483647


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_HYGON ALL_DIM")
    if not _can_use_all_dim_fast_path(inp, dim, keepdim):
        return generic_all_dim(inp, dim=dim, keepdim=keepdim)

    n = inp.shape[-1]
    m = inp.numel() // n
    shape = tuple(inp.shape[:-1]) + ((1,) if keepdim else ())
    out = torch.empty(shape, dtype=torch.bool, device=inp.device)
    with torch_device_fn.device(inp.device):
        _all_f32_rows[(triton.cdiv(m, 4),)](
            inp,
            out,
            m,
            n,
            BLOCK_M=4,
            BLOCK_N=triton.next_power_of_2(n),
            num_warps=4,
        )
    return out


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_HYGON ALL_DIMS")

    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        all_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
