# Copyright 2026, The FlagOS Contributors.
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

_BLOCK_SIZE = 4096
_MAX_LEAD_RANK = 5

_COMPLEX_DTYPE = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


@libentry()
@triton.jit
def _view_as_complex_copy_flat_kernel(
    src_ptr,
    dst_ptr,
    n_real,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_real
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)


@libentry()
@triton.jit
def _view_as_complex_copy_strided_kernel(
    src_ptr,
    dst_ptr,
    d0,
    d1,
    d2,
    d3,
    d4,
    s0,
    s1,
    s2,
    s3,
    s4,
    n_real,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    j = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = j < n_real

    i = j // 2
    lane = j - i * 2

    if RANK == 2:
        outer = i // d1
        src_base = (i - outer * d1) * s1 + outer * s0
    else:
        lead_dims = [d0, d1, d2, d3, d4]
        lead_strides = [s0, s1, s2, s3, s4]
        rem = i
        src_base = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        for k in tl.static_range(RANK - 1, -1, -1):
            idx = rem % lead_dims[k]
            rem = rem // lead_dims[k]
            src_base += idx * lead_strides[k]

    value = tl.load(src_ptr + src_base + lane, mask=mask)
    tl.store(dst_ptr + j, value, mask=mask)


def view_as_complex_copy(self: torch.Tensor) -> torch.Tensor:
    r"""Copy a real tensor whose last dimension has size 2 into a complex tensor."""
    logger.debug("GEMS_KUNLUNXIN VIEW_AS_COMPLEX_COPY")

    if self.dim() == 0:
        raise RuntimeError("Input tensor must have one or more dimensions")
    if self.size(-1) != 2:
        raise RuntimeError("Tensor must have a last dimension of size 2")
    complex_dtype = _COMPLEX_DTYPE.get(self.dtype)
    if complex_dtype is None:
        raise RuntimeError(
            "view_as_complex is only supported for half, float and double "
            "tensors, but got a tensor of scalar type: "
            f"{str(self.dtype).replace('torch.', '').capitalize()}"
        )
    if self.stride(-1) != 1:
        raise RuntimeError("Tensor must have a last dimension with stride 1")

    if self.is_conj():
        self = self.resolve_conj()
    if self.is_neg():
        self = self.resolve_neg()

    leading = tuple(self.shape[:-1])
    out = torch.empty(leading, dtype=complex_dtype, device=self.device)
    n_complex = out.numel()
    if n_complex == 0:
        return out

    out_real = torch.view_as_real(out)
    n_real = 2 * n_complex
    with torch_device_fn.device(self.device):
        grid = lambda meta: (triton.cdiv(n_real, meta["BLOCK_SIZE"]),)
        if self.is_contiguous():
            _view_as_complex_copy_flat_kernel[grid](
                self,
                out_real,
                n_real,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
        else:
            rank = self.dim() - 1
            if rank > _MAX_LEAD_RANK:
                raise RuntimeError(
                    f"view_as_complex_copy supports at most {_MAX_LEAD_RANK + 1} "
                    f"dimensions, got {self.dim()}"
                )
            lead_dims = list(self.shape[:-1]) + [1] * (_MAX_LEAD_RANK - rank)
            lead_strides = list(self.stride()[:-1]) + [0] * (_MAX_LEAD_RANK - rank)
            _view_as_complex_copy_strided_kernel[grid](
                self,
                out_real,
                *lead_dims,
                *lead_strides,
                n_real,
                RANK=rank,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
    return out
