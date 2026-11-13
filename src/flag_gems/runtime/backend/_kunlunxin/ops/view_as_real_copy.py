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

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 4096
_MAX_LEAD_RANK = 5

_REAL_DTYPE = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


@libentry()
@triton.jit
def _view_as_real_copy_flat_kernel(
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
def _view_as_real_copy_strided_kernel(
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


def view_as_real_copy(self: torch.Tensor) -> torch.Tensor:
    r"""Copy a complex tensor into a real tensor whose last dimension has size 2."""
    logger.debug("GEMS_KUNLUNXIN VIEW_AS_REAL_COPY")

    real_dtype = _REAL_DTYPE.get(self.dtype)
    if real_dtype is None:
        raise RuntimeError(
            f"view_as_real is only supported for complex tensors, not for "
            f"{str(self.dtype).replace('torch.', '')} tensors"
        )

    if self.is_conj():
        self = self.resolve_conj()
    if self.is_neg():
        self = self.resolve_neg()

    out = torch.empty(tuple(self.shape) + (2,), dtype=real_dtype, device=self.device)
    n_complex = self.numel()
    if n_complex == 0:
        return out

    # The trailing size-1 axis makes the last dimension contiguous, so the
    # complex tensor can be reinterpreted as its real pairs without a copy.
    src_real = self.as_strided(
        tuple(self.shape) + (1,), tuple(self.stride()) + (1,)
    ).view(real_dtype)
    n_real = 2 * n_complex
    contiguous = src_real.is_contiguous()
    if not contiguous and self.dim() > _MAX_LEAD_RANK:
        raise RuntimeError(
            f"view_as_real_copy supports at most {_MAX_LEAD_RANK} "
            f"dimensions, got {self.dim()}"
        )
    with torch_device_fn.device(self.device):
        # The operator is a pure move from the real view of `self` into a
        # contiguous buffer of the same shape, which is what `tle_copy`
        # expresses: one tile transfer when both sides are contiguous,
        # otherwise rows of a run with independent source and destination row
        # strides. A pointwise kernel moves the same bytes several times
        # slower, on the flat and on the strided path alike.
        if tle_copy(src_real, out):
            return out
        grid = lambda meta: (triton.cdiv(n_real, meta["BLOCK_SIZE"]),)
        if contiguous:
            _view_as_real_copy_flat_kernel[grid](
                src_real,
                out,
                n_real,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
        else:
            rank = self.dim()
            dims = list(src_real.shape[:-1]) + [1] * (_MAX_LEAD_RANK - rank)
            strides = list(src_real.stride()[:-1]) + [0] * (_MAX_LEAD_RANK - rank)
            _view_as_real_copy_strided_kernel[grid](
                src_real,
                out,
                *dims,
                *strides,
                n_real,
                RANK=rank,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
    return out
