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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def flip_kernel(
    inp,
    out,
    meta,
    n_elements,
    NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    linear = offsets
    inp_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    # Runtime metadata avoids the constexpr-tuple indexing limitation in the
    # Triton version shipped with the Kunlunxin stack.  Keeping all address
    # arithmetic positive also avoids the large-tensor fault caused by the old
    # negative-stride StridedBuffer path.
    for dim in tl.static_range(NDIM - 1, -1, -1):
        size = tl.load(meta + dim)
        stride = tl.load(meta + NDIM + dim)
        should_flip = tl.load(meta + 2 * NDIM + dim)
        coord = linear % size
        linear //= size
        coord = tl.where(should_flip != 0, size - 1 - coord, coord)
        inp_offsets += coord * stride

    value = tl.load(inp + inp_offsets, mask=mask)
    tl.store(out + offsets, value, mask=mask)


def _flip_real(A: torch.Tensor, flip_dims) -> torch.Tensor:
    out = torch.empty_like(A, memory_format=torch.contiguous_format)
    if A.numel() == 0:
        return out

    ndim = A.ndim
    meta = torch.tensor(
        tuple(A.shape) + tuple(A.stride()) + tuple(flip_dims),
        dtype=torch.int64,
        device=A.device,
    )
    block_size = 256
    grid = (triton.cdiv(A.numel(), block_size),)
    with torch_device_fn.device(A.device):
        flip_kernel[grid](
            A,
            out,
            meta,
            A.numel(),
            NDIM=ndim,
            BLOCK_SIZE=block_size,
        )
    return out


def flip(A: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN FLIP")
    flip_dims = [False] * A.ndim
    for dim in dims:
        if dim < -A.ndim or dim >= A.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-A.ndim}, {A.ndim - 1}], but got {dim})"
            )
        dim %= A.ndim
        if flip_dims[dim]:
            raise RuntimeError(f"dim {dim} appears multiple times in the list of dims")
        flip_dims[dim] = True

    if A.ndim == 0 or A.numel() <= 1 or not any(flip_dims):
        return A.clone()

    if A.is_complex():
        real_view = torch.view_as_real(A.resolve_conj())
        flipped = _flip_real(real_view, tuple(flip_dims) + (False,))
        return torch.view_as_complex(flipped)
    return _flip_real(A, tuple(flip_dims))
