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

"""gcu400 reverse cumulative sum (cumsum backward).

The common implementation splits a reduce axis longer than one tile into a
chunk kernel plus a separate offset kernel, so the whole gradient buffer is
written and read twice.  With the reduce axis innermost the rows are contiguous,
so one program can scan a row end to start while carrying the suffix sum of the
later chunks in a register: the buffer is then read and written exactly once.

The reverse-inclusive scan of a chunk is computed as
``sum(chunk) - cumsum(chunk) + chunk``, which needs no materialised flipped
tile.  Chunk length and length multiples are powers of two, so no mask is
emitted on the fast path.

Slices whose reduce axis is not innermost keep the common kernel, which already
walks the contiguous inner axis for them.  Non power-of-two reduce lengths also
fall back: the masked tail tile of the common kernel aborts the GCU compiler
today (``make_gcuir`` crash in the toolkit), so those lengths are left to the
common path instead of emitting the same construct here.
"""

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


# Tuned on GCU400/L600 with kernel-mode do_bench; the env override only exists
# for the offline sweep that produced this number.
_CHUNK_LEN = _env_int("FLAGGEMS_CSB_CHUNK", 8192)


@libentry()
@triton.jit
def _cumsum_backward_one_tile_kernel(
    Grad,
    GradIn,
    K,
    BK: tl.constexpr,
):
    """Suffix sum of one contiguous row that fits in a single tile."""
    row = tl.program_id(0)
    off = row * K + tl.arange(0, BK)
    x = tl.load(Grad + off).to(tl.float32)
    tl.store(GradIn + off, tl.sum(x, axis=0) - tl.cumsum(x, axis=0) + x)


@libentry()
@triton.jit
def _cumsum_backward_last_kernel(
    Grad,
    GradIn,
    K,
    NUM_CHUNKS: tl.constexpr,
    BK: tl.constexpr,
):
    """Suffix sum of one contiguous row, walked from the last chunk backwards.

    ``carry`` holds the sum of every element after the current chunk, so each
    chunk only needs its own reverse scan.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BK)
    base = row * K
    carry = 0.0
    for step in tl.range(0, NUM_CHUNKS):
        off = base + (NUM_CHUNKS - 1 - step) * BK + cols
        x = tl.load(Grad + off).to(tl.float32)
        tl.store(GradIn + off, tl.sum(x, axis=0) - tl.cumsum(x, axis=0) + x + carry)
        carry += tl.sum(x, axis=0)


def _is_pow2(value):
    return value > 0 and (value & (value - 1)) == 0


def cumsum_backward(grad_output, input, dim=None):
    """Reverse cumulative sum of ``grad_output`` along ``dim``."""
    logger.debug("GEMS_ENFLAME CUMSUM_BACKWARD")

    ndim = grad_output.ndim
    if ndim == 0:
        return grad_output.to(input.dtype).clone()

    if dim is None:
        dim = ndim - 1
    dim = int(dim) % ndim

    reduce_len = input.shape[dim]
    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= input.shape[i]
    outer_size = 1
    for i in range(dim):
        outer_size *= input.shape[i]

    def _common():
        # Imported lazily: the backend loads before `flag_gems.ops` is bound.
        from flag_gems.ops.cumsum_backward import (
            cumsum_backward as _common_cumsum_backward,
        )

        return _common_cumsum_backward(grad_output, input, dim)

    if (
        inner_size != 1
        or reduce_len <= 1
        or input.numel() == 0
        or not _is_pow2(reduce_len)
    ):
        return _common()
    if grad_output.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return _common()

    grad = grad_output.contiguous()
    grad_in = torch.empty(
        (outer_size, reduce_len), dtype=input.dtype, device=grad.device
    )
    chunk = min(_CHUNK_LEN, reduce_len)
    num_chunks = reduce_len // chunk
    grid = (outer_size,)
    if num_chunks == 1:
        _cumsum_backward_one_tile_kernel[grid](grad, grad_in, reduce_len, BK=chunk)
    else:
        _cumsum_backward_last_kernel[grid](
            grad, grad_in, reduce_len, NUM_CHUNKS=num_chunks, BK=chunk
        )
    return grad_in.view(input.shape)


__all__ = ["cumsum_backward"]
