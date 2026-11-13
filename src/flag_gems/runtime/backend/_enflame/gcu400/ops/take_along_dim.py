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

logger = logging.getLogger(__name__)


@triton.jit
def take_along_dim_kernel(
    inp_ptr,
    idx_ptr,
    out_ptr,
    R,
    D_in,
    D_out,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    nprog = tl.num_programs(axis=0)
    lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    n_full = (N // BLOCK_SIZE) * BLOCK_SIZE

    # Persistent grid-stride over full tiles: addresses are recomputed from
    # the induction variable each iteration (no loop-carried pointers).
    for base in range(pid * BLOCK_SIZE, n_full, nprog * BLOCK_SIZE):
        offs = base + lane
        # Decompose the flat output offset into (row, col) over (R, D_out).
        r = offs // D_out

        idx = tl.load(idx_ptr + offs).to(tl.int64)
        # take_along_dim does not wrap negative indices, but wrapping here is a
        # harmless no-op for the valid (already non-negative) case and keeps the
        # kernel address math well defined.
        idx = tl.where(idx < 0, idx + D_in, idx)

        valid = (idx >= 0) & (idx < D_in)
        tl.device_assert(valid, "take_along_dim(): index out of range")

        in_off = r * D_in + idx
        vals = tl.load(inp_ptr + in_off, mask=valid, other=0)
        tl.store(out_ptr + offs, vals)

    # Trailing partial tile: handled once by program 0, with a mask.
    if n_full < N:
        if pid == 0:
            offs = n_full + lane
            mask = offs < N
            r = offs // D_out

            idx = tl.load(idx_ptr + offs, mask=mask, other=0).to(tl.int64)
            idx = tl.where(idx < 0, idx + D_in, idx)

            valid = (idx >= 0) & (idx < D_in)
            tl.device_assert(valid | (~mask), "take_along_dim(): index out of range")

            in_off = r * D_in + idx
            vals = tl.load(inp_ptr + in_off, mask=mask & valid, other=0)
            tl.store(out_ptr + offs, vals, mask=mask)


_BLOCK_SIZE = 1024
_MAX_GRID = 65535
_MIN_PROGRAMS = 256
_ITERS = 16


def _launch(inp2d, idx2d, out2d):
    R, D_in = inp2d.shape
    _, D_out = idx2d.shape
    N = out2d.numel()
    if N == 0:
        return
    # One tile per program up to the cap; beyond it, each program strides
    # over multiple tiles so grid.x never exceeds the 65535 hardware limit.
    nblocks = triton.cdiv(N, _BLOCK_SIZE)
    nprog = max(min(nblocks, _MIN_PROGRAMS), triton.cdiv(nblocks, _ITERS))
    grid = (min(nprog, _MAX_GRID),)
    with torch_device_fn.device(inp2d.device):
        take_along_dim_kernel[grid](
            inp2d, idx2d, out2d, R, D_in, D_out, N, BLOCK_SIZE=_BLOCK_SIZE
        )


def _prepare(input, indices, dim):
    """Reduce take_along_dim to a 2D (R, D) gather.

    Returns (inp2d, idx2d, out_shape, restore) where ``restore`` maps the flat
    (R, D_out) output back to the final tensor layout.
    """
    if dim is None:
        # Flatten the input; the result has the shape of ``indices``.
        inp2d = input.reshape(1, -1).contiguous()
        idx2d = indices.reshape(1, -1).contiguous()
        # dim=None flattens both operands; the result is 1-D.
        out_shape = (indices.numel(),)

        def restore_flat(out2d):
            return out2d.reshape(out_shape)

        return inp2d, idx2d, out_shape, restore_flat

    if input.ndim != indices.ndim:
        raise RuntimeError(
            "take_along_dim: input and indices should have the same "
            f"number of dimensions, but got {input.ndim} dimensions for input, "
            f"and {indices.ndim} dimensions for indices"
        )

    dim = dim % input.ndim

    # Broadcast every non-``dim`` axis; the ``dim`` axis keeps each tensor's own
    # size (input provides the value space, indices the output extent).
    batch_shape = []
    for d in range(input.ndim):
        if d == dim:
            batch_shape.append(1)
        else:
            batch_shape.append(max(input.shape[d], indices.shape[d]))

    inp_shape = list(batch_shape)
    inp_shape[dim] = input.shape[dim]
    idx_shape = list(batch_shape)
    idx_shape[dim] = indices.shape[dim]

    inp_b = torch.broadcast_to(input, inp_shape)
    idx_b = torch.broadcast_to(indices, idx_shape)

    # Move ``dim`` to the last axis and collapse the batch dims into rows.
    inp_p = inp_b.movedim(dim, -1).contiguous()
    idx_p = idx_b.movedim(dim, -1).contiguous()

    D_in = input.shape[dim]
    D_out = indices.shape[dim]
    inp2d = inp_p.reshape(-1, D_in)
    idx2d = idx_p.reshape(-1, D_out)

    # Final output shape: broadcasted batch dims with the ``dim`` axis == D_out.
    out_shape = list(batch_shape)
    out_shape[dim] = D_out
    out_shape = tuple(out_shape)
    permuted_shape = tuple(idx_p.shape)

    def restore_nd(out2d):
        return out2d.reshape(permuted_shape).movedim(-1, dim).contiguous()

    return inp2d, idx2d, out_shape, restore_nd


def take_along_dim(input, indices, dim=None):
    logger.debug("GEMS TAKE_ALONG_DIM")
    if input.device != indices.device:
        raise RuntimeError("input and indices must be on the same device")
    inp2d, idx2d, out_shape, restore = _prepare(input, indices, dim)
    out2d = torch.empty(idx2d.shape, device=input.device, dtype=input.dtype)
    _launch(inp2d, idx2d, out2d)
    return restore(out2d)


def take_along_dim_out(input, indices, dim=None, *, out):
    logger.debug("GEMS TAKE_ALONG_DIM_OUT")
    if not (input.device == indices.device == out.device):
        raise RuntimeError("input, indices and out must be on the same device")
    if out.dtype != input.dtype:
        raise RuntimeError(f"out must have dtype {input.dtype}, but got {out.dtype}")
    inp2d, idx2d, out_shape, restore = _prepare(input, indices, dim)
    out2d = torch.empty(idx2d.shape, device=input.device, dtype=input.dtype)
    _launch(inp2d, idx2d, out2d)
    result = restore(out2d)
    if tuple(out.shape) != tuple(result.shape):
        out.resize_(result.shape)
    out.copy_(result)
    return out
