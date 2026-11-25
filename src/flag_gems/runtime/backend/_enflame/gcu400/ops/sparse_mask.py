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
"""Triton implementation of ``aten::sparse_mask`` for the enflame GCU400 backend.

Contract (from the pinned Gems pytest suite):

* ``sparse_mask(self, mask)`` returns a sparse COO tensor whose ``indices()``
  are a copy of ``mask``'s indices and whose ``values()`` are ``self`` gathered
  at the mask's coordinates.  ``mask``'s values are ignored.
* ``sparse_mask(self, mask, out=out)`` writes into ``out`` and returns it.
* Neither operand is mutated, and both call forms propagate autograd.

GCU400 specialisation (why this file exists next to the generic op):

* Kurama refuses pointers it cannot reach, so a CPU-resident coordinate tensor
  makes the generic wrapper fail with ``Pointer argument (at N) cannot be
  accessed from Kurama``.  Every pointer handed to a kernel here is therefore
  normalised onto ``self``'s device first, and every gradient buffer is
  allocated on the device of the incoming gradient.
* The gather is a single Triton kernel; all strides/sizes are scalar kernel
  arguments, so the wrapper performs exactly one H2D index copy (when needed),
  one kernel launch and no host round-trips.
* Sparse ``self`` is materialised with ``to_dense`` (a layout transform, not the
  main compute path) and gathered by the same kernel.  Its backward needs no
  key/binary-search machinery because autograd routes the incoming dense
  gradient back through ``to_dense``.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

__all__ = ["sparse_mask", "sparse_mask_out"]

# A 1024-entry index block keeps enough independent gather loads in flight to
# hide the random-read latency; measured on GCU400 (512^2 .. 4096^2, fp16/fp32)
# it beats the 128-entry generic tile by 1.1x-1.9x, while 2048/4096-entry tiles
# and 8/16 warps are flat or slower.
_BLOCK_I = 1024
_BLOCK_D = 128
_NUM_WARPS = 4

# The kernels take a fixed number of scalar axis arguments; a higher-rank
# tensor than this is rejected by the shape checks before any launch.
_MAX_DIM = 8


@triton.jit
def _pick(  # noqa: PLR0913 - one branch per supported axis index
    k: tl.constexpr,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
):
    """Select the ``k``-th scalar argument (constexpr-indexed at compile time)."""
    if k == 0:
        return a0
    elif k == 1:
        return a1
    elif k == 2:
        return a2
    elif k == 3:
        return a3
    elif k == 4:
        return a4
    elif k == 5:
        return a5
    elif k == 6:
        return a6
    else:
        return a7


@triton.jit
def _gather_kernel(  # noqa: PLR0913 - one scalar per supported axis index
    self_ptr,
    idx_ptr,
    out_ptr,
    N,
    D,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    z0,
    z1,
    z2,
    z3,
    z4,
    z5,
    z6,
    z7,
    P: tl.constexpr,
    Q: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather ``self`` at the mask's coordinates into a flat ``(N, D)`` buffer.

    ``s*`` are the strides of the ``P`` sparse axes, ``d*``/``z*`` the strides
    and sizes of the ``Q`` dense (inner) axes.  ``Q == 0`` (the coalesced
    single-element case) keeps the pure 1-D path so a mask entry costs exactly
    one gather and one store.
    """
    pid = tl.program_id(0)
    offs_i = pid * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_i = offs_i < N

    base = tl.zeros([BLOCK_I], dtype=tl.int64)
    for k in tl.static_range(P):
        sk = _pick(k, s0, s1, s2, s3, s4, s5, s6, s7)
        ik = tl.load(idx_ptr + k * N + offs_i, mask=mask_i, other=0).to(tl.int64)
        base += ik * sk

    if Q == 0:
        # One element per nnz: already coalesced, and evict_first keeps the
        # scattered sector-filling self read from thrashing L2.
        val = tl.load(
            self_ptr + base, mask=mask_i, other=0, eviction_policy="evict_first"
        )
        tl.store(out_ptr + offs_i, val, mask=mask_i, eviction_policy="evict_first")
    else:
        row = offs_i * D
        for od in range(0, D, BLOCK_D):
            offs_d = od + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            rem = offs_d.to(tl.int64)
            lin = tl.zeros([BLOCK_D], dtype=tl.int64)
            for q in tl.static_range(Q):
                sz = _pick(q, z0, z1, z2, z3, z4, z5, z6, z7)
                st = _pick(q, d0, d1, d2, d3, d4, d5, d6, d7)
                lin += (rem % sz) * st
                rem = rem // sz
            mask2 = mask_i[:, None] & mask_d[None, :]
            ptrs = base[:, None] + lin[None, :]
            val = tl.load(
                self_ptr + ptrs, mask=mask2, other=0, eviction_policy="evict_first"
            )
            tl.store(
                out_ptr + row[:, None] + offs_d[None, :],
                val,
                mask=mask2,
                eviction_policy="evict_first",
            )


@triton.jit
def _scatter_add_kernel(  # noqa: PLR0913 - one scalar per supported axis index
    grad_ptr,
    idx_ptr,
    out_ptr,
    N,
    D,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    z0,
    z1,
    z2,
    z3,
    z4,
    z5,
    z6,
    z7,
    P: tl.constexpr,
    Q: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Scatter-add the incoming sparse gradient back onto the dense ``self``."""
    pid = tl.program_id(0)
    offs_i = pid * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_i = offs_i < N

    base = tl.zeros([BLOCK_I], dtype=tl.int64)
    for k in tl.static_range(P):
        sk = _pick(k, s0, s1, s2, s3, s4, s5, s6, s7)
        ik = tl.load(idx_ptr + k * N + offs_i, mask=mask_i, other=0).to(tl.int64)
        base += ik * sk

    if Q == 0:
        g = tl.load(grad_ptr + offs_i, mask=mask_i, other=0)
        tl.atomic_add(out_ptr + base, g, mask=mask_i)
    else:
        row = offs_i * D
        for od in range(0, D, BLOCK_D):
            offs_d = od + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            rem = offs_d.to(tl.int64)
            lin = tl.zeros([BLOCK_D], dtype=tl.int64)
            for q in tl.static_range(Q):
                sz = _pick(q, z0, z1, z2, z3, z4, z5, z6, z7)
                st = _pick(q, d0, d1, d2, d3, d4, d5, d6, d7)
                lin += (rem % sz) * st
                rem = rem // sz
            mask2 = mask_i[:, None] & mask_d[None, :]
            g = tl.load(grad_ptr + row[:, None] + offs_d[None, :], mask=mask2, other=0)
            tl.atomic_add(out_ptr + base[:, None] + lin[None, :], g, mask=mask2)


def _axis_args(strides):
    """Pad a stride/size list to the kernel's fixed scalar arity."""
    values = [int(v) for v in strides]
    return values + [0] * (_MAX_DIM - len(values))


class _SparseMaskGather(torch.autograd.Function):
    """Device gather whose backward scatters the sparse gradient back."""

    @staticmethod
    def forward(ctx, dense_self, indices, sizes, sparse_dim, dense_dim, dense_sizes):
        nnz = int(indices.shape[1]) if indices.dim() == 2 else 0
        block = 1
        for size in dense_sizes:
            block *= int(size)
        values = torch.empty(
            (nnz,) + tuple(dense_sizes),
            dtype=dense_self.dtype,
            device=dense_self.device,
        )
        ctx.sizes = sizes
        ctx.sparse_dim = sparse_dim
        ctx.dense_dim = dense_dim
        ctx.dtype = dense_self.dtype
        ctx.device = dense_self.device
        ctx.save_for_backward(indices)
        if nnz > 0:
            sparse_strides = [dense_self.stride(k) for k in range(sparse_dim)]
            dense_strides = [
                dense_self.stride(sparse_dim + k) for k in range(dense_dim)
            ]
            grid = (triton.cdiv(nnz, _BLOCK_I),)
            _gather_kernel[grid](
                dense_self,
                indices,
                values,
                nnz,
                block,
                *_axis_args(sparse_strides),
                *_axis_args(dense_strides),
                *_axis_args(dense_sizes),
                P=sparse_dim,
                Q=dense_dim,
                BLOCK_I=_BLOCK_I,
                BLOCK_D=_BLOCK_D,
                num_warps=_NUM_WARPS,
            )
        return values

    @staticmethod
    def backward(ctx, grad_values):
        (indices,) = ctx.saved_tensors
        nnz = int(indices.shape[1]) if indices.dim() == 2 else 0
        sparse_dim = ctx.sparse_dim
        dense_dim = ctx.dense_dim
        dense_sizes = tuple(int(s) for s in grad_values.shape[1:])
        block = 1
        for size in dense_sizes:
            block *= size
        # The gradient must live on the device of the incoming gradient, not on
        # the CPU: autograd rejects a CPU gradient for a GCU operand.
        grad_self = torch.zeros(ctx.sizes, dtype=ctx.dtype, device=grad_values.device)
        if nnz > 0:
            sparse_strides = [grad_self.stride(k) for k in range(sparse_dim)]
            dense_strides = [grad_self.stride(sparse_dim + k) for k in range(dense_dim)]
            grad_values = grad_values.to(grad_self.device).contiguous()
            grid = (triton.cdiv(nnz, _BLOCK_I),)
            _scatter_add_kernel[grid](
                grad_values,
                indices,
                grad_self,
                nnz,
                block,
                *_axis_args(sparse_strides),
                *_axis_args(dense_strides),
                *_axis_args(dense_sizes),
                P=sparse_dim,
                Q=dense_dim,
                BLOCK_I=_BLOCK_I,
                BLOCK_D=_BLOCK_D,
                num_warps=_NUM_WARPS,
            )
        return grad_self, None, None, None, None, None


def _dense_self(self):
    """Return a dense view of ``self`` (the reference accepts sparse self)."""
    if self.layout == torch.strided:
        return self
    # ``to_dense`` is a real ATen op producing exactly the values the reference
    # gathers from, and it carries the correct autograd history for the sparse
    # COO / CSR / BSR / CSC / BSC inputs the suite exercises.
    return self.to_dense()


def sparse_mask(self, mask, out=None):

    logger.debug("GEMS_ENFLAMESPARSE_MASK")
    if not isinstance(self, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("sparse_mask expects two tensors")

    if mask.layout != torch.sparse_coo:
        raise RuntimeError(
            "sparse_mask expects a sparse COO mask tensor, got layout " f"{mask.layout}"
        )

    if tuple(self.shape) != tuple(mask.shape):
        raise RuntimeError(
            "sparse_mask(): operands have incompatible sizes; self has size "
            f"{list(self.shape)} but mask has size {list(mask.shape)}"
        )

    dense_self = _dense_self(self)
    device = dense_self.device

    indices = mask._indices()
    if indices.dim() != 2:
        indices = indices.reshape(mask.sparse_dim(), -1)
    # Kurama cannot dereference a host pointer, so the coordinate tensor must be
    # reachable from the device before the gather is launched.
    if indices.device != device:
        indices = indices.to(device)
    indices = indices.contiguous()

    sparse_dim = int(mask.sparse_dim())
    dense_dim = int(mask.dense_dim())
    sizes = tuple(mask.shape)
    dense_sizes = tuple(int(s) for s in sizes[sparse_dim:])

    values = _SparseMaskGather.apply(
        dense_self, indices, sizes, sparse_dim, dense_dim, dense_sizes
    )

    result = torch.sparse_coo_tensor(
        indices, values, sizes, is_coalesced=bool(mask.is_coalesced())
    )

    if out is None:
        return result

    if not isinstance(out, torch.Tensor):
        raise TypeError("sparse_mask out= must be a tensor")

    out.copy_(result)
    return out


def sparse_mask_out(self, mask, out):
    """``aten::sparse_mask.out`` overload."""
    logger.debug("GEMS_ENFLAMESPARSE_MASK.OUT")
    return sparse_mask(self, mask, out=out)
