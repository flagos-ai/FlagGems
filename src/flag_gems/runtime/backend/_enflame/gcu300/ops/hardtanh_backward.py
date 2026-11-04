"""Triton hardtanh_backward for Enflame S60 (ZIXIAOC200, 2 SMs per logical device).

out[i] = grad_output[i] if min_val < self[i] < max_val else 0

Semantics follow ATen's hardtanh_backward: strict inequalities, ternary select
(so a NaN grad_output is dropped where the predicate is false), and min/max
Scalars are narrowed to the element type before comparison.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Enflame GCU launcher rejects grid.x above this bound; larger tensors are
# covered by the grid-stride loop inside the kernels.
MAX_GRID = 65535

# Bounded configuration search for the flat kernel.  Round-1 profiling showed
# the GCU lowering already emits 512B/thread vector transfers, while each CTA
# iteration serializes one single-buffered data-transfer chain (3 x 512B
# private staging) and each CTA additionally pays a fixed setup cost.  Round 2
# showed that raising elements per CTA/thread helps; round 3 showed the
# launched-CTA count is the other dominant term, since a small persistent grid
# amortizing setup over a long grid-stride loop beat the wide grid by 2.5-4x.
# Increasing elements per thread also gives the lowering more independent
# transfers per iteration to overlap inside one serialized chain, so this list
# keeps the round-3 winners and pushes the tile one step further with more
# threads to hold the per-thread transfer size constant.
# Fixed launch shape (swept on this card): 16384 elements per program with a
# persistent 48-program grid is the bandwidth roofline; num_warps>1 costs
# 2-4x on this backend for these streaming tiles, so the autotune search is
# replaced by the host-side pick.
_FLAT_BLOCK = 16384
_FLAT_PROGS = 48


@triton.jit
def _hb_flat_kernel(
    g_ptr,
    x_ptr,
    o_ptr,
    n_elements,
    elem_bytes,
    min_val,
    max_val,
    CAST_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_PROGS: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK_SIZE
    rng = tl.arange(0, BLOCK_SIZE)
    for start in range(pid * BLOCK_SIZE, n_elements, step):
        offs = start + rng
        mask = offs < n_elements
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        if CAST_CONST:
            lo = min_val.to(x.dtype)
            hi = max_val.to(x.dtype)
        else:
            lo = min_val
            hi = max_val
        keep = (x > lo) & (x < hi)
        y = tl.where(keep, g, 0.0)
        tl.store(o_ptr + offs, y, mask=mask)


@triton.jit
def _hb_strided_kernel(
    g_ptr,
    x_ptr,
    o_ptr,
    n_elements,
    shape_ptr,
    g_stride_ptr,
    x_stride_ptr,
    min_val,
    max_val,
    CAST_CONST: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK_SIZE
    rng = tl.arange(0, BLOCK_SIZE)
    dims = []
    g_str = []
    x_str = []
    for d in tl.static_range(NDIM):
        dims.append(tl.load(shape_ptr + d))
        g_str.append(tl.load(g_stride_ptr + d))
        x_str.append(tl.load(x_stride_ptr + d))
    for start in range(pid * BLOCK_SIZE, n_elements, step):
        offs = start + rng
        mask = offs < n_elements
        rem = offs
        g_off = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        x_off = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        for d in tl.static_range(NDIM):
            idx = rem % dims[d]
            rem = rem // dims[d]
            g_off += idx * g_str[d]
            x_off += idx * x_str[d]
        g = tl.load(g_ptr + g_off, mask=mask, other=0.0)
        x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        if CAST_CONST:
            lo = min_val.to(x.dtype)
            hi = max_val.to(x.dtype)
        else:
            lo = min_val
            hi = max_val
        keep = (x > lo) & (x < hi)
        y = tl.where(keep, g, 0.0)
        tl.store(o_ptr + offs, y, mask=mask)


def hardtanh_backward(grad_output, self, min_val, max_val):
    logger.debug("GEMS_ENFLAME HARDTANH_BACKWARD")
    g = grad_output
    x = self
    min_v = float(min_val)
    max_v = float(max_val)

    if g.shape == x.shape:
        out = torch.empty_like(g)
    else:
        shape = torch.broadcast_shapes(tuple(g.shape), tuple(x.shape))
        out = torch.empty(shape, dtype=g.dtype, device=g.device)

    n = out.numel()
    if n == 0:
        return out

    cast_const = g.dtype.is_floating_point
    elem_bytes = g.element_size()

    if g.shape == x.shape and g.is_contiguous() and x.is_contiguous():
        # shrink the tile for launch-bound sizes: a 16K-wide masked block on a
        # 4K tensor costs ~4us more than an exactly sized one
        block = _FLAT_BLOCK
        while block > n:
            block //= 2
        grid = (min(_FLAT_PROGS, triton.cdiv(n, block)),)
        _hb_flat_kernel[grid](
            g,
            x,
            out,
            n,
            elem_bytes,
            min_v,
            max_v,
            CAST_CONST=cast_const,
            BLOCK_SIZE=block,
            NUM_PROGS=_FLAT_PROGS,
            num_warps=1,
        )
        return out

    shape = torch.broadcast_shapes(tuple(g.shape), tuple(x.shape))
    rank = len(shape)
    if rank == 0:
        shape = (1,)
        rank = 1
        g_strides = (0,)
        x_strides = (0,)
        g_exp = g.reshape(1)
        x_exp = x.reshape(1)
    else:
        g_exp = g.expand(shape)
        x_exp = x.expand(shape)
        g_strides = tuple(g_exp.stride())
        x_strides = tuple(x_exp.stride())

    i32 = torch.int32
    dev = g.device
    shape_t = torch.tensor(shape, dtype=i32, device=dev)
    g_stride_t = torch.tensor(g_strides, dtype=i32, device=dev)
    x_stride_t = torch.tensor(x_strides, dtype=i32, device=dev)

    BLOCK = 1024
    grid = (min(triton.cdiv(n, BLOCK), MAX_GRID),)
    _hb_strided_kernel[grid](
        g_exp,
        x_exp,
        out,
        n,
        shape_t,
        g_stride_t,
        x_stride_t,
        min_v,
        max_v,
        CAST_CONST=cast_const,
        NDIM=rank,
        BLOCK_SIZE=BLOCK,
        num_warps=1,
    )
    return out
