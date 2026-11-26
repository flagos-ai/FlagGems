"""TLE implementation of aten::_add_relu.Tensor.

    out[i] = max(0, self[i] + alpha * other[i])

The pinned FlagGems reference computes ``result = x + y; tl.where(result > 0, result, 0)``
with alpha fixed at 1.

The relu is evaluated with the element-wise *buffer* ops (``tle.dsa.add`` / ``tle.dsa.max``)
instead of ``to_tensor`` -> ``tl.add`` -> ``tl.where`` -> ``to_buffer``.  The round-4 profile
measured the tensor-domain chain as the binding pipe on the 16-bit workloads: the Triton add
and compare are canonicalized to f32 on this backend, so bfloat16 pays ``vcast -> vadd ->
vcast -> vcmp -> vsel`` plus the full-tile UB->UB copy that ``tle.dsa.to_buffer`` emits, i.e.
7 vector passes against fp16's 4 for the identical byte volume (AIV 51.0 us vs 28.0 us with
the same MTE2/MTE3 times), and the round-4 analysis left the avoidability of that UB->UB copy
open.  The buffer ops operate in place on the staged UB buffers, so the chain is one add plus
one max against a preallocated zero tile, with no promote, no demote and no extra copy.
``max(x + y, 0)`` is the same function as the reference's guarded select - both map a
non-positive value to zero - and it is the pointer dtype end to end.  The buffer ops take no
scalar operand, so the alpha != 1 case keeps the tensor-domain path.

Structure: 1-D vec-explicit TLE kernel whose grid is capped at the number of vector cores, so
each program owns many contiguous tiles and walks them with a grid stride.  The tile body
(GM -> UB staging, vector compute, UB -> GM store) is wrapped in ``tle.dsa.pipeline`` so the
MTE2 load of the next tile overlaps the vector work of the current one.  ``tle.dsa.copy`` has
no mask, so the tile start is clamped to ``n - BLOCK`` (the last tile is shifted back and
recomputes identical values) and inputs smaller than one tile go to a masked register kernel.
"""

import logging

import torch
import torch_npu  # noqa: F401  (must be imported before triton on Ascend)
import triton
import triton.experimental.tle as tle
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _add_relu_ub_kernel(
    x_ptr,
    y_ptr,
    o_ptr,
    n_elements,
    n_tiles,
    alpha,
    MUL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cnt = (n_tiles - pid + nprog - 1) // nprog

    dt = x_ptr.dtype.element_ty
    ar = tl.arange(0, BLOCK)
    xb = tle.dsa.alloc([BLOCK], dtype=dt, mem_addr_space=tle.dsa.ascend.UB)
    yb = tle.dsa.alloc([BLOCK], dtype=dt, mem_addr_space=tle.dsa.ascend.UB)
    ab = tle.dsa.alloc([BLOCK], dtype=dt, mem_addr_space=tle.dsa.ascend.UB)
    rb = tle.dsa.alloc([BLOCK], dtype=dt, mem_addr_space=tle.dsa.ascend.UB)
    zb = tle.dsa.to_buffer(tl.zeros([BLOCK], dt), space=tle.dsa.ascend.UB)

    for i in tle.dsa.pipeline(0, cnt, 1, num_stages=2):
        tid = pid + i * nprog
        start = tl.minimum(tid.to(tl.int64) * BLOCK, n_elements.to(tl.int64) - BLOCK)
        offs = start + ar

        tle.dsa.copy(x_ptr + offs, xb, [BLOCK])
        tle.dsa.copy(y_ptr + offs, yb, [BLOCK])

        if MUL:
            xf = tle.dsa.to_tensor(xb)
            yf = tle.dsa.to_tensor(yb)
            v = xf + yf * alpha.to(dt)
            tle.dsa.copy(
                tle.dsa.to_buffer(tl.where(v > 0, v, 0.0), space=tle.dsa.ascend.UB),
                o_ptr + offs,
                [BLOCK],
            )
        else:
            tle.dsa.add(xb, yb, ab)
            tle.dsa.max(ab, zb, rb)
            tle.dsa.copy(rb, o_ptr + offs, [BLOCK])


@triton.jit
def _add_relu_small_kernel(
    x_ptr,
    y_ptr,
    o_ptr,
    n_elements,
    alpha,
    MUL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)
    m = idx < n_elements
    dt = x_ptr.dtype.element_ty
    xf = tl.load(x_ptr + idx, mask=m, other=0.0)
    yf = tl.load(y_ptr + idx, mask=m, other=0.0)
    if MUL:
        v = xf + yf * alpha.to(dt)
    else:
        v = xf + yf
    tl.store(o_ptr + idx, tl.where(v > 0, v, 0.0), mask=m)


@triton.jit
def _add_relu_strided_kernel(
    x_ptr,
    y_ptr,
    o_ptr,
    n_elements,
    alpha,
    SHAPE: tl.constexpr,
    SX: tl.constexpr,
    SY: tl.constexpr,
    MUL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Out-of-contract broadcast fallback: per-element offset decomposition."""
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n_elements

    rem = idx
    ox = tl.zeros([BLOCK], tl.int64)
    oy = tl.zeros([BLOCK], tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        c = rem % SHAPE[d]
        rem = rem // SHAPE[d]
        ox += c * SX[d]
        oy += c * SY[d]

    dt = x_ptr.dtype.element_ty
    xv = tl.load(x_ptr + ox, mask=mask, other=0.0).to(tl.float32)
    yv = tl.load(y_ptr + oy, mask=mask, other=0.0).to(tl.float32)
    if MUL:
        v = xv + yv * alpha
    else:
        v = xv + yv
    tl.store(o_ptr + idx, tl.where(v > 0, v, 0.0).to(dt), mask=mask)


# 8 KB per tile: the buffer-op chain holds five tile-sized UB regions (xb, yb, ab, rb, zb)
# and the pipeline's multi-buffer duplicates the staged loads, so a 16 KB tile overflows the
# 192 KB UB (measured: 280608 B required vs 196608 B available, bfloat16).  Half the tile
# keeps the same total vector-pass count at a UB footprint that fits.
_TILE_BYTES = 8192
_VEC_CORES = None


def _vector_cores():
    """Vector-core count of the current device, or None when unavailable."""
    global _VEC_CORES
    if _VEC_CORES is None:
        n = None
        try:
            dev = torch.npu.current_device()
        except Exception:
            dev = None
        if dev is not None:
            try:
                from triton.backends.ascend.driver import driver as _drv

                n = _drv.active.utils.get_device_properties(dev).get("num_vectorcore")
            except Exception:
                n = None
            if not n:
                try:
                    n = torch_npu.npu.get_device_properties(dev).vector_core_num
                except Exception:
                    n = None
            if not n:
                try:
                    n = torch_npu.npu.npu_config.get_device_limit(dev).get(
                        "vector_core_num"
                    )
                except Exception:
                    n = None
        _VEC_CORES = int(n) if n else 0
    return _VEC_CORES or None


def _alpha_value(alpha):
    if isinstance(alpha, (int, float)):
        return float(alpha)
    if isinstance(alpha, torch.Tensor) and alpha.numel() == 1:
        return float(alpha.item())
    return 1.0


_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def _add_relu(self, other, *, alpha=1):
    """``_add_relu`` backed by the TLE kernel."""
    logger.debug("GEMS _ADD_RELU")
    if self.dtype not in _SUPPORTED_DTYPES:
        from flag_gems.ops._add_relu import _add_relu as _generic

        return _generic(self, other)

    x = self
    y = other
    shape = torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    n = out.numel()
    if n == 0:
        return out

    a = _alpha_value(alpha)
    mul = a != 1.0
    block = _TILE_BYTES // out.element_size()

    if (
        x.shape == shape
        and y.shape == shape
        and x.is_contiguous()
        and y.is_contiguous()
    ):
        if n >= block:
            n_tiles = (n + block - 1) // block
            vc = _vector_cores()
            progs = min(n_tiles, vc) if vc else n_tiles
            _add_relu_ub_kernel[(progs,)](
                x, y, out, n, n_tiles, a, MUL=mul, BLOCK=block
            )
        else:
            small = 1 << max(n - 1, 0).bit_length()
            _add_relu_small_kernel[(1,)](x, y, out, n, a, MUL=mul, BLOCK=small)
        return out

    # Generic broadcast path (not exercised by the pinned correctness set).
    grid = ((n + block - 1) // block,)
    nd = len(shape)
    xs = (1,) * (nd - x.dim()) + tuple(x.stride())
    ys = (1,) * (nd - y.dim()) + tuple(y.stride())
    xsh = (1,) * (nd - x.dim()) + tuple(x.shape)
    ysh = (1,) * (nd - y.dim()) + tuple(y.shape)
    xsl = list(xs)
    ysl = list(ys)
    for d in range(nd):
        if xsh[d] == 1:
            xsl[d] = 0
        if ysh[d] == 1:
            ysl[d] = 0
    _add_relu_strided_kernel[grid](
        x,
        y,
        out,
        n,
        a,
        SHAPE=tuple(shape),
        SX=tuple(xsl),
        SY=tuple(ysl),
        MUL=mul,
        BLOCK=block,
    )
    return out
