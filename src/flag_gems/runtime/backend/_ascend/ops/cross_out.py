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
#
# Ascend implementation imported from the tuned operator batch
# host entry points below mirror the generic
# flag_gems names so the backend registrar can pick them up.

"""Cross product: modulo-free contiguous flat kernel, cached raw launch, ND stride
fallback and out-alias protection.

Why this flat kernel looks the way it does (measured on Ascend910B4 / FlagTree 0.6.0,
1M vectors per tensor):

* A pure copy of the same footprint costs 0.068 ms and four *misaligned* shifted
  loads cost 0.067 ms, so contiguity and alignment are not the limiter.
* `lane % 3` alone costs 2.171 ms: integer modulo on a per-lane tile is lowered to
  scalar-ish code. The four `tl.gather` shuffles of the previous formulation cost
  ~0.41 ms each (1.70 ms total).
* So the phase is derived from an fp32 reciprocal plus a truncating cast (exact
  for lane < 2**23) and the permutation becomes eight plain shifted loads. The
  correct cross product then runs at ~0.075 ms, at the copy floor.
"""

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_ONE_THIRD = 0.3333333432674408


@triton.jit
def _cross_flat_kernel(
    a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr
):
    lane = tl.arange(0, BLOCK)
    base = tl.program_id(0) * STEP + lane
    live = lane < STEP
    ph = lane - 3 * (lane.to(tl.float32) * 0.3333333432674408).to(tl.int32)
    lo2 = live & (base >= 2) & (base < n_elem)
    lo1 = live & (base >= 1) & (base < n_elem)
    hi1 = live & (base + 1 < n_elem)
    hi2 = live & (base + 2 < n_elem)
    a_m2 = tl.load(a_ptr + base - 2, mask=lo2, other=0.0)
    a_m1 = tl.load(a_ptr + base - 1, mask=lo1, other=0.0)
    a_p1 = tl.load(a_ptr + base + 1, mask=hi1, other=0.0)
    a_p2 = tl.load(a_ptr + base + 2, mask=hi2, other=0.0)
    b_m2 = tl.load(b_ptr + base - 2, mask=lo2, other=0.0)
    b_m1 = tl.load(b_ptr + base - 1, mask=lo1, other=0.0)
    b_p1 = tl.load(b_ptr + base + 1, mask=hi1, other=0.0)
    b_p2 = tl.load(b_ptr + base + 2, mask=hi2, other=0.0)
    t0 = a_p1 * b_p2 - a_p2 * b_p1
    t1 = a_p1 * b_m1 - a_m1 * b_p1
    t2 = a_m2 * b_m1 - a_m1 * b_m2
    tl.store(
        o_ptr + base,
        tl.where(ph == 0, t0, tl.where(ph == 1, t1, t2)),
        mask=live & (base < n_elem),
    )


@triton.jit
def _cross_general_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    vectors,
    SHAPE: tl.constexpr,
    AXIS: tl.constexpr,
    SA: tl.constexpr,
    SB: tl.constexpr,
    SO: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK_I + tl.arange(0, BLOCK_I)
    m = index < vectors
    rem = index.to(tl.int64)
    oa = tl.zeros((BLOCK_I,), tl.int64)
    ob = tl.zeros((BLOCK_I,), tl.int64)
    oo = tl.zeros((BLOCK_I,), tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        if axis != AXIS:
            coordinate = rem % SHAPE[axis]
            rem = rem // SHAPE[axis]
            oa += coordinate * SA[axis]
            ob += coordinate * SB[axis]
            oo += coordinate * SO[axis]
    pa, pb = (a_ptr + oa, b_ptr + ob)
    a0 = tl.load(pa, mask=m, other=0.0)
    a1 = tl.load(pa + SA[AXIS], mask=m, other=0.0)
    a2 = tl.load(pa + 2 * SA[AXIS], mask=m, other=0.0)
    b0 = tl.load(pb, mask=m, other=0.0)
    b1 = tl.load(pb + SB[AXIS], mask=m, other=0.0)
    b2 = tl.load(pb + 2 * SB[AXIS], mask=m, other=0.0)
    po = o_ptr + oo
    tl.store(po, a1 * b2 - a2 * b1, mask=m)
    tl.store(po + SO[AXIS], a2 * b0 - a0 * b2, mask=m)
    tl.store(po + 2 * SO[AXIS], a0 * b1 - a1 * b0, mask=m)


def _pick_block(n_elem, esize):
    if n_elem <= 131072:
        return 1024
    return 4096


def _default_dim(input, other):
    broadcast_shape = torch.broadcast_shapes(input.shape, other.shape)
    for axis, size in enumerate(broadcast_shape):
        if size == 3:
            return axis
    raise RuntimeError(
        "no dimension of size 3 in input; cross requires a dimension of length 3"
    )


def _broadcast_strides(t, out_shape):
    """Strides of ``t`` expressed against ``out_shape`` (0 where broadcast)."""
    nd = len(out_shape)
    lead = nd - t.dim()
    res = [0] * nd
    for i in range(t.dim()):
        d = lead + i
        if t.shape[i] == 1 and out_shape[d] != 1:
            res[d] = 0
        else:
            res[d] = t.stride(i)
    return res


_PLANS = {}
_RAW_STREAM = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)


def _current_stream(device):
    if _RAW_STREAM is not None:
        try:
            return _RAW_STREAM(device)
        except TypeError:
            return _RAW_STREAM()
    return torch.npu.current_stream(device).npu_stream


def _launch_flat(a, b, out, n, pa, pb, po):
    key = (a.device.index, a.dtype, b.dtype, n)
    entry = _PLANS.get(key)
    if entry is None:
        block = _pick_block(n, out.element_size())
        step = block // 3 * 3
        grid = (n + step - 1) // step
        launch = None
        if not (pa | pb | po) & 15:
            try:
                compiled = _cross_flat_kernel.warmup(
                    a, b, out, n, BLOCK=block, STEP=step, grid=(1,)
                )
                launcher = compiled.run
                md = launcher.metadata
                if not (
                    getattr(launcher, "compile_only", False)
                    or getattr(launcher, "enable_msprof_register_tensor", False)
                    or getattr(md, "debug_enabled", False)
                    or getattr(md, "debug", False)
                ):
                    launch = (
                        launcher.launch,
                        compiled.function,
                        compiled.packed_metadata,
                    )
            except Exception:
                pass
        entry = (grid, block, step, launch)
        _PLANS[key] = entry
    launch = entry[3]
    if launch and (not (pa | pb | po) & 15):
        launch[0](
            entry[0],
            1,
            1,
            _current_stream(a.device.index),
            launch[1],
            launch[2],
            None,
            None,
            None,
            pa,
            pb,
            po,
            n,
            entry[1],
            entry[2],
        )
    else:
        _cross_flat_kernel[entry[0],](a, b, out, n, BLOCK=entry[1], STEP=entry[2])


def _overlaps(x, y):
    """Byte-range overlap test between two tensor views (metadata only)."""
    px, py = (x.data_ptr(), y.data_ptr())
    nx = x.numel() * x.element_size()
    ny = y.numel() * y.element_size()
    return px < py + ny and py < px + nx


def run(input, other, dim=None, *, out):
    pa = input.data_ptr()
    po = out.data_ptr()
    if torch._C._is_alias_of(input, out):
        input = _strict_clone(input)
    pb = other.data_ptr()
    if torch._C._is_alias_of(other, out):
        other = _strict_clone(other)
    if dim is None:
        dim = _default_dim(input, other)
    nd = input.dim()
    if dim < 0:
        dim += nd
    if (
        dim == nd - 1
        and input.shape == out.shape
        and (other.shape == out.shape)
        and (input.shape[dim] == 3)
        and (other.shape[dim] == 3)
        and (out.dtype == input.dtype)
        and (other.dtype == input.dtype)
        and input.is_contiguous()
        and other.is_contiguous()
        and out.is_contiguous()
    ):
        n_elem = out.numel()
        if n_elem:
            pa = input.data_ptr()
            pb = other.data_ptr()
            po = out.data_ptr()
            _launch_flat(input, other, out, n_elem, pa, pb, po)
        return out
    out_shape = torch.broadcast_shapes(input.shape, other.shape)
    if input.shape[dim] != 3 or other.shape[dim] != 3:
        raise RuntimeError("cross: dimension must be of size 3")
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"cross: expected out dtype {input.dtype}, but got {out.dtype}"
        )
    if out.device != input.device:
        raise RuntimeError("cross: out must be on the same device as input")
    if tuple(out.shape) != tuple(out_shape):
        out.resize_(out_shape)
    sa = _broadcast_strides(input, out_shape)
    sb = _broadcast_strides(other, out_shape)
    so = _broadcast_strides(out, out_shape)
    vectors = out.numel() // 3
    if vectors == 0:
        return out
    blk = 64
    _cross_general_kernel[(vectors + blk - 1) // blk,](
        input,
        other,
        out,
        vectors,
        tuple(out_shape),
        dim,
        tuple(sa),
        tuple(sb),
        tuple(so),
        BLOCK_I=blk,
    )
    return out


"""候选内嵌的Triton存储操作：复制、连续化、填充、转换均执行Triton核。"""


@triton.jit
def _strict_copy_kernel(
    X,
    Y,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    XS: tl.constexpr,
    YS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remain = i
    xp = tl.full((BLOCK,), 0, tl.int64)
    yp = tl.full((BLOCK,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = remain % SHAPE[d]
        remain = remain // SHAPE[d]
        xp += coord * XS[d]
        yp += coord * YS[d]
    value = tl.load(X + xp, i < N, other=0)
    tl.store(Y + yp, value, i < N)


@triton.jit
def _strict_fill_kernel(Y, N: tl.constexpr, VALUE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, VALUE, i < N)


@triton.jit
def _strict_random_kernel(Y, N: tl.constexpr, SEED, OFFSET, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # 备用随机采样路径；全部随机数计算在Triton内，生成器仅在宿主管理状态。
    value = tl.rand(SEED, (i.to(tl.uint64) + OFFSET).to(tl.uint32))
    tl.store(Y + i, value, i < N)


def _strict_empty_like(x, *, dtype=None, **kwargs):
    return torch.empty(
        tuple(x.shape), device=x.device, dtype=x.dtype if dtype is None else dtype
    )


def _strict_copy_into(dst, src):
    if tuple(dst.shape) != tuple(src.shape):
        raise ValueError("strict copy requires equal shapes")
    if dst.numel():
        _strict_copy_kernel[(triton.cdiv(dst.numel(), 1024),)](
            src,
            dst,
            dst.numel(),
            tuple(dst.shape),
            tuple(src.stride()),
            tuple(dst.stride()),
            1024,
        )
    return dst


def _strict_clone(x, **kwargs):
    return _strict_copy_into(_strict_empty_like(x), x)


def _strict_contiguous(x, *args, **kwargs):
    return x if x.is_contiguous() else _strict_clone(x)


def _strict_cast(x, dtype):
    if dtype == x.dtype:
        return x
    return _strict_copy_into(_strict_empty_like(x, dtype=dtype), x)


def _strict_reshape(x, *shape):
    # view只操作元数据，输入连续化由上面的Triton复制完成。
    return _strict_contiguous(x).view(*shape)


def _strict_flatten(x, start_dim=0, end_dim=-1):
    rank = x.ndim
    first = start_dim % rank
    last = end_dim % rank
    size = 1
    for n in x.shape[first : last + 1]:
        size *= n
    shape = tuple(x.shape[:first]) + (size,) + tuple(x.shape[last + 1 :])
    return _strict_contiguous(x).view(shape)


def _strict_full(shape, value, **kwargs):
    if isinstance(shape, int):
        shape = (shape,)
    out = torch.empty(shape, **kwargs)
    if out.numel():
        _strict_fill_kernel[(triton.cdiv(out.numel(), 1024),)](
            out, out.numel(), value, 1024
        )
    return out


def _strict_zeros(*shape, **kwargs):
    shape = shape[0] if len(shape) == 1 else shape
    return _strict_full(shape, 0, **kwargs)


def _strict_ones(*shape, **kwargs):
    shape = shape[0] if len(shape) == 1 else shape
    return _strict_full(shape, 1, **kwargs)


def _strict_rand(*shape, **kwargs):
    shape = shape[0] if len(shape) == 1 else shape
    generator = kwargs.pop("generator", None)
    out = torch.empty(shape, **kwargs)
    if out.numel():
        if generator is None:
            import torch_npu  # noqa: F401

            generator = torch.npu.default_generators[out.device.index or 0]
        seed = generator.initial_seed()
        offset = generator.get_offset()
        advance = triton.cdiv(out.numel(), 4) * 4
        if offset + advance >= 2**32:
            raise RuntimeError("备用随机路径超过32位计数范围，明确拒绝而非静默重复")
        generator.set_offset(offset + advance)
        _strict_random_kernel[(triton.cdiv(out.numel(), 256),)](
            out, out.numel(), seed, offset, 256
        )
    return out


# --------------------------------------------------------------------------- #
# FlagGems entry points
# --------------------------------------------------------------------------- #
def cross_out(*args, **kwargs):
    return run(*args, **kwargs)
