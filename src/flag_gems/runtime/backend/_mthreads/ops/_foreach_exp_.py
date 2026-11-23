"""Triton implementation of aten::_foreach_exp_ (in-place exp over a tensor list).

Round 6: host-side metadata trimming on top of the round-5 structure.

  * one pass over the list builds both the per-dtype tensor list and its numel
    list, so the uniform check no longer re-reads every numel;
  * the uniform test is an O(k) C-level ``min == max`` comparison;
  * the launcher closure returned by ``CompiledKernel[grid]`` is cached together
    with its grid, so a repeated grid skips the closure construction.

Device code is unchanged from round 5: equal numels use the 2D
``(blocks, k)`` grid with 19 argument slots, unequal numels keep the 1D exact
grid with block prefix sums, and every tile takes the unmasked path when full.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_SLOTS = 16

_FLOAT_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32, torch.float64))

_ATEN_CAST_NAMES = {
    torch.int64: "Long",
    torch.int32: "Int",
    torch.int16: "Short",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.bool: "Bool",
}

_N_SLOT_ARGS = ["n%d" % i for i in range(_MAX_SLOTS)]
_B_SLOT_ARGS = ["b%d" % i for i in range(1, _MAX_SLOTS)]
_STRIDED_SCALARS = (
    ["n"] + ["sh%d" % i for i in range(8)] + ["st%d" % i for i in range(8)]
)

# (kind, dtype, device) -> (compiled kernel, grid, launcher closure)
_CACHE = {}


@triton.jit
def _exp_vals(x, IS_F64: tl.constexpr):
    if IS_F64:
        return tl.exp(x)
    return tl.exp(x.to(tl.float32)).to(x.dtype)


@triton.jit
def _exp_tile(ptr, base, n, BLOCK: tl.constexpr, IS_F64: tl.constexpr):
    """One block worth of in-place exp; unmasked when the tile is full."""
    offs = base + tl.arange(0, BLOCK)
    if base + BLOCK <= n:
        tl.store(ptr + offs, _exp_vals(tl.load(ptr + offs), IS_F64))
    else:
        mask = offs < n
        tl.store(
            ptr + offs,
            _exp_vals(tl.load(ptr + offs, mask=mask, other=0.0), IS_F64),
            mask=mask,
        )


@triton.jit(do_not_specialize=["n"])
def _exp_flat_kernel(ptr, n, BLOCK: tl.constexpr, IS_F64: tl.constexpr):
    _exp_tile(ptr, tl.program_id(0) * BLOCK, n, BLOCK, IS_F64)


@triton.jit(do_not_specialize=["n"])
def _exp_uniform_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    n,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Equal numel in every slot: grid is (blocks_per_tensor, k)."""
    s = tl.program_id(1)
    base = tl.program_id(0) * BLOCK
    if s == 0:
        _exp_tile(p0, base, n, BLOCK, IS_F64)
    elif s == 1:
        _exp_tile(p1, base, n, BLOCK, IS_F64)
    elif s == 2:
        _exp_tile(p2, base, n, BLOCK, IS_F64)
    elif s == 3:
        _exp_tile(p3, base, n, BLOCK, IS_F64)
    elif s == 4:
        _exp_tile(p4, base, n, BLOCK, IS_F64)
    elif s == 5:
        _exp_tile(p5, base, n, BLOCK, IS_F64)
    elif s == 6:
        _exp_tile(p6, base, n, BLOCK, IS_F64)
    elif s == 7:
        _exp_tile(p7, base, n, BLOCK, IS_F64)
    elif s == 8:
        _exp_tile(p8, base, n, BLOCK, IS_F64)
    elif s == 9:
        _exp_tile(p9, base, n, BLOCK, IS_F64)
    elif s == 10:
        _exp_tile(p10, base, n, BLOCK, IS_F64)
    elif s == 11:
        _exp_tile(p11, base, n, BLOCK, IS_F64)
    elif s == 12:
        _exp_tile(p12, base, n, BLOCK, IS_F64)
    elif s == 13:
        _exp_tile(p13, base, n, BLOCK, IS_F64)
    elif s == 14:
        _exp_tile(p14, base, n, BLOCK, IS_F64)
    else:
        _exp_tile(p15, base, n, BLOCK, IS_F64)


@triton.jit(do_not_specialize=_N_SLOT_ARGS + _B_SLOT_ARGS)
def _exp_fused_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    n8,
    n9,
    n10,
    n11,
    n12,
    n13,
    n14,
    n15,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    b8,
    b9,
    b10,
    b11,
    b12,
    b13,
    b14,
    b15,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Heterogeneous numels: grid is (sum of per-slot block counts,)."""
    pid = tl.program_id(0)
    if pid < b1:
        _exp_tile(p0, pid * BLOCK, n0, BLOCK, IS_F64)
    elif pid < b2:
        _exp_tile(p1, (pid - b1) * BLOCK, n1, BLOCK, IS_F64)
    elif pid < b3:
        _exp_tile(p2, (pid - b2) * BLOCK, n2, BLOCK, IS_F64)
    elif pid < b4:
        _exp_tile(p3, (pid - b3) * BLOCK, n3, BLOCK, IS_F64)
    elif pid < b5:
        _exp_tile(p4, (pid - b4) * BLOCK, n4, BLOCK, IS_F64)
    elif pid < b6:
        _exp_tile(p5, (pid - b5) * BLOCK, n5, BLOCK, IS_F64)
    elif pid < b7:
        _exp_tile(p6, (pid - b6) * BLOCK, n6, BLOCK, IS_F64)
    elif pid < b8:
        _exp_tile(p7, (pid - b7) * BLOCK, n7, BLOCK, IS_F64)
    elif pid < b9:
        _exp_tile(p8, (pid - b8) * BLOCK, n8, BLOCK, IS_F64)
    elif pid < b10:
        _exp_tile(p9, (pid - b9) * BLOCK, n9, BLOCK, IS_F64)
    elif pid < b11:
        _exp_tile(p10, (pid - b10) * BLOCK, n10, BLOCK, IS_F64)
    elif pid < b12:
        _exp_tile(p11, (pid - b11) * BLOCK, n11, BLOCK, IS_F64)
    elif pid < b13:
        _exp_tile(p12, (pid - b12) * BLOCK, n12, BLOCK, IS_F64)
    elif pid < b14:
        _exp_tile(p13, (pid - b13) * BLOCK, n13, BLOCK, IS_F64)
    elif pid < b15:
        _exp_tile(p14, (pid - b14) * BLOCK, n14, BLOCK, IS_F64)
    else:
        _exp_tile(p15, (pid - b15) * BLOCK, n15, BLOCK, IS_F64)


@triton.jit(do_not_specialize=_STRIDED_SCALARS)
def _exp_strided_kernel(
    ptr,
    n,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    sh6,
    sh7,
    st0,
    st1,
    st2,
    st3,
    st4,
    st5,
    st6,
    st7,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    d = idx
    a = tl.zeros((BLOCK,), dtype=tl.int32)
    d0 = d % sh0
    d = d // sh0
    a = a + d0 * st0
    d1 = d % sh1
    d = d // sh1
    a = a + d1 * st1
    d2 = d % sh2
    d = d // sh2
    a = a + d2 * st2
    d3 = d % sh3
    d = d // sh3
    a = a + d3 * st3
    d4 = d % sh4
    d = d // sh4
    a = a + d4 * st4
    d5 = d % sh5
    d = d // sh5
    a = a + d5 * st5
    d6 = d % sh6
    d = d // sh6
    a = a + d6 * st6
    d7 = d % sh7
    a = a + d7 * st7
    x = tl.load(ptr + a, mask=mask, other=0.0)
    tl.store(ptr + a, _exp_vals(x, IS_F64), mask=mask)


def _flat_addressable(t: torch.Tensor) -> bool:
    """True when [data_ptr, data_ptr + numel) covers exactly this tensor."""
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted(
        (s, sz) for s, sz in zip(t.stride(), t.shape) if sz != 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def _launch_group(ts, numels, dtype, device):
    """Dispatch 1..16 flat-addressable tensors of one dtype through one launch."""
    k = len(ts)
    is_f64 = dtype == torch.float64
    if k == 1:
        n = numels[0]
        grid = ((n + _BLOCK - 1) // _BLOCK, 1, 1)
        args = (ts[0], n, _BLOCK, is_f64)
        key = ("flat", dtype, device)
        entry = _CACHE.get(key)
        if entry is not None and entry[1] == grid:
            try:
                entry[2](*args)
                return
            except Exception:
                pass
        compiled = _exp_flat_kernel[grid](*args, num_warps=_NUM_WARPS)
        _CACHE[key] = (compiled, grid, compiled[grid])
        return
    n0 = numels[0]
    if min(numels) == max(numels):
        grid = ((n0 + _BLOCK - 1) // _BLOCK, k, 1)
        args = (*ts, *([ts[0]] * (_MAX_SLOTS - k)), n0, is_f64, _BLOCK)
        key = ("uniform", dtype, device)
        entry = _CACHE.get(key)
        if entry is not None and entry[1] == grid:
            try:
                entry[2](*args)
                return
            except Exception:
                pass
        compiled = _exp_uniform_kernel[grid](*args, num_warps=_NUM_WARPS)
        _CACHE[key] = (compiled, grid, compiled[grid])
        return
    cum = []
    acc = 0
    for n in numels:
        acc += (n + _BLOCK - 1) // _BLOCK
        cum.append(acc)
    total = acc
    bounds = (cum + [total] * 15)[:15]
    fill = ts[0]
    ptrs = ts + [fill] * (_MAX_SLOTS - k)
    ns = numels + [0] * (_MAX_SLOTS - k)
    grid = (total, 1, 1)
    args = (*ptrs, *ns, *bounds, is_f64, _BLOCK)
    key = ("fused", dtype, device)
    entry = _CACHE.get(key)
    if entry is not None and entry[1] == grid:
        try:
            entry[2](*args)
            return
        except Exception:
            pass
    compiled = _exp_fused_kernel[grid](*args, num_warps=_NUM_WARPS)
    _CACHE[key] = (compiled, grid, compiled[grid])


def _launch_strided(t):
    n = t.numel()
    ndim = min(t.dim(), 8)
    shape = list(t.shape)[:ndim] + [1] * (8 - ndim)
    stride = list(t.stride())[:ndim] + [0] * (8 - ndim)
    grid = ((n + _BLOCK - 1) // _BLOCK, 1, 1)
    args = (t, n, *shape, *stride, _BLOCK, t.dtype == torch.float64)
    key = ("strided", t.dtype, t.device)
    entry = _CACHE.get(key)
    if entry is not None and entry[1] == grid:
        try:
            entry[2](*args)
            return
        except Exception:
            pass
    compiled = _exp_strided_kernel[grid](*args, num_warps=_NUM_WARPS)
    _CACHE[key] = (compiled, grid, compiled[grid])


def _foreach_exp_(self):
    tensors = self
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(
            "argument 'self' must be tuple of Tensors, not "
            + type(tensors).__name__.capitalize()
        )
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    groups = {}
    gappy = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dtype = t.dtype
        if dtype not in _FLOAT_DTYPES:
            name = _ATEN_CAST_NAMES.get(
                dtype, str(dtype).replace("torch.", "").capitalize()
            )
            raise RuntimeError(
                f"result type Float can't be cast to the desired output type {name}"
            )
        n = t.numel()
        if n == 0:
            continue
        if _flat_addressable(t):
            bucket = groups.get(dtype)
            if bucket is None:
                groups[dtype] = ([t], [n])
            else:
                bucket[0].append(t)
                bucket[1].append(n)
        else:
            gappy.append(t)

    for dtype, bucket in groups.items():
        ts, numels = bucket
        device = ts[0].device
        if len(ts) <= _MAX_SLOTS:
            _launch_group(ts, numels, dtype, device)
        else:
            for i in range(0, len(ts), _MAX_SLOTS):
                _launch_group(
                    ts[i : i + _MAX_SLOTS], numels[i : i + _MAX_SLOTS], dtype, device
                )

    for t in gappy:
        if torch._debug_has_internal_overlap(t) == 1:
            raise RuntimeError(
                "unsupported operation: more than one element of the "
                "written-to tensor refers to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
        _launch_strided(t)
    return None
