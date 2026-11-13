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

"""In-place ``_foreach_addcdiv_`` (Scalar overload) implemented with Triton.

Semantics: for every entry ``i`` of the tensor list,
``self[i] = self[i] + value * tensor1[i] / tensor2[i]`` evaluated in float32
opmath and stored back into the existing ``self[i]`` storage. ``run`` returns
``None`` (void, in-place).

The measured cost of a long list of small tensors is host dispatch, not device
work (a fused launch of 16x[64,64] runs 4.8 us on device inside a ~230 us
call), so the hot path is kept minimal:

  * one metadata pass over the list decides between the fused hot path and the
    general path;
  * the fused launch plan (compiled kernel, packed metadata, grid, constexpr
    block bounds) is cached per (entry count, per-entry numels, dtype, device);
  * the launch is issued with ``CompiledKernel.run`` and a cached stream getter,
    skipping Triton's JIT front end, its per-argument binding and the runner
    closure's driver queries. Alignment specialization is disabled for the
    fused kernel's pointer parameters so the plan key needs no pointer probes,
    and a failed fast launch permanently falls back to the ordinary JIT launch.

Bandwidth-bound entries keep one alignment-specialized kernel launch per entry;
non-contiguous / broadcast entries use a strided kernel; lists whose entries
share storage fall back to ordered per-entry launches.
"""

import logging

import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAX_T = 16
_SMALL = 65536

_PTR_ARGS = tuple(range(3 * _MAX_T))  # pointer params of the fused kernel
_FAST_ALLOWED = None  # None == not probed yet
_FAST_CACHE = {}  # generic cached CompiledKernel runners
_PLAN = {}  # small fused launch plans
_STREAM_FN = None


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _entry_flat(a_ptr, b_ptr, c_ptr, block_start, n, value, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64) - block_start
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    acc = a + (value * b) / c
    tl.store(a_ptr + offs, acc.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _addcdiv_flat_kernel(a_ptr, b_ptr, c_ptr, value, n, BLOCK: tl.constexpr):
    _entry_flat(a_ptr, b_ptr, c_ptr, 0, n, value, BLOCK)


@triton.jit(do_not_specialize_on_alignment=_PTR_ARGS)
def _addcdiv_fused16_kernel(
    a0,
    b0,
    c0,
    a1,
    b1,
    c1,
    a2,
    b2,
    c2,
    a3,
    b3,
    c3,
    a4,
    b4,
    c4,
    a5,
    b5,
    c5,
    a6,
    b6,
    c6,
    a7,
    b7,
    c7,
    a8,
    b8,
    c8,
    a9,
    b9,
    c9,
    a10,
    b10,
    c10,
    a11,
    b11,
    c11,
    a12,
    b12,
    c12,
    a13,
    b13,
    c13,
    a14,
    b14,
    c14,
    a15,
    b15,
    c15,
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
    value,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
    B4: tl.constexpr,
    B5: tl.constexpr,
    B6: tl.constexpr,
    B7: tl.constexpr,
    B8: tl.constexpr,
    B9: tl.constexpr,
    B10: tl.constexpr,
    B11: tl.constexpr,
    B12: tl.constexpr,
    B13: tl.constexpr,
    B14: tl.constexpr,
    B15: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # B_j is the exclusive upper block bound of slot j-1, so the last active
    # slot is selected by the "else" branch and unused slots are padded with the
    # total block count (which no program satisfies before the last slot).
    pid = tl.program_id(0)
    if pid < B1:
        _entry_flat(a0, b0, c0, 0, n0, value, BLOCK)
    elif pid < B2:
        _entry_flat(a1, b1, c1, B1, n1, value, BLOCK)
    elif pid < B3:
        _entry_flat(a2, b2, c2, B2, n2, value, BLOCK)
    elif pid < B4:
        _entry_flat(a3, b3, c3, B3, n3, value, BLOCK)
    elif pid < B5:
        _entry_flat(a4, b4, c4, B4, n4, value, BLOCK)
    elif pid < B6:
        _entry_flat(a5, b5, c5, B5, n5, value, BLOCK)
    elif pid < B7:
        _entry_flat(a6, b6, c6, B6, n6, value, BLOCK)
    elif pid < B8:
        _entry_flat(a7, b7, c7, B7, n7, value, BLOCK)
    elif pid < B9:
        _entry_flat(a8, b8, c8, B8, n8, value, BLOCK)
    elif pid < B10:
        _entry_flat(a9, b9, c9, B9, n9, value, BLOCK)
    elif pid < B11:
        _entry_flat(a10, b10, c10, B10, n10, value, BLOCK)
    elif pid < B12:
        _entry_flat(a11, b11, c11, B11, n11, value, BLOCK)
    elif pid < B13:
        _entry_flat(a12, b12, c12, B12, n12, value, BLOCK)
    elif pid < B14:
        _entry_flat(a13, b13, c13, B13, n13, value, BLOCK)
    elif pid < B15:
        _entry_flat(a14, b14, c14, B14, n14, value, BLOCK)
    else:
        _entry_flat(a15, b15, c15, B15, n15, value, BLOCK)


@triton.jit
def _addcdiv_strided_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    value,
    n,
    sa0,
    sa1,
    sa2,
    sa3,
    sb0,
    sb1,
    sb2,
    sb3,
    sc0,
    sc1,
    sc2,
    sc3,
    ds0,
    ds1,
    ds2,
    ds3,
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
):
    # ``offs`` is the flat index of the logical (row-major) shape and ``dsK`` are
    # the dense strides of that shape, so dims are decomposed outermost first.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs
    if RANK == 1:
        ia = rem * sa0
        ib = rem * sb0
        ic = rem * sc0
    elif RANK == 2:
        q = rem // ds0
        rem = rem - q * ds0
        ia = q * sa0 + rem * sa1
        ib = q * sb0 + rem * sb1
        ic = q * sc0 + rem * sc1
    elif RANK == 3:
        q = rem // ds0
        rem = rem - q * ds0
        ia = q * sa0
        ib = q * sb0
        ic = q * sc0
        q = rem // ds1
        rem = rem - q * ds1
        ia += q * sa1 + rem * sa2
        ib += q * sb1 + rem * sb2
        ic += q * sc1 + rem * sc2
    else:
        q = rem // ds0
        rem = rem - q * ds0
        ia = q * sa0
        ib = q * sb0
        ic = q * sc0
        q = rem // ds1
        rem = rem - q * ds1
        ia += q * sa1
        ib += q * sb1
        ic += q * sc1
        q = rem // ds2
        rem = rem - q * ds2
        ia += q * sa2 + rem * sa3
        ib += q * sb2 + rem * sb3
        ic += q * sc2 + rem * sc3
    a = tl.load(a_ptr + ia, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + ib, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + ic, mask=mask, other=0.0).to(tl.float32)
    acc = a + (value * b) / c
    tl.store(a_ptr + ia, acc.to(a_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------


def _fast_probe():
    """True when the low-level CompiledKernel launch reproduces a JIT launch."""
    global _FAST_ALLOWED
    try:
        from triton.runtime.jit import DistributedRtContext
    except Exception:
        _FAST_ALLOWED = True
        return True
    try:
        _FAST_ALLOWED = not bool(DistributedRtContext().is_lite_mode)
    except Exception:
        _FAST_ALLOWED = False
    return _FAST_ALLOWED


def _jit_launch_flat(args, grid0, device):
    grid = (grid0, 1, 1)
    return _addcdiv_flat_kernel[grid](*args, num_warps=4)


def _jit_launch_fused(args, grid0, device):
    grid = (grid0, 1, 1)
    return _addcdiv_fused16_kernel[grid](*args, num_warps=4)


def _jit_launch_strided(args, grid0, device):
    grid = (grid0, 1, 1)
    return _addcdiv_strided_kernel[grid](*args, num_warps=4)


def _launch(jit_launch, key, args, grid0, device):
    """Launch once per specialization through a cached CompiledKernel runner."""
    global _FAST_ALLOWED
    if _FAST_ALLOWED is None:
        _fast_probe()
    if _FAST_ALLOWED:
        runner = _FAST_CACHE.get(key)
        if runner is not None:
            try:
                runner(*args)
                return
            except Exception:
                _FAST_CACHE.clear()
                _FAST_ALLOWED = False
    ck = jit_launch(args, grid0, device)
    if _FAST_ALLOWED and ck is not None:
        try:
            _FAST_CACHE[key] = ck[(grid0, 1, 1)]
        except Exception:
            _FAST_CACHE.clear()
            _FAST_ALLOWED = False


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def _align_strides(t, target_shape):
    """Broadcast strides of ``t`` against ``target_shape`` (stride 0 == repeat)."""
    rank = len(target_shape)
    if t.dim() == 0:
        return (0,) * rank
    offset = rank - t.dim()
    out = []
    for i in range(rank):
        j = i - offset
        if j < 0:
            out.append(0)
            continue
        size = t.shape[j]
        tgt = target_shape[i]
        if size == tgt:
            out.append(t.stride(j))
        elif size == 1:
            out.append(0)
        else:
            raise RuntimeError(
                "foreach_addcdiv_: tensor list element with shape {} cannot be "
                "broadcast against shape {}".format(tuple(t.shape), tuple(target_shape))
            )
    return tuple(out)


def _collapse(target_shape, *stride_sets):
    """Drop size-1 dims and merge adjacent dims that stay dense for all operands."""
    rank = len(target_shape)
    dims = []
    for i in range(rank):
        if target_shape[i] != 1:
            dims.append((target_shape[i], tuple(s[i] for s in stride_sets)))
    if not dims:
        return (1,), [(1,) for _ in stride_sets]
    merged = [[dims[0][0], list(dims[0][1])]]
    for size, strides in dims[1:]:
        prev_size, prev_strides = merged[-1]
        can_merge = True
        for k in range(len(strides)):
            if prev_strides[k] != strides[k] * size:
                can_merge = False
                break
        if can_merge:
            merged[-1][0] = prev_size * size
            merged[-1][1] = list(strides)
        else:
            merged.append([size, list(strides)])
    shape = tuple(m[0] for m in merged)
    stride_sets_out = [tuple(m[1][k] for m in merged) for k in range(len(stride_sets))]
    return shape, stride_sets_out


def _launch_strided(a, b, c, value, numel, shape, sa, sb, sc):
    rank = len(shape)
    if rank > 4:
        # Extremely rare exotic layout: peel the outermost dimension with
        # zero-copy narrow views until the rank fits the kernel.
        for i in range(shape[0]):
            sub = [t.narrow(0, i, 1).squeeze(0) for t in (a, b, c)]
            sub_shape = tuple(shape[1:])
            sub_strides = [_align_strides(t, sub_shape) for t in sub]
            shape_c, (sa2, sb2, sc2) = _collapse(
                sub_shape, sub_strides[0], sub_strides[1], sub_strides[2]
            )
            _launch_strided(
                sub[0], sub[1], sub[2], value, sub[0].numel(), shape_c, sa2, sb2, sc2
            )
        return
    ds = [1, 1, 1, 1]
    acc = 1
    for i in range(rank - 1, -1, -1):
        ds[i] = acc
        acc *= shape[i]
    sa_p = list(sa) + [0] * (4 - rank)
    sb_p = list(sb) + [0] * (4 - rank)
    sc_p = list(sc) + [0] * (4 - rank)
    BLOCK = 256 if numel <= _SMALL else 1024
    args = (
        a,
        b,
        c,
        value,
        numel,
        sa_p[0],
        sa_p[1],
        sa_p[2],
        sa_p[3],
        sb_p[0],
        sb_p[1],
        sb_p[2],
        sb_p[3],
        sc_p[0],
        sc_p[1],
        sc_p[2],
        sc_p[3],
        ds[0],
        ds[1],
        ds[2],
        ds[3],
        BLOCK,
        rank,
    )
    grid0 = triton.cdiv(numel, BLOCK)
    key = (3, args[5:], a.dtype, b.dtype, c.dtype, grid0)
    _launch(_jit_launch_strided, key, args, grid0, a.device.index)


def _launch_entry(a, b, c, value, numel, small):
    BLOCK = 256 if small else 1024
    grid0 = triton.cdiv(numel, BLOCK)
    args = (a, b, c, value, numel, BLOCK)
    key = (
        1,
        numel,
        BLOCK,
        a.dtype,
        b.dtype,
        c.dtype,
        a.data_ptr() & 15,
        b.data_ptr() & 15,
        c.data_ptr() & 15,
        grid0,
    )
    _launch(_jit_launch_flat, key, args, grid0, a.device.index)


def _launch_fused(group, value, numels, boundaries, total, small):
    """General fused launch (JIT or cached runner) for one chunk of entries."""
    k = len(group)
    if k == 1:
        a, b, c = group[0]
        _launch_entry(a, b, c, value, numels[0], small)
        return
    ptrs = [None] * (3 * _MAX_T)
    for i in range(_MAX_T):
        if i < k:
            a, b, c = group[i]
        else:
            a, b, c = group[0]
        ptrs[3 * i] = a
        ptrs[3 * i + 1] = b
        ptrs[3 * i + 2] = c
    ns = numels + [0] * (_MAX_T - k)
    bounds = list(boundaries)
    bounds += [total] * (_MAX_T - 1 - len(bounds))
    BLOCK = 256 if small else 1024
    args = tuple(ptrs) + tuple(ns) + (value,) + tuple(bounds) + (BLOCK,)
    key = (
        2,
        tuple(numels),
        tuple(bounds),
        BLOCK,
        tuple(t.dtype for e in group for t in e),
        total,
    )
    _launch(_jit_launch_fused, key, args, total, group[0][0].device.index)


def _strided_entry(a, b, c, value, numel):
    target_shape = tuple(a.shape)
    sa = _align_strides(a, target_shape)
    sb = _align_strides(b, target_shape)
    sc = _align_strides(c, target_shape)
    shape_c, (sa_c, sb_c, sc_c) = _collapse(target_shape, sa, sb, sc)
    _launch_strided(a, b, c, value, numel, shape_c, sa_c, sb_c, sc_c)


# ---------------------------------------------------------------------------
# Hot path for a long list of small contiguous entries
# ---------------------------------------------------------------------------


def _stream(dev):
    global _STREAM_FN
    fn = _STREAM_FN
    if fn is None:
        from triton.runtime import driver

        fn = driver.active.get_current_stream
        _STREAM_FN = fn
    return fn(dev)


def _fused_args(tensors, k, ns, value, post):
    args = list(tensors)
    if k < _MAX_T:
        fill = tensors[:3]
        for _ in range(k, _MAX_T):
            args += fill
    args.extend(ns)
    args.append(value)
    args.extend(post)
    return args


def _hot_launch(tensors, numels, value, dev):
    """Fused launch of <= _MAX_T small flat entries with a cached launch plan."""
    global _FAST_ALLOWED
    k = len(numels)
    key = (k, tuple(numels), tensors[0].dtype, dev)
    plan = _PLAN.get(key)
    if plan is None:
        counts = [triton.cdiv(ne, 256) for ne in numels]
        total = sum(counts)
        bounds = []
        acc = 0
        for cnt in counts[:-1]:
            acc += cnt
            bounds.append(acc)
        bounds += [total] * (_MAX_T - 1 - len(bounds))
        ns = tuple(numels) + (0,) * (_MAX_T - k)
        post = tuple(bounds) + (256,)
        ck = _jit_launch_fused(_fused_args(tensors, k, ns, value, post), total, dev)
        if _FAST_ALLOWED and ck is not None:
            try:
                _PLAN[key] = (ck.run, ck.function, ck.packed_metadata, total, ns, post)
            except Exception:
                _FAST_ALLOWED = False
        return
    run, fn, md, grid0, ns, post = plan
    if _FAST_ALLOWED:
        try:
            run(
                grid0,
                1,
                1,
                _stream(dev),
                fn,
                md,
                None,
                None,
                None,
                *_fused_args(tensors, k, ns, value, post),
            )
            return
        except Exception:
            _PLAN.clear()
            _FAST_ALLOWED = False
    _jit_launch_fused(_fused_args(tensors, k, ns, value, post), grid0, dev)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _foreach_addcdiv__Scalar(self, tensor1, tensor2, value=1):
    logger.debug("GEMS_HYGON _FOREACH_ADDCDIV__SCALAR")
    a = self if type(self) is list else list(self)
    n = len(a)
    if n == 0:
        return None
    b = tensor1 if type(tensor1) is list else list(tensor1)
    c = tensor2 if type(tensor2) is list else list(tensor2)
    if len(b) != n or len(c) != n:
        raise RuntimeError(
            "foreach_addcdiv_: tensor list length mismatch: expected {}, got {} and {}".format(
                n, len(b), len(c)
            )
        )
    v = float(value)

    if _FAST_ALLOWED is None:
        _fast_probe()

    if n <= _MAX_T:
        # One metadata pass: decide the fused hot path, otherwise fall through.
        numels = [0] * n
        flat_ptrs = [0] * n
        dt = a[0].dtype
        hot = True
        for i in range(n):
            ai = a[i]
            bi = b[i]
            ci = c[i]
            ne = ai.numel()
            if ne == 0 or ne > _SMALL:
                hot = False
                break
            numels[i] = ne
            flat_ptrs[i] = ai.data_ptr()
            if not (
                ai.shape == bi.shape
                and ai.shape == ci.shape
                and ai.is_contiguous()
                and bi.is_contiguous()
                and ci.is_contiguous()
                and ai.dtype is dt
                and bi.dtype is dt
                and ci.dtype is dt
            ):
                hot = False
                break
        if hot and len(set(flat_ptrs)) == n:
            tensors = []
            for i in range(n):
                tensors.append(a[i])
                tensors.append(b[i])
                tensors.append(c[i])
            _hot_launch(tensors, numels, v, a[0].device.index)
            return None

    if n == 1:
        a0 = a[0]
        b0 = b[0]
        c0 = c[0]
        n_el = a0.numel()
        if n_el == 0:
            return None
        if (
            a0.shape == b0.shape == c0.shape
            and a0.is_contiguous()
            and b0.is_contiguous()
            and c0.is_contiguous()
        ):
            _launch_entry(a0, b0, c0, v, n_el, n_el <= _SMALL)
        else:
            _strided_entry(a0, b0, c0, v, n_el)
        return None

    # Alias detection: writing self[i] must not race another list entry.
    write_owner = {}
    for i in range(n):
        ai = a[i]
        if ai.numel():
            write_owner.setdefault(ai.data_ptr(), i)
    aliased = False
    for i in range(n):
        for t in (a[i], b[i], c[i]):
            if t.numel():
                owner = write_owner.get(t.data_ptr())
                if owner is not None and owner != i:
                    aliased = True
                    break
        if aliased:
            break

    if aliased:
        for i in range(n):
            ai = a[i]
            bi = b[i]
            ci = c[i]
            n_el = ai.numel()
            if n_el == 0:
                continue
            if (
                ai.shape == bi.shape == ci.shape
                and ai.is_contiguous()
                and bi.is_contiguous()
                and ci.is_contiguous()
            ):
                _launch_entry(ai, bi, ci, v, n_el, n_el <= _SMALL)
            else:
                _strided_entry(ai, bi, ci, v, n_el)
        return None

    # Partition into (size class, flat-eligible) groups. Only the launch-bound
    # size class is fused; large entries launch one kernel each.
    small_flat = []
    large_flat = []
    strided = []
    for i in range(n):
        ai = a[i]
        bi = b[i]
        ci = c[i]
        n_el = ai.numel()
        if n_el == 0:
            continue
        if (
            ai.shape == bi.shape == ci.shape
            and ai.is_contiguous()
            and bi.is_contiguous()
            and ci.is_contiguous()
        ):
            (small_flat if n_el <= _SMALL else large_flat).append((ai, bi, ci, n_el))
        else:
            strided.append((ai, bi, ci, n_el))

    for start in range(0, len(small_flat), _MAX_T):
        chunk = small_flat[start : start + _MAX_T]
        numels = [e[3] for e in chunk]
        counts = [triton.cdiv(n_el, 256) for n_el in numels]
        total = sum(counts)
        boundaries = []
        acc = 0
        for cnt in counts[:-1]:
            acc += cnt
            boundaries.append(acc)
        _launch_fused(
            [(e[0], e[1], e[2]) for e in chunk],
            v,
            numels,
            boundaries,
            total,
            True,
        )

    for ai, bi, ci, n_el in large_flat:
        _launch_entry(ai, bi, ci, v, n_el, False)

    for ai, bi, ci, n_el in strided:
        _strided_entry(ai, bi, ci, v, n_el)

    return None
