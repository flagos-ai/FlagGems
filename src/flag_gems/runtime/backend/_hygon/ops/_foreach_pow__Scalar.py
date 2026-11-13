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

"""aten::_foreach_pow_.Scalar: in-place pow(t, exponent) for every tensor in a list."""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

logger = logging.getLogger(__name__)

_BLOCK = 1024
_NWARPS = 4
_MAX_SLOTS = 16


@triton.jit
def _apply(v, exp, MODE: tl.constexpr):
    f = v.to(tl.float32)
    if MODE == 1:
        r = libdevice.sqrt(f)
    elif MODE == 2:
        r = f * f
    else:
        r = libdevice.pow(f, exp)
    return r.to(v.dtype)


@triton.jit
def _pow_flat(
    ptr, base, numel, exp, BLOCK: tl.constexpr, MODE: tl.constexpr, EVEN: tl.constexpr
):
    offs = base + tl.arange(0, BLOCK)
    if EVEN:
        v = tl.load(ptr + offs)
        tl.store(ptr + offs, _apply(v, exp, MODE))
    else:
        mask = offs < numel
        v = tl.load(ptr + offs, mask=mask, other=1.0)
        tl.store(ptr + offs, _apply(v, exp, MODE), mask=mask)


@triton.jit
def _pow_flat_kernel(
    x0,
    n0,
    x1,
    n1,
    x2,
    n2,
    x3,
    n3,
    x4,
    n4,
    x5,
    n5,
    x6,
    n6,
    x7,
    n7,
    x8,
    n8,
    x9,
    n9,
    x10,
    n10,
    x11,
    n11,
    x12,
    n12,
    x13,
    n13,
    x14,
    n14,
    x15,
    n15,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
    exp,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < s1:
        _pow_flat(x0, pid * BLOCK, n0, exp, BLOCK, MODE, EVEN)
    elif pid < s2:
        _pow_flat(x1, (pid - s1) * BLOCK, n1, exp, BLOCK, MODE, EVEN)
    elif pid < s3:
        _pow_flat(x2, (pid - s2) * BLOCK, n2, exp, BLOCK, MODE, EVEN)
    elif pid < s4:
        _pow_flat(x3, (pid - s3) * BLOCK, n3, exp, BLOCK, MODE, EVEN)
    elif pid < s5:
        _pow_flat(x4, (pid - s4) * BLOCK, n4, exp, BLOCK, MODE, EVEN)
    elif pid < s6:
        _pow_flat(x5, (pid - s5) * BLOCK, n5, exp, BLOCK, MODE, EVEN)
    elif pid < s7:
        _pow_flat(x6, (pid - s6) * BLOCK, n6, exp, BLOCK, MODE, EVEN)
    elif pid < s8:
        _pow_flat(x7, (pid - s7) * BLOCK, n7, exp, BLOCK, MODE, EVEN)
    elif pid < s9:
        _pow_flat(x8, (pid - s8) * BLOCK, n8, exp, BLOCK, MODE, EVEN)
    elif pid < s10:
        _pow_flat(x9, (pid - s9) * BLOCK, n9, exp, BLOCK, MODE, EVEN)
    elif pid < s11:
        _pow_flat(x10, (pid - s10) * BLOCK, n10, exp, BLOCK, MODE, EVEN)
    elif pid < s12:
        _pow_flat(x11, (pid - s11) * BLOCK, n11, exp, BLOCK, MODE, EVEN)
    elif pid < s13:
        _pow_flat(x12, (pid - s12) * BLOCK, n12, exp, BLOCK, MODE, EVEN)
    elif pid < s14:
        _pow_flat(x13, (pid - s13) * BLOCK, n13, exp, BLOCK, MODE, EVEN)
    elif pid < s15:
        _pow_flat(x14, (pid - s14) * BLOCK, n14, exp, BLOCK, MODE, EVEN)
    else:
        _pow_flat(x15, (pid - s15) * BLOCK, n15, exp, BLOCK, MODE, EVEN)


@triton.jit
def _pow_flat_uniform_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    numel,
    exp,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = tl.program_id(1)
    base = pid * BLOCK
    if idx == 0:
        _pow_flat(x0, base, numel, exp, BLOCK, MODE, EVEN)
    elif idx == 1:
        _pow_flat(x1, base, numel, exp, BLOCK, MODE, EVEN)
    elif idx == 2:
        _pow_flat(x2, base, numel, exp, BLOCK, MODE, EVEN)
    elif idx == 3:
        _pow_flat(x3, base, numel, exp, BLOCK, MODE, EVEN)
    elif idx == 4:
        _pow_flat(x4, base, numel, exp, BLOCK, MODE, EVEN)
    elif idx == 5:
        _pow_flat(x5, base, numel, exp, BLOCK, MODE, EVEN)
    elif idx == 6:
        _pow_flat(x6, base, numel, exp, BLOCK, MODE, EVEN)
    elif idx == 7:
        _pow_flat(x7, base, numel, exp, BLOCK, MODE, EVEN)
    elif idx == 8:
        _pow_flat(x8, base, numel, exp, BLOCK, MODE, EVEN)
    elif idx == 9:
        _pow_flat(x9, base, numel, exp, BLOCK, MODE, EVEN)
    elif idx == 10:
        _pow_flat(x10, base, numel, exp, BLOCK, MODE, EVEN)
    elif idx == 11:
        _pow_flat(x11, base, numel, exp, BLOCK, MODE, EVEN)
    elif idx == 12:
        _pow_flat(x12, base, numel, exp, BLOCK, MODE, EVEN)
    elif idx == 13:
        _pow_flat(x13, base, numel, exp, BLOCK, MODE, EVEN)
    elif idx == 14:
        _pow_flat(x14, base, numel, exp, BLOCK, MODE, EVEN)
    else:
        _pow_flat(x15, base, numel, exp, BLOCK, MODE, EVEN)


@triton.jit
def _pow_strided_kernel(
    ptr,
    numel,
    d3,
    t3,
    d2,
    t2,
    d1,
    t1,
    d0,
    t0,
    exp,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    rem = offs
    addr = (rem % d3) * t3
    rem = rem // d3
    if NDIM > 1:
        addr += (rem % d2) * t2
        rem = rem // d2
    if NDIM > 2:
        addr += (rem % d1) * t1
        rem = rem // d1
    if NDIM > 3:
        addr += (rem % d0) * t0
    v = tl.load(ptr + addr, mask=mask, other=1.0)
    r = _apply(v, exp, MODE)
    tl.store(ptr + addr, r, mask=mask)


def _nblocks(numel, block):
    return (numel + block - 1) // block


def _block_for(t):
    # 32 bytes per thread for 2-byte dtypes (16 elements), 16 bytes for 4-byte.
    return 4096 if t.element_size() <= 2 else 1024


# A Triton JITFunction re-binds every argument, recomputes the specialization key
# and looks the kernel up in a dict on every call; that per-call dispatch dominates
# the tiny workloads (16 tensors of [64, 64], only microseconds of GPU work). After
# the first launch of a given signature we therefore reuse the compiled kernel's
# own launcher. The key covers device, dtype, fast-path mode, block size, every
# per-tensor numel and the pointer alignment class, so any change in the call shape
# either builds a new launcher or falls back to a plain JIT launch.
_LAUNCH_CACHE = {}
_BYPASS_BROKEN = set()

try:
    from triton import knobs as _knobs
    from triton.runtime import driver as _driver
    from triton.runtime.jit import DistributedRtContext as _DistCtx

    _BYPASS_OK = True
except Exception:  # pragma: no cover - defensive only
    _BYPASS_OK = False


def _make_runner(kern, grid0, grid1):
    run = kern.run
    fn = kern.function
    pmeta = kern.packed_metadata
    grid = (grid0, grid1, 1)
    get_dev = _driver.active.get_current_device
    get_stream = _driver.active.get_current_stream
    launch_md = kern.launch_metadata
    rt = _knobs.runtime
    lite = _DistCtx().is_lite_mode

    if lite:

        def runner(*args):
            stream = get_stream(get_dev())
            enter = rt.launch_enter_hook
            md = launch_md(grid, stream, *args) if enter is not None else None
            ctx = _DistCtx()
            run(
                grid0,
                grid1,
                1,
                stream,
                fn,
                pmeta,
                md,
                enter,
                rt.launch_exit_hook,
                ctx.comm_ptr,
                ctx.mem_ptr,
                *args,
            )

    else:

        def runner(*args):
            stream = get_stream(get_dev())
            enter = rt.launch_enter_hook
            md = launch_md(grid, stream, *args) if enter is not None else None
            run(
                grid0,
                grid1,
                1,
                stream,
                fn,
                pmeta,
                md,
                enter,
                rt.launch_exit_hook,
                *args,
            )

    return runner


def _launch_uniform(tensors, numel, exp, mode, block, even, bpt):
    cnt = len(tensors)
    if cnt < _MAX_SLOTS:
        args = list(tensors) + [tensors[-1]] * (_MAX_SLOTS - cnt)
    else:
        args = list(tensors)
    args.append(numel)
    args.append(exp)

    key = (
        tensors[0].device.index,
        tensors[0].dtype,
        mode,
        block,
        numel,
        cnt,
        tuple(t.data_ptr() & 15 for t in tensors),
    )
    runner = _LAUNCH_CACHE.get(key)
    if runner is not None:
        runner(*args, block, mode, even)
        return

    kern = _pow_flat_uniform_kernel[(bpt, cnt)](
        *args, BLOCK=block, MODE=mode, EVEN=even, num_warps=_NWARPS
    )
    if _BYPASS_OK and key not in _BYPASS_BROKEN:
        try:
            runner = _make_runner(kern, bpt, cnt)
        except Exception:  # pragma: no cover - keep the plain JIT path
            runner = None
        if runner is None:
            _BYPASS_BROKEN.add(key)
        else:
            _LAUNCH_CACHE[key] = runner


def _launch_flat(tensors, numels, exp, mode, block, cached):
    total = 0
    ss = []
    m = block - 1
    even = True
    for n in numels:
        if n & m:
            even = False
        total += (n + m) // block
        ss.append(total)
    if total == 0:
        return
    cnt = len(tensors)
    if cnt < _MAX_SLOTS:
        pad = _MAX_SLOTS - cnt
        xs = list(tensors) + [tensors[-1]] * pad
        ns = numels + [0] * pad
        ss.extend([total] * (_MAX_SLOTS - 1 - cnt))
    else:
        xs = list(tensors)
        ns = numels
        ss.pop()
    args = [v for pair in zip(xs, ns) for v in pair]
    args.extend(ss)
    args.append(exp)

    key = None
    if cached:
        key = (
            tensors[0].device.index,
            tensors[0].dtype,
            mode,
            block,
            tuple(numels),
            tuple(t.data_ptr() & 15 for t in tensors),
        )
        runner = _LAUNCH_CACHE.get(key)
        if runner is not None:
            runner(*args, block, mode, even)
            return

    kern = _pow_flat_kernel[(total,)](
        *args, BLOCK=block, MODE=mode, EVEN=even, num_warps=_NWARPS
    )
    if key is not None and _BYPASS_OK and key not in _BYPASS_BROKEN:
        try:
            runner = _make_runner(kern, total, 1)
        except Exception:  # pragma: no cover - keep the plain JIT path
            runner = None
        if runner is None:
            _BYPASS_BROKEN.add(key)
        else:
            _LAUNCH_CACHE[key] = runner


def _collapse(shape, stride):
    out = []
    for d, s in zip(shape, stride):
        if d == 1:
            continue
        if out and out[-1][1] == s * d:
            out[-1] = (out[-1][0] * d, s)
        else:
            out.append((d, s))
    return out


def _launch_strided_one(t, numel, dims, exp, mode):
    nd = len(dims)
    shapes = [1, 1, 1, 1]
    strides = [0, 0, 0, 0]
    for k in range(nd):
        shapes[4 - nd + k] = dims[k][0]
        strides[4 - nd + k] = dims[k][1]
    _pow_strided_kernel[(_nblocks(numel, _BLOCK),)](
        t,
        numel,
        shapes[3],
        strides[3],
        shapes[2],
        strides[2],
        shapes[1],
        strides[1],
        shapes[0],
        strides[0],
        exp,
        NDIM=nd,
        BLOCK=_BLOCK,
        MODE=mode,
        num_warps=_NWARPS,
    )


def _launch_strided(t, exp, mode):
    dims = _collapse(list(t.shape), list(t.stride()))
    if len(dims) <= 4:
        _launch_strided_one(t, t.numel(), dims, exp, mode)
        return
    lead = len(dims) - 4
    sizes = [d for d, _ in dims[:lead]]
    rest = dims[lead:]
    inner = 1
    for d, _ in rest:
        inner *= d
    idx = [0] * lead
    while True:
        view = t
        for k in range(lead):
            view = view[idx[k]]
        _launch_strided_one(view, inner, rest, exp, mode)
        k = lead - 1
        while k >= 0:
            idx[k] += 1
            if idx[k] < sizes[k]:
                break
            idx[k] = 0
            k -= 1
        if k < 0:
            break


def _process_general(tensors, numels, exp, mode):
    devs = []
    groups = []
    for t, n in zip(tensors, numels):
        d = t.device
        for gi in range(len(devs)):
            if devs[gi] == d:
                groups[gi].append((t, n))
                break
        else:
            devs.append(d)
            groups.append([(t, n)])
    for g in groups:
        m = len(g)
        i = 0
        while i < m:
            chunk = g[i : i + _MAX_SLOTS]
            i += _MAX_SLOTS
            cs = [p[0] for p in chunk]
            cn = [p[1] for p in chunk]
            contig = True
            for t in cs:
                if not t.is_contiguous():
                    contig = False
                    break
            if contig:
                dt = cs[0].dtype
                uniform = True
                for t in cs:
                    if t.dtype is not dt:
                        uniform = False
                        break
                _launch_flat(cs, cn, exp, mode, _block_for(cs[0]), uniform)
            else:
                for t, n in zip(cs, cn):
                    if n == 0:
                        continue
                    if t.is_contiguous():
                        _launch_flat([t], [n], exp, mode, _block_for(t), True)
                    else:
                        _launch_strided(t, exp, mode)


def _foreach_pow__Scalar(self, exponent):
    logger.debug("GEMS_HYGON _FOREACH_POW__SCALAR")
    if isinstance(self, torch.Tensor):
        tensors = (self,)
        n_t = 1
    else:
        tensors = self
        n_t = len(tensors)
        if n_t == 0:
            return None
    if isinstance(exponent, torch.Tensor):
        exponent = exponent.item()
    exp = float(exponent)
    if exp == 0.5:
        mode = 1
    elif exp == 2.0:
        mode = 2
    else:
        mode = 0

    t0 = tensors[0]
    dev0 = t0.device
    dt0 = t0.dtype
    numels = []
    plain = n_t <= _MAX_SLOTS
    for t in tensors:
        numels.append(t.numel())
        if plain and (t.dtype is not dt0 or t.device != dev0 or not t.is_contiguous()):
            plain = False
    if plain:
        first = numels[0]
        same = True
        for n in numels:
            if n != first:
                same = False
                break
        if same:
            block = _block_for(t0)
            bpt = (first + block - 1) // block
            if bpt:
                _launch_uniform(
                    tensors, first, exp, mode, block, first % block == 0, bpt
                )
            return None
        _launch_flat(tensors, numels, exp, mode, _block_for(t0), True)
    else:
        _process_general(tensors, numels, exp, mode)
    return None
