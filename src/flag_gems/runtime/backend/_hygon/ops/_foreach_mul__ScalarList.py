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

logger = logging.getLogger(__name__)

try:
    import triton.runtime.driver as _driver
except Exception:  # pragma: no cover
    _driver = None
try:
    import triton.knobs as _knobs

    _ENTER_HOOK = _knobs.runtime.launch_enter_hook
    _EXIT_HOOK = _knobs.runtime.launch_exit_hook
except Exception:  # pragma: no cover
    _ENTER_HOOK = None
    _EXIT_HOOK = None

_MAX_DIMS = 8
_FUSE = 4
_MISS = object()
# key -> (launcher __call__, compiled function handle, packed metadata)
_LAUNCH_CACHE = {}
_FAST_ENABLED = _driver is not None


@triton.jit
def _mul_fused_kernel(
    x0,
    x1,
    x2,
    x3,
    n0,
    n1,
    n2,
    n3,
    s0,
    s1,
    s2,
    s3,
    BLOCK: tl.constexpr,
    NSLOT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m0 = offs < n0
    v0 = tl.load(x0 + offs, mask=m0)
    tl.store(x0 + offs, v0 * s0, mask=m0)
    if NSLOT > 1:
        m1 = offs < n1
        v1 = tl.load(x1 + offs, mask=m1)
        tl.store(x1 + offs, v1 * s1, mask=m1)
        if NSLOT > 2:
            m2 = offs < n2
            v2 = tl.load(x2 + offs, mask=m2)
            tl.store(x2 + offs, v2 * s2, mask=m2)
            if NSLOT > 3:
                m3 = offs < n3
                v3 = tl.load(x3 + offs, mask=m3)
                tl.store(x3 + offs, v3 * s3, mask=m3)


@triton.jit
def _mul_strided_kernel(
    x_ptr,
    n_elem,
    s,
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
    NDIM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    idx = offs
    off = (idx % sh0) * st0
    idx = idx // sh0
    if NDIM > 1:
        off += (idx % sh1) * st1
        idx = idx // sh1
    if NDIM > 2:
        off += (idx % sh2) * st2
        idx = idx // sh2
    if NDIM > 3:
        off += (idx % sh3) * st3
        idx = idx // sh3
    if NDIM > 4:
        off += (idx % sh4) * st4
        idx = idx // sh4
    if NDIM > 5:
        off += (idx % sh5) * st5
        idx = idx // sh5
    if NDIM > 6:
        off += (idx % sh6) * st6
        idx = idx // sh6
    if NDIM > 7:
        off += (idx % sh7) * st7
    x = tl.load(x_ptr + off, mask=mask)
    tl.store(x_ptr + off, x * s, mask=mask)


def _normalize_dims(t):
    """(sizes, strides) with dims collapsed and ordered fastest-varying first."""
    dims = [(int(sz), int(st)) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    merged = []
    for sz, st in dims:
        if merged and merged[-1][1] == sz * st:
            psz, pst = merged[-1]
            merged[-1] = (psz * sz, st)
        else:
            merged.append((sz, st))
    merged.reverse()
    return [d[0] for d in merged], [d[1] for d in merged]


def _pad(values, n, fill):
    vals = list(values[:n])
    vals.extend([fill] * (n - len(vals)))
    return vals


def _launch_strided(t, s):
    numel = t.numel()
    sizes, strides = _normalize_dims(t)
    if len(sizes) > _MAX_DIMS:
        raise RuntimeError("unsupported tensor rank for strided multiply")
    ndim = max(len(sizes), 1)
    sh = _pad(sizes, _MAX_DIMS, 1)
    st = _pad(strides, _MAX_DIMS, 0)
    block = 256 if numel <= 65536 else 1024
    grid = ((numel + block - 1) // block,)
    _mul_strided_kernel[grid](
        t,
        numel,
        s,
        sh[0],
        sh[1],
        sh[2],
        sh[3],
        sh[4],
        sh[5],
        sh[6],
        sh[7],
        st[0],
        st[1],
        st[2],
        st[3],
        st[4],
        st[5],
        st[6],
        st[7],
        BLOCK=block,
        NDIM=ndim,
        num_warps=4,
    )


def _launch_chunk(chunk):
    k = len(chunk)
    max_numel = 0
    for t, _ in chunk:
        numel = t.numel()
        if numel > max_numel:
            max_numel = numel
    block = 256 if max_numel <= 65536 else 1024
    gx = (max_numel + block - 1) // block
    ts = [c[0] for c in chunk]
    ss = [c[1] for c in chunk]
    ns = [c[0].numel() for c in chunk]
    t0 = ts[0]
    t1 = ts[1] if k > 1 else t0
    t2 = ts[2] if k > 2 else t0
    t3 = ts[3] if k > 3 else t0
    n0 = ns[0]
    n1 = ns[1] if k > 1 else 0
    n2 = ns[2] if k > 2 else 0
    n3 = ns[3] if k > 3 else 0
    s0 = ss[0]
    s1 = ss[1] if k > 1 else 0.0
    s2 = ss[2] if k > 2 else 0.0
    s3 = ss[3] if k > 3 else 0.0

    if _FAST_ENABLED and n0 != 1 and n1 != 1 and n2 != 1 and n3 != 1:
        # One precompiled binary is reused across calls. Triton specializes
        # 16-byte pointer alignment and 16-divisibility of the integer args, and
        # both are required for vectorized masked accesses, so the exact residue
        # classes are part of the key.
        key = (
            t0.dtype,
            k,
            block,
            t0.data_ptr() & 15,
            t1.data_ptr() & 15,
            t2.data_ptr() & 15,
            t3.data_ptr() & 15,
            n0 & 15,
            n1 & 15,
            n2 & 15,
            n3 & 15,
        )
        entry = _LAUNCH_CACHE.get(key, _MISS)
        if entry is _MISS:
            try:
                ck = _mul_fused_kernel.warmup(
                    t0,
                    t1,
                    t2,
                    t3,
                    n0,
                    n1,
                    n2,
                    n3,
                    s0,
                    s1,
                    s2,
                    s3,
                    BLOCK=block,
                    NSLOT=k,
                    num_warps=4,
                    grid=(1,),
                )
                ck._init_handles()
                entry = (ck.run, ck.function, ck.packed_metadata)
            except Exception:
                entry = None
            _LAUNCH_CACHE[key] = entry
        if entry is not None:
            stream = _driver.active.get_current_stream(
                _driver.active.get_current_device()
            )
            entry[0](
                gx,
                1,
                1,
                stream,
                entry[1],
                entry[2],
                None,
                _ENTER_HOOK,
                _EXIT_HOOK,
                t0,
                t1,
                t2,
                t3,
                n0,
                n1,
                n2,
                n3,
                s0,
                s1,
                s2,
                s3,
                block,
                k,
            )
            return

    _mul_fused_kernel[(gx,)](
        t0,
        t1,
        t2,
        t3,
        n0,
        n1,
        n2,
        n3,
        s0,
        s1,
        s2,
        s3,
        BLOCK=block,
        NSLOT=k,
        num_warps=4,
    )


def _foreach_mul__ScalarList(self, scalars):
    logger.debug("GEMS_HYGON _FOREACH_MUL__SCALARLIST")
    if type(self) is list:
        tensors = self
    elif isinstance(self, (list, tuple)):
        tensors = list(self)
    else:
        tensors = [self]
    n = len(tensors)

    if isinstance(scalars, (int, float)) or (
        isinstance(scalars, torch.Tensor) and scalars.dim() == 0
    ):
        sc_list = [scalars] * n
    elif isinstance(scalars, (list, tuple)):
        sc_list = scalars
    else:
        sc_list = list(scalars)
    if len(sc_list) != n:
        raise RuntimeError(
            "Tensor list and scalar list must have the same length, but got "
            f"{n} tensors and {len(sc_list)} scalars"
        )

    pairs = []
    append = pairs.append
    for i in range(n):
        t = tensors[i]
        if t.numel() == 0:
            continue
        s = sc_list[i]
        if type(s) is not float:
            if isinstance(s, torch.Tensor):
                s = s.item()
            s = float(s)
        append((t, s))

    npairs = len(pairs)
    if npairs == 0:
        return None

    # Fast path: at most _FUSE contiguous tensors of one dtype with distinct
    # storage -> a single fused launch, in list order.
    dt = pairs[0][0].dtype
    fast = npairs <= _FUSE and pairs[0][0].is_contiguous()
    if fast:
        seen = set()
        for t, _ in pairs:
            if t.dtype is not dt or not t.is_contiguous():
                fast = False
                break
            ptr = t.data_ptr()
            if ptr in seen:
                fast = False
                break
            seen.add(ptr)
    if fast:
        _launch_chunk(pairs)
        return None

    seq = False
    if npairs > 1:
        seen = set()
        for t, _ in pairs:
            ptr = t.data_ptr()
            if ptr in seen:
                seq = True
                break
            seen.add(ptr)

    if seq:
        for t, s in pairs:
            if t.is_contiguous():
                _launch_chunk([(t, s)])
            else:
                _launch_strided(t, s)
        return None

    groups = {}
    strided = []
    for t, s in pairs:
        if t.is_contiguous():
            key = (t.device, t.dtype)
            bucket = groups.get(key)
            if bucket is None:
                groups[key] = [(t, s)]
            else:
                bucket.append((t, s))
        else:
            strided.append((t, s))

    for items in groups.values():
        if len(items) > 1:
            items.sort(key=lambda p: -p[0].numel())
        for i in range(0, len(items), _FUSE):
            _launch_chunk(items[i : i + _FUSE])

    for t, s in strided:
        _launch_strided(t, s)
    return None
