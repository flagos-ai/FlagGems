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

"""Triton implementation of aten::_foreach_add.Scalar.

For each tensor T_i in the input sequence and the scalar s:
    out_i = T_i + s     (elementwise, out-of-place, fresh storage)

Strategy
--------
* Uniform group (all tensors contiguous, same dtype, same device): outputs are
  allocated with empty_like and the whole list is covered by ONE Triton launch
  whose 1-D grid spans the concatenation of per-tensor block ranges.  Each
  tensor's range is padded to whole BLOCKs so a program never straddles two
  tensors; the slot is selected from program_id(0) against cumulative block
  boundaries (scalar kernel arguments, no metadata tensor / H2D copy), and each
  branch loads through its own kernel-argument pointer so alignment
  specialisation survives.  Lists longer than 8 are chunked.
* Non-contiguous tensors: host-collapsed (shape, stride) with rank as a
  constexpr and a contiguous freshly allocated output.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import (
    _foreach_add_Scalar as default_foreach_add_Scalar,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_BLOCK = 1024
_BLOCK_BIG = 2048
# Below this many elements in a group the call is launch-latency bound, so the
# smaller tile (more blocks, more SMs touched) is kept.
_BIG_ELEMS = 1 << 22
_MAXF = 8

_INTEGRAL_DTYPES = (
    torch.bool,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
)


@triton.jit
def _add_contig_kernel(in_ptr, out_ptr, n, s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(in_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, v + s, mask=mask)


@triton.jit
def _add_fused_kernel(
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    alpha,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < s1:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n0
        v = tl.load(i0 + offs, mask=m, other=0)
        tl.store(o0 + offs, v + alpha, mask=m, cache_modifier=".cs")
    elif pid < s2:
        offs = (pid - s1) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n1
        v = tl.load(i1 + offs, mask=m, other=0)
        tl.store(o1 + offs, v + alpha, mask=m, cache_modifier=".cs")
    elif pid < s3:
        offs = (pid - s2) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n2
        v = tl.load(i2 + offs, mask=m, other=0)
        tl.store(o2 + offs, v + alpha, mask=m, cache_modifier=".cs")
    elif pid < s4:
        offs = (pid - s3) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n3
        v = tl.load(i3 + offs, mask=m, other=0)
        tl.store(o3 + offs, v + alpha, mask=m, cache_modifier=".cs")
    elif pid < s5:
        offs = (pid - s4) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n4
        v = tl.load(i4 + offs, mask=m, other=0)
        tl.store(o4 + offs, v + alpha, mask=m, cache_modifier=".cs")
    elif pid < s6:
        offs = (pid - s5) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n5
        v = tl.load(i5 + offs, mask=m, other=0)
        tl.store(o5 + offs, v + alpha, mask=m, cache_modifier=".cs")
    elif pid < s7:
        offs = (pid - s6) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n6
        v = tl.load(i6 + offs, mask=m, other=0)
        tl.store(o6 + offs, v + alpha, mask=m, cache_modifier=".cs")
    else:
        offs = (pid - s7) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n7
        v = tl.load(i7 + offs, mask=m, other=0)
        tl.store(o7 + offs, v + alpha, mask=m, cache_modifier=".cs")


@triton.jit
def _add_strided_kernel(
    in_ptr,
    out_ptr,
    n,
    s,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx.to(tl.int64)
    off = rem * 0
    if RANK >= 1:
        off = off + (rem % d0) * t0
        rem = rem // d0
    if RANK >= 2:
        off = off + (rem % d1) * t1
        rem = rem // d1
    if RANK >= 3:
        off = off + (rem % d2) * t2
        rem = rem // d2
    if RANK >= 4:
        off = off + (rem % d3) * t3
        rem = rem // d3
    if RANK >= 5:
        off = off + (rem % d4) * t4
        rem = rem // d4
    if RANK >= 6:
        off = off + (rem % d5) * t5
        rem = rem // d5
    if RANK >= 7:
        off = off + (rem % d6) * t6
        rem = rem // d6
    if RANK >= 8:
        off = off + (rem % d7) * t7
        rem = rem // d7
    v = tl.load(in_ptr + off, mask=mask, other=0)
    tl.store(out_ptr + idx, v + s, mask=mask)


def _collapse(shape, strides):
    """Drop size-1 dims and merge adjacent dims that are contiguous with each other."""
    dims = [(int(sz), int(st)) for sz, st in zip(shape, strides) if sz != 1]
    out = []
    for sz, st in dims:
        if out and out[-1][1] == st * sz:
            psz, pst = out[-1]
            out[-1] = (psz * sz, pst)
        else:
            out.append((sz, st))
    # Row-major linear index decomposes fastest-varying dim first.
    return out[::-1]


def _launch_strided(t, out, scalar):
    n = t.numel()
    dims = _collapse(t.shape, t.stride())
    rank = len(dims)
    if rank == 0:
        # Degenerate: every dim has size 1, so numel == 1 and offset is 0.
        _add_contig_kernel[(1,)](t, out, n, scalar, BLOCK=_BLOCK)
        return
    ds = [d for d, _ in dims]
    ts = [st for _, st in dims]
    ds += [1] * (_MAXF - rank)
    ts += [0] * (_MAXF - rank)
    _add_strided_kernel[(triton.cdiv(n, _BLOCK),)](
        t, out, n, scalar, *ds, *ts, RANK=rank, BLOCK=_BLOCK
    )


def _launch_fused(tensors, outs, idxs, scalar):
    """One launch covering every tensor in idxs (same dtype/device, contiguous)."""
    ins = []
    ots = []
    ns = []
    tot = 0
    elems = 0
    for i in idxs:
        t = tensors[i]
        n = t.numel()
        ins.append(t)
        ots.append(outs[i])
        ns.append(n)
        elems += n
    # The bigger tile only pays off once the group is well past the
    # launch-latency regime; small groups keep the smaller tile.
    block = _BLOCK_BIG if elems >= _BIG_ELEMS else _BLOCK
    starts = []
    for n in ns:
        starts.append(tot)
        tot += (n + block - 1) // block
    k = len(idxs)
    if tot == 0:
        return
    pad = _MAXF - k
    if pad:
        ins += [ins[-1]] * pad
        ots += [ots[-1]] * pad
        ns += [0] * pad
        starts += [tot] * pad
    _add_fused_kernel[(tot,)](*ins, *ots, *ns, *starts, scalar, BLOCK=block)


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _list_supported(seq):
    """A foreach call is specialized only if every tensor in it qualifies."""
    if seq is None:
        return True
    if isinstance(seq, torch.Tensor):
        return _use_triton_kernel(seq)
    for t in seq:
        if isinstance(t, torch.Tensor) and not _use_triton_kernel(t):
            return False
    return True


def _foreach_add_Scalar(self, scalar):
    logger.debug("GEMS_MTHREADS _FOREACH_ADD_SCALAR")

    if not _list_supported(self):
        return default_foreach_add_Scalar(self, scalar)

    with torch_device_fn.device(
        self[0].device
        if isinstance(self, (list, tuple)) and self
        else torch.device("musa")
    ):
        return _launch_foreach_add_Scalar(self, scalar)


def _launch_foreach_add_Scalar(self, scalar):
    tensors = self if type(self) is list else list(self)
    nt = len(tensors)
    if nt == 0:
        return []

    if isinstance(scalar, torch.Tensor):
        raise TypeError("tensor scalar is not supported by the Scalar overload")
    if isinstance(scalar, bool):
        scalar = int(scalar)
    if not isinstance(scalar, int):
        for t in tensors:
            if t.dtype in _INTEGRAL_DTYPES:
                raise RuntimeError(
                    "result type Float can't be cast to the desired output type"
                )

    outs = [None] * nt
    t0 = tensors[0]
    dtype0 = t0.dtype
    dev0 = t0.device

    uniform = True
    for t in tensors:
        if t.dtype is not dtype0 or t.device != dev0 or not t.is_contiguous():
            uniform = False
            break

    if uniform:
        for i in range(nt):
            outs[i] = torch.empty_like(tensors[i])
        if nt == 1:
            n = tensors[0].numel()
            if n:
                _add_contig_kernel[(triton.cdiv(n, _BLOCK),)](
                    tensors[0], outs[0], n, scalar, BLOCK=_BLOCK
                )
            return outs
        for c in range(0, nt, _MAXF):
            _launch_fused(tensors, outs, range(c, min(c + _MAXF, nt)), scalar)
        return outs

    groups = {}
    for i, t in enumerate(tensors):
        groups.setdefault((t.device, t.dtype, t.is_contiguous()), []).append(i)

    for (dev, dtype, contig), idxs in groups.items():
        if contig:
            for i in idxs:
                outs[i] = torch.empty(tensors[i].shape, dtype=dtype, device=dev)
            for c in range(0, len(idxs), _MAXF):
                chunk = idxs[c : c + _MAXF]
                if len(chunk) == 1:
                    i = chunk[0]
                    n = tensors[i].numel()
                    if n:
                        _add_contig_kernel[(triton.cdiv(n, _BLOCK),)](
                            tensors[i], outs[i], n, scalar, BLOCK=_BLOCK
                        )
                else:
                    _launch_fused(tensors, outs, chunk, scalar)
        else:
            for i in idxs:
                t = tensors[i]
                out = torch.empty(t.shape, dtype=dtype, device=dev)
                outs[i] = out
                if t.numel():
                    _launch_strided(t, out, scalar)
    return outs
