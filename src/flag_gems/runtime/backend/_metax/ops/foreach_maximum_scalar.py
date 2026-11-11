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

import torch
import triton
import triton.language as tl

_MAX_RANK = 8
_BLOCK = 1024
_NUM_WARPS = 4
# Fused short-list path: a small block over many CTAs keeps the per-CTA work
# near the kernel launch floor while each thread still moves one 16-byte vector.
_SMALL_BLOCK = 256
_SMALL_WARPS = 1
# Lists no longer than this are fused into a single launch: below it the
# per-call host cost (launch + allocation) dominates the measured latency,
# above it the kernel is purely DRAM bound and the per-tensor path is kept.
_FUSE_MAX_ELEMS = 1 << 22


# Bounded per-size search for the streaming per-tensor kernel: 4 to 16
# elements per thread (BLOCK / num_warps) to vary bytes in flight per thread.
# The tuner runs during warmup, never inside the timed steady state.
_CONTIG_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=8),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=8),
    triton.Config({"BLOCK": 8192}, num_warps=8),
]

_DEV_KEY = None


def _device_key():
    """Runtime device-family key so a winner is never reused across devices."""
    global _DEV_KEY
    if _DEV_KEY is None:
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        _DEV_KEY = (hash(p.name) & 0xFFFF) | (int(p.major) << 20) | (int(p.minor) << 24)
    return _DEV_KEY


@triton.jit
def _sel_store(x_ptr, o_ptr, n, offs, s):
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    thresh = s.to(x.dtype)
    y = tl.where(x != x, x, tl.where(x > thresh, x, thresh))
    tl.store(o_ptr + offs, y, mask=mask)


@triton.autotune(configs=_CONTIG_CONFIGS, key=["dev_key", "n"])
@triton.jit
def _max_scalar_contig_kernel(
    x_ptr, out_ptr, n, s, dev_key, USE_I64: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    if USE_I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    _sel_store(x_ptr, out_ptr, n, offs, s)


@triton.jit
def _max_scalar_fused4_kernel(
    x0, o0, n0, x1, o1, n1, x2, o2, n2, x3, o3, n3, s, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    _sel_store(x0, o0, n0, offs, s)
    _sel_store(x1, o1, n1, offs, s)
    _sel_store(x2, o2, n2, offs, s)
    _sel_store(x3, o3, n3, offs, s)


@triton.jit
def _max_scalar_strided_kernel(
    x_ptr,
    out_ptr,
    n,
    s,
    z0,
    z1,
    z2,
    z3,
    z4,
    z5,
    z6,
    z7,
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    ix0,
    ix1,
    ix2,
    ix3,
    ix4,
    ix5,
    ix6,
    ix7,
    ox0,
    ox1,
    ox2,
    ox3,
    ox4,
    ox5,
    ox6,
    ox7,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    f = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = f < n
    xi = (f // v0 % z0) * ix0
    xi += (f // v1 % z1) * ix1
    xi += (f // v2 % z2) * ix2
    xi += (f // v3 % z3) * ix3
    xi += (f // v4 % z4) * ix4
    xi += (f // v5 % z5) * ix5
    xi += (f // v6 % z6) * ix6
    xi += (f // v7 % z7) * ix7
    xo = (f // v0 % z0) * ox0
    xo += (f // v1 % z1) * ox1
    xo += (f // v2 % z2) * ox2
    xo += (f // v3 % z3) * ox3
    xo += (f // v4 % z4) * ox4
    xo += (f // v5 % z5) * ox5
    xo += (f // v6 % z6) * ox6
    xo += (f // v7 % z7) * ox7
    x = tl.load(x_ptr + xi, mask=mask, other=0)
    thresh = s.to(x.dtype)
    y = tl.where(x != x, x, tl.where(x > thresh, x, thresh))
    tl.store(out_ptr + xo, y, mask=mask)


def _strided_args(t, out):
    shape = list(t.shape)
    rank = len(shape)
    if rank > _MAX_RANK:
        return None
    if rank == 0:
        ishape = [1] * _MAX_RANK
        istride = [0] * _MAX_RANK
        ostride = [0] * _MAX_RANK
    else:
        ishape = shape + [1] * (_MAX_RANK - rank)
        istride = list(t.stride()) + [0] * (_MAX_RANK - rank)
        ostride = list(out.stride()) + [0] * (_MAX_RANK - rank)
    suffix = [1] * _MAX_RANK
    acc = 1
    for d in range(_MAX_RANK - 1, -1, -1):
        suffix[d] = acc
        acc *= ishape[d]
    return ishape, suffix, istride, ostride


def foreach_maximum_scalar(self, scalar, *args, **kwargs):
    tensors = self
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    n_tensors = len(tensors)
    if n_tensors == 0:
        return []
    if isinstance(scalar, torch.Tensor):
        scalar = scalar.item()
    s = float(scalar)

    if n_tensors == 4:
        t0, t1, t2, t3 = tensors
        d = t0.dtype
        n0 = t0.numel()
        n1 = t1.numel()
        n2 = t2.numel()
        n3 = t3.numel()
        total = n0 + n1 + n2 + n3
        if (
            t1.dtype is d
            and t2.dtype is d
            and t3.dtype is d
            and 0 < total <= _FUSE_MAX_ELEMS
            and t0.is_contiguous()
            and t1.is_contiguous()
            and t2.is_contiguous()
            and t3.is_contiguous()
            and t1.device == t0.device
            and t2.device == t0.device
            and t3.device == t0.device
        ):
            dev = t0.device
            sh = t0.shape
            if t1.shape == sh and t2.shape == sh and t3.shape == sh:
                # One allocation, one view call: the four disjoint contiguous
                # views share a fresh buffer that never aliases any input.
                buf = torch.empty((4,) + tuple(sh), dtype=d, device=dev)
                o0, o1, o2, o3 = buf.unbind(0)
            else:
                o0 = torch.empty(sh, dtype=d, device=dev)
                o1 = torch.empty(t1.shape, dtype=d, device=dev)
                o2 = torch.empty(t2.shape, dtype=d, device=dev)
                o3 = torch.empty(t3.shape, dtype=d, device=dev)
            mx = max(n0, max(n1, max(n2, n3)))
            _max_scalar_fused4_kernel[((mx + _SMALL_BLOCK - 1) >> 8,)](
                t0,
                o0,
                n0,
                t1,
                o1,
                n1,
                t2,
                o2,
                n2,
                t3,
                o3,
                n3,
                s,
                _SMALL_BLOCK,
                num_warps=_SMALL_WARPS,
            )
            return [o0, o1, o2, o3]

    outs = []
    for t in tensors:
        out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
        outs.append(out)
        n = t.numel()
        if n == 0:
            continue
        if t.is_contiguous():
            use_i64 = n > 2**31 - 1
            dk = _device_key()
            grid = lambda meta: ((n + meta["BLOCK"] - 1) // meta["BLOCK"],)
            _max_scalar_contig_kernel[grid](t, out, n, s, dk, use_i64)
        else:
            args_ok = _strided_args(t, out)
            if args_ok is None:
                raise NotImplementedError("rank too large for strided kernel")
            ishape, suffix, istride, ostride = args_ok
            grid = ((n + _BLOCK - 1) // _BLOCK,)
            _max_scalar_strided_kernel[grid](
                t,
                out,
                n,
                s,
                *ishape,
                *suffix,
                *istride,
                *ostride,
                _BLOCK,
                num_warps=_NUM_WARPS,
            )
    return outs
