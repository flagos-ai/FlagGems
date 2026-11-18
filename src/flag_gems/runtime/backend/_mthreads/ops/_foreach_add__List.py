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

"""Triton implementation of aten::_foreach_add_ (in-place, list overloads)."""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import (
    _foreach_add__List as default_foreach_add__List,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_BLOCK = 1024
_MAX_RANK = 8
_MAX_GROUP = 8
_SLOTS4 = 4

_DEV = getattr(torch, "musa", None)
if _DEV is None:
    _DEV = torch.cuda

try:
    from triton import knobs as _TRITON_KNOBS
    from triton.runtime import driver as _TRITON_DRIVER
except Exception:
    _TRITON_DRIVER = None
    _TRITON_KNOBS = None

if _TRITON_KNOBS is not None:
    try:
        _ENTER_HOOK = _TRITON_KNOBS.runtime.launch_enter_hook
        _LEAVE_HOOK = _TRITON_KNOBS.runtime.launch_exit_hook
    except Exception:
        _ENTER_HOOK = None
        _LEAVE_HOOK = None
    try:
        # skip building per-call launch metadata when no hooks are registered
        _enter_calls = getattr(_ENTER_HOOK, "calls", None)
        _leave_calls = getattr(_LEAVE_HOOK, "calls", None)
        if _enter_calls is None or _leave_calls is None:
            _MD_NONE = False
        else:
            _MD_NONE = len(_enter_calls) == 0 and len(_leave_calls) == 0
    except Exception:
        _MD_NONE = False
else:
    _ENTER_HOOK = None
    _LEAVE_HOOK = None
    _MD_NONE = False

# Cached launch plans for the fused multi-tensor path.  A plan holds only a
# compiled kernel handle plus launch constants; tensor values are never cached
# and every call re-reads its own arguments.
_PLAN_CACHE = {}
_STATS = {"fast": 0, "slow": 0}
_ZEROS = {k: (0,) * k for k in (0, 1, 2, 3, 4)}
_FAST_OK = [True]
_CTX_CACHE = {}


@triton.jit
def _add_body(a_ptr, b_ptr, offs, mask, alpha, FP32: tl.constexpr):
    a = tl.load(a_ptr + offs, mask=mask, other=0)
    b = tl.load(b_ptr + offs, mask=mask, other=0)
    if FP32:
        res = a.to(tl.float32) + alpha * b.to(tl.float32)
    else:
        res = a + (alpha * b).to(a_ptr.dtype.element_ty)
    tl.store(a_ptr + offs, res.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _add_flat_kernel(
    a_ptr, b_ptr, n_elem, alpha, FP32: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    _add_body(a_ptr, b_ptr, offs, mask, alpha, FP32)


@triton.jit
def _add_multi4_flat_kernel(
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    b2,
    b3,
    n_elem,
    alpha,
    FP32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    if t == 0:
        _add_body(a0, b0, offs, mask, alpha, FP32)
    elif t == 1:
        _add_body(a1, b1, offs, mask, alpha, FP32)
    elif t == 2:
        _add_body(a2, b2, offs, mask, alpha, FP32)
    else:
        _add_body(a3, b3, offs, mask, alpha, FP32)


@triton.jit
def _add_multi_flat_kernel(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    n_elem,
    alpha,
    FP32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    if t == 0:
        _add_body(a0, b0, offs, mask, alpha, FP32)
    elif t == 1:
        _add_body(a1, b1, offs, mask, alpha, FP32)
    elif t == 2:
        _add_body(a2, b2, offs, mask, alpha, FP32)
    elif t == 3:
        _add_body(a3, b3, offs, mask, alpha, FP32)
    elif t == 4:
        _add_body(a4, b4, offs, mask, alpha, FP32)
    elif t == 5:
        _add_body(a5, b5, offs, mask, alpha, FP32)
    elif t == 6:
        _add_body(a6, b6, offs, mask, alpha, FP32)
    else:
        _add_body(a7, b7, offs, mask, alpha, FP32)


@triton.jit
def _add_scalar_kernel(
    a_ptr, b_val, n_elem, alpha, FP32: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    a = tl.load(a_ptr + offs, mask=mask, other=0)
    if FP32:
        res = a.to(tl.float32) + alpha * b_val
    else:
        res = a + (alpha * b_val).to(a_ptr.dtype.element_ty)
    tl.store(a_ptr + offs, res.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _add_strided_kernel(
    a_ptr,
    b_ptr,
    n_elem,
    alpha,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    sa0,
    sa1,
    sa2,
    sa3,
    sa4,
    sa5,
    sa6,
    sa7,
    sb0,
    sb1,
    sb2,
    sb3,
    sb4,
    sb5,
    sb6,
    sb7,
    RANK: tl.constexpr,
    FP32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    shp = (s0, s1, s2, s3, s4, s5, s6, s7)
    sa = (sa0, sa1, sa2, sa3, sa4, sa5, sa6, sa7)
    sb = (sb0, sb1, sb2, sb3, sb4, sb5, sb6, sb7)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    idx = offs
    a_off = tl.zeros((BLOCK,), dtype=tl.int64)
    b_off = tl.zeros((BLOCK,), dtype=tl.int64)
    for i in tl.static_range(RANK):
        sz = shp[i]
        c = idx % sz
        idx = idx // sz
        c64 = c.to(tl.int64)
        a_off += c64 * sa[i]
        b_off += c64 * sb[i]
    a = tl.load(a_ptr + a_off, mask=mask, other=0)
    b = tl.load(b_ptr + b_off, mask=mask, other=0)
    if FP32:
        res = a.to(tl.float32) + alpha * b.to(tl.float32)
    else:
        res = a + (alpha * b).to(a_ptr.dtype.element_ty)
    tl.store(a_ptr + a_off, res.to(a_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------- host helpers

_HAS_RUN = hasattr(_add_multi_flat_kernel, "run")


def _is_seq(x):
    return isinstance(x, (list, tuple))


def _as_float(v):
    if isinstance(v, torch.Tensor):
        return float(v.item())
    return float(v)


def _aligned_strides(b, shape):
    """Strides of ``b`` aligned (right) to ``shape`` with 0 for broadcast dims."""
    rank = len(shape)
    if b.dim() > rank:
        raise RuntimeError(
            "_foreach_add_: the size of tensor list a (dim %d) must be greater "
            "than or equal to the size of tensor list b (dim %d)" % (rank, b.dim())
        )
    pad = rank - b.dim()
    bshape = (1,) * pad + tuple(b.shape)
    bstride = (0,) * pad + tuple(b.stride())
    out = []
    for s, bs, st in zip(shape, bshape, bstride):
        if bs == s:
            out.append(int(st))
        elif bs == 1:
            out.append(0)
        else:
            raise RuntimeError(
                "_foreach_add_: shapes %s and %s are not broadcastable"
                % (tuple(shape), tuple(b.shape))
            )
    return out


def _maybe_device_ctx(idx):
    if idx is None or idx <= 0:
        return None
    ctx = _CTX_CACHE.get(idx, 0)
    if ctx == 0:
        try:
            ctx = None if _DEV.current_device() == idx else _DEV.device(idx)
        except Exception:
            return None
        _CTX_CACHE[idx] = ctx
    return ctx


# --------------------------------------------------------- fused uniform launch


def _plan_launch(plan, args):
    """Launch a cached CompiledKernel; False means use the JIT launch instead."""
    fn, md_fn, function, packed, blocks, nchunk, stream, tails = plan
    md = None if _MD_NONE else md_fn((blocks, nchunk, 1), stream, *args)
    if tails is None:
        # This backend's launcher appends a backend-specific number of trailing
        # slots after the kernel arguments; a wrong count fails inside
        # PyArg_ParseTuple, i.e. before any kernel starts.
        for k in (0, 1, 2, 3, 4):
            try:
                fn(
                    blocks,
                    nchunk,
                    1,
                    stream,
                    function,
                    packed,
                    md,
                    _ENTER_HOOK,
                    _LEAVE_HOOK,
                    *args,
                    *_ZEROS[k],
                )
            except TypeError:
                continue
            plan[7] = _ZEROS[k]
            return True
        plan[7] = False
        return False
    try:
        fn(
            blocks,
            nchunk,
            1,
            stream,
            function,
            packed,
            md,
            _ENTER_HOOK,
            _LEAVE_HOOK,
            *args,
            *tails,
        )
        return True
    except TypeError:
        return False


def _launch_uniform(selfs, others, nel, al, is_fp, n):
    """Single fused launch for a small uniform group of contiguous pairs."""
    key = (selfs[0].dtype, nel, n)
    if nel >= 65536:
        block = 4096
        warps = 8
    else:
        block = _BLOCK
        warps = 2 if nel <= 2048 else 4
    blocks = (nel + block - 1) // block
    if n < _SLOTS4:
        ap = list(selfs) + [selfs[0]] * (_SLOTS4 - n)
        bp = list(others) + [others[0]] * (_SLOTS4 - n)
        args = tuple(ap) + tuple(bp) + (nel, al)
    else:
        args = tuple(selfs) + tuple(others) + (nel, al)
    plan = _PLAN_CACHE.get(key)
    if plan is not None:
        if plan[7] is not False and _FAST_OK[0] and _plan_launch(plan, args):
            _STATS["fast"] += 1
            return
        plan[7] = False
    elif _HAS_RUN and _TRITON_DRIVER is not None:
        try:
            stream = _TRITON_DRIVER.active.get_current_stream(selfs[0].get_device())
        except Exception:
            stream = None
        if stream is not None:
            try:
                ck = _add_multi4_flat_kernel.run(
                    *args,
                    grid=(blocks, n),
                    warmup=False,
                    FP32=is_fp,
                    BLOCK=block,
                    num_warps=warps,
                )
            except TypeError:
                ck = None
            if ck is not None:
                # run() above already performed this call's in-place update.
                _PLAN_CACHE[key] = [
                    ck.run,
                    ck.launch_metadata,
                    ck.function,
                    ck.packed_metadata,
                    blocks,
                    n,
                    stream,
                    None,
                ]
                _STATS["slow"] += 1
                return
    _add_multi4_flat_kernel[(blocks, n)](
        *args, FP32=is_fp, BLOCK=block, num_warps=warps
    )
    _STATS["slow"] += 1


# ------------------------------------------------------------- general paths


def _launch_pair(a, b, alpha):
    n = a.numel()
    if n == 0:
        return
    is_fp = a.dtype.is_floating_point
    block = _BLOCK
    grid = ((n + block - 1) // block,)
    warps = 2 if n <= 2048 else 4
    if a.is_contiguous() and b.is_contiguous() and b.numel() == n:
        _add_flat_kernel[grid](a, b, n, alpha, FP32=is_fp, BLOCK=block, num_warps=warps)
        return
    shape = tuple(a.shape)
    rank = len(shape)
    if rank == 0:
        _add_flat_kernel[grid](a, b, n, alpha, FP32=is_fp, BLOCK=block, num_warps=warps)
        return
    if rank > _MAX_RANK:
        raise RuntimeError(
            "_foreach_add_: rank %d exceeds supported %d" % (rank, _MAX_RANK)
        )
    a_strides = [int(s) for s in a.stride()]
    b_strides = _aligned_strides(b, shape)
    shape_r = tuple(shape[::-1]) + (1,) * (_MAX_RANK - rank)
    a_r = a_strides[::-1] + [0] * (_MAX_RANK - rank)
    b_r = b_strides[::-1] + [0] * (_MAX_RANK - rank)
    _add_strided_kernel[grid](
        a,
        b,
        n,
        alpha,
        shape_r[0],
        shape_r[1],
        shape_r[2],
        shape_r[3],
        shape_r[4],
        shape_r[5],
        shape_r[6],
        shape_r[7],
        a_r[0],
        a_r[1],
        a_r[2],
        a_r[3],
        a_r[4],
        a_r[5],
        a_r[6],
        a_r[7],
        b_r[0],
        b_r[1],
        b_r[2],
        b_r[3],
        b_r[4],
        b_r[5],
        b_r[6],
        b_r[7],
        RANK=rank,
        FP32=is_fp,
        BLOCK=block,
        num_warps=warps,
    )


def _launch_scalar(a, val, alpha):
    n = a.numel()
    if n == 0:
        return
    is_fp = a.dtype.is_floating_point
    block = _BLOCK
    grid = ((n + block - 1) // block,)
    warps = 2 if n <= 2048 else 4
    if a.is_contiguous():
        _add_scalar_kernel[grid](
            a, val, n, alpha, FP32=is_fp, BLOCK=block, num_warps=warps
        )
        return
    shape = tuple(a.shape)
    rank = len(shape)
    if rank == 0:
        _add_scalar_kernel[grid](
            a, val, n, alpha, FP32=is_fp, BLOCK=block, num_warps=warps
        )
        return
    if rank > _MAX_RANK:
        raise RuntimeError(
            "_foreach_add_: rank %d exceeds supported %d" % (rank, _MAX_RANK)
        )
    a_strides = [int(s) for s in a.stride()]
    shape_r = tuple(shape[::-1]) + (1,) * (_MAX_RANK - rank)
    a_r = a_strides[::-1] + [0] * (_MAX_RANK - rank)
    b_r = [0] * _MAX_RANK
    _add_strided_kernel[grid](
        a,
        a,
        n,
        alpha,
        shape_r[0],
        shape_r[1],
        shape_r[2],
        shape_r[3],
        shape_r[4],
        shape_r[5],
        shape_r[6],
        shape_r[7],
        a_r[0],
        a_r[1],
        a_r[2],
        a_r[3],
        a_r[4],
        a_r[5],
        a_r[6],
        a_r[7],
        b_r[0],
        b_r[1],
        b_r[2],
        b_r[3],
        b_r[4],
        b_r[5],
        b_r[6],
        b_r[7],
        RANK=rank,
        FP32=is_fp,
        BLOCK=block,
        num_warps=warps,
    )


def _launch_group(members, nel, alpha, is_fp):
    block = _BLOCK
    blocks = (nel + block - 1) // block
    warps = 2 if nel <= 2048 else 4
    nmem = len(members)
    if nmem == 1:
        a, b = members[0]
        _add_flat_kernel[(blocks,)](
            a, b, nel, alpha, FP32=is_fp, BLOCK=block, num_warps=warps
        )
        return
    for start in range(0, nmem, _MAX_GROUP):
        chunk = members[start : start + _MAX_GROUP]
        nchunk = len(chunk)
        if nchunk < _MAX_GROUP:
            chunk = chunk + [chunk[-1]] * (_MAX_GROUP - nchunk)
        ap = [m[0] for m in chunk]
        bp = [m[1] for m in chunk]
        _add_multi_flat_kernel[(blocks, nchunk)](
            ap[0],
            ap[1],
            ap[2],
            ap[3],
            ap[4],
            ap[5],
            ap[6],
            ap[7],
            bp[0],
            bp[1],
            bp[2],
            bp[3],
            bp[4],
            bp[5],
            bp[6],
            bp[7],
            nel,
            alpha,
            FP32=is_fp,
            BLOCK=block,
            num_warps=warps,
        )


def _dispatch(selfs, others, scalar_other, alphas):
    n = len(selfs)
    if others is None:
        for i in range(n):
            _launch_scalar(selfs[i], scalar_other, alphas[i])
        return

    groups = {}
    order = []
    for i in range(n):
        a = selfs[i]
        b = others[i]
        al = alphas[i]
        nel = a.numel()
        if nel == 0:
            continue
        if a.is_contiguous() and b.is_contiguous() and b.numel() == nel:
            key = (a.dtype, nel, al)
            g = groups.get(key)
            if g is None:
                groups[key] = [a, b]
                order.append(key)
            else:
                g.append(a)
                g.append(b)
        else:
            _launch_pair(a, b, al)

    for key in order:
        dtype, nel, al = key
        pair = groups[key]
        members = [(pair[j], pair[j + 1]) for j in range(0, len(pair), 2)]
        _launch_group(members, nel, al, dtype.is_floating_point)


def _run_general(selfs, others, scalar_other, alphas):
    if others is not None:
        for i in range(len(selfs)):
            a = selfs[i]
            b = others[i]
            if b.numel() != a.numel():
                _aligned_strides(b, tuple(a.shape))
    ctx = _maybe_device_ctx(selfs[0].get_device())
    if ctx is None:
        _dispatch(selfs, others, scalar_other, alphas)
    else:
        with ctx:
            _dispatch(selfs, others, scalar_other, alphas)


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


def _foreach_add__List(self, other, alpha=1):
    logger.debug("GEMS_MTHREADS _FOREACH_ADD__LIST")

    if not _list_supported(self):
        return default_foreach_add__List(self, other, alpha)

    with torch_device_fn.device(
        self[0].device
        if isinstance(self, (list, tuple)) and self
        else torch.device("musa")
    ):
        return _launch_foreach_add__List(self, other, alpha)


def _launch_foreach_add__List(self, other, alpha=1):
    if type(self) is list:
        selfs = self
    else:
        selfs = list(self) if _is_seq(self) else [self]
    n = len(selfs)
    if n == 0:
        return None

    if _is_seq(other):
        others = other if type(other) is list else list(other)
        if len(others) != n:
            raise RuntimeError(
                "_foreach_add_: expected the same number of tensors in self and "
                "other, but got %d and %d" % (n, len(others))
            )
        scalar_other = None
    else:
        others = None
        if isinstance(other, torch.Tensor):
            if other.dim() != 0:
                raise TypeError(
                    "_foreach_add_: other must be a list of tensors or a scalar"
                )
            scalar_other = float(other.item())
        else:
            scalar_other = float(other)

    if not _is_seq(alpha):
        al = alpha if type(alpha) is float else _as_float(alpha)
        if others is not None and n == _SLOTS4:
            a0 = selfs[0]
            nel = a0.numel()
            if nel:
                dt = a0.dtype
                ok = True
                for i in range(n):
                    a = selfs[i]
                    b = others[i]
                    if (
                        a.dtype is not dt
                        or b.dtype is not dt
                        or a.numel() != nel
                        or b.numel() != nel
                        or not a.is_contiguous()
                        or not b.is_contiguous()
                    ):
                        ok = False
                        break
                if ok:
                    ctx = _maybe_device_ctx(a0.get_device())
                    if ctx is None:
                        _launch_uniform(selfs, others, nel, al, dt.is_floating_point, n)
                    else:
                        with ctx:
                            _launch_uniform(
                                selfs, others, nel, al, dt.is_floating_point, n
                            )
                    return None
        _run_general(selfs, others, scalar_other, [al] * n)
        return None

    alphas = list(alpha)
    if len(alphas) != n:
        raise RuntimeError(
            "_foreach_add_: alpha list has %d elements but self list has %d"
            % (len(alphas), n)
        )
    _run_general(selfs, others, scalar_other, [_as_float(x) for x in alphas])
    return None
