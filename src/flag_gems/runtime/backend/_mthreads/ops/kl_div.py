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

"""Fused KL-divergence (ATen `kl_div`) implemented in Triton.

Semantics reproduced from ATen:
    log_target=False : loss = xlogy(target, target) - target * self
                       with xlogy(0, y) = 0
    log_target=True  : loss = exp(target) * (target - self)
    reduction 0 (none) : elementwise loss, broadcast shape
    reduction 1 (mean) : sum(loss) / numel(broadcast)
    reduction 2 (sum)  : sum(loss)
    reduction 3 (batchmean) : sum(loss) / shape[0]

All elementwise math and reduction accumulation happen in FP32; only the
final stored value is cast to the promoted output dtype.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.kl_div import kl_div as default_kl_div
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_MAX_RANK = 4

# ---------------------------------------------------------------------------
# offset helper (rank-specialized index decomposition for non-contiguous /
# broadcast inputs).  All degenerate (size-1) dims are collapsed on the host.
# ---------------------------------------------------------------------------


@triton.jit
def _kl_offsets(lin, s0, s1, s2, s3, d1, d2, d3, RANK: tl.constexpr):
    if RANK == 1:
        return lin * s0
    elif RANK == 2:
        q0 = lin // d1
        q1 = lin - q0 * d1
        return q0 * s0 + q1 * s1
    elif RANK == 3:
        q0 = lin // (d1 * d2)
        r = lin - q0 * (d1 * d2)
        q1 = r // d2
        q2 = r - q1 * d2
        return q0 * s0 + q1 * s1 + q2 * s2
    else:
        q0 = lin // (d1 * d2 * d3)
        r = lin - q0 * (d1 * d2 * d3)
        q1 = r // (d2 * d3)
        r2 = r - q1 * (d2 * d3)
        q2 = r2 // d3
        q3 = r2 - q2 * d3
        return q0 * s0 + q1 * s1 + q2 * s2 + q3 * s3


@triton.jit
def _kl_loss(s, t, log_target: tl.constexpr):
    if log_target:
        return tl.exp(t) * (t - s)
    else:
        logt = tl.log(t)
        xlogy = tl.where(t == 0.0, 0.0, t * logt)
        return xlogy - t * s


# ---------------------------------------------------------------------------
# reduction = none, contiguous flat path
# ---------------------------------------------------------------------------


@triton.jit
def _kl_none_kernel(
    s_ptr,
    t_ptr,
    o_ptr,
    n_elements,
    log_target: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    s = tl.load(s_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(t_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    loss = _kl_loss(s, t, log_target)
    tl.store(o_ptr + offs, loss, mask=mask)


# ---------------------------------------------------------------------------
# reduction = none, strided / broadcast path
# ---------------------------------------------------------------------------


@triton.jit
def _kl_none_strided_kernel(
    s_ptr,
    t_ptr,
    o_ptr,
    n_elements,
    s0,
    s1,
    s2,
    s3,
    t0,
    t1,
    t2,
    t3,
    d1,
    d2,
    d3,
    RANK: tl.constexpr,
    log_target: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    so = _kl_offsets(offs, s0, s1, s2, s3, d1, d2, d3, RANK)
    to = _kl_offsets(offs, t0, t1, t2, t3, d1, d2, d3, RANK)
    s = tl.load(s_ptr + so, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(t_ptr + to, mask=mask, other=0.0).to(tl.float32)
    loss = _kl_loss(s, t, log_target)
    tl.store(o_ptr + offs, loss, mask=mask)


# ---------------------------------------------------------------------------
# reductions (mean / sum / batchmean)
# ---------------------------------------------------------------------------


@triton.jit
def _kl_reduce_kernel(
    s_ptr,
    t_ptr,
    o_ptr,
    n_elements,
    scale,
    iters,
    n_progs,
    log_target: tl.constexpr,
    BLOCK: tl.constexpr,
    SINGLE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(0, iters):
        blk = pid + i * n_progs
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        s = tl.load(s_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(t_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        loss = _kl_loss(s, t, log_target)
        acc += tl.where(mask, loss, 0.0)
    total = tl.sum(acc, axis=0)
    if SINGLE:
        tl.store(o_ptr, total * scale)
    else:
        tl.store(o_ptr + pid, total)


# Bounded tile/warp/stage search for the partial (two-kernel) reduction.  The
# program count is fixed by the host (independent of BLOCK) so the grid-1
# finalize kernel always knows exactly how many partials to sum.
_REDUCE_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=8),
    triton.Config({"BLOCK": 2048}, num_warps=2),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=8),
    triton.Config({"BLOCK": 8192}, num_warps=4),
    triton.Config({"BLOCK": 8192}, num_warps=8),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=_REDUCE_CONFIGS, key=["BUCKET"])
@triton.jit
def _kl_reduce_partial_kernel(
    s_ptr,
    t_ptr,
    o_ptr,
    n_elements,
    log_target: tl.constexpr,
    BUCKET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_progs = tl.num_programs(axis=0)
    n_blocks = (n_elements + BLOCK - 1) // BLOCK
    iters = (n_blocks + n_progs - 1) // n_progs
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(0, iters):
        blk = pid + i * n_progs
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        s = tl.load(s_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(t_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        loss = _kl_loss(s, t, log_target)
        acc += tl.where(mask, loss, 0.0)
    tl.store(o_ptr + pid, tl.sum(acc, axis=0))


@triton.jit
def _kl_reduce_strided_kernel(
    s_ptr,
    t_ptr,
    o_ptr,
    n_elements,
    scale,
    iters,
    n_progs,
    s0,
    s1,
    s2,
    s3,
    t0,
    t1,
    t2,
    t3,
    d1,
    d2,
    d3,
    RANK: tl.constexpr,
    log_target: tl.constexpr,
    BLOCK: tl.constexpr,
    SINGLE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(0, iters):
        blk = pid + i * n_progs
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        so = _kl_offsets(offs, s0, s1, s2, s3, d1, d2, d3, RANK)
        to = _kl_offsets(offs, t0, t1, t2, t3, d1, d2, d3, RANK)
        s = tl.load(s_ptr + so, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(t_ptr + to, mask=mask, other=0.0).to(tl.float32)
        loss = _kl_loss(s, t, log_target)
        acc += tl.where(mask, loss, 0.0)
    total = tl.sum(acc, axis=0)
    if SINGLE:
        tl.store(o_ptr, total * scale)
    else:
        tl.store(o_ptr + pid, total)


@triton.jit
def _kl_finalize_kernel(part_ptr, o_ptr, n_partials, scale, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    vals = tl.load(part_ptr + offs, mask=offs < n_partials, other=0.0)
    total = tl.sum(vals, axis=0) * scale
    tl.store(o_ptr, total)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------

_REDUCTION_NAMES = {"none": 0, "mean": 1, "sum": 2, "batchmean": 3}
_NONE_BLOCK = 1024
_REDUCE_BLOCK = 4096
_REDUCE_WARPS = 8
_MAX_PROGRAMS = 480
_PROG_UNIT = 1024


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        key = reduction.strip().lower()
        if key not in _REDUCTION_NAMES:
            raise ValueError("unsupported reduction: %r" % (reduction,))
        return _REDUCTION_NAMES[key]
    if hasattr(reduction, "value"):
        try:
            return int(reduction.value)
        except (TypeError, ValueError):
            pass
    return int(reduction)


def _strided_pair(shape, s_strides, t_strides):
    """Collapse (shape, self-strides, target-strides) jointly.

    A pair of adjacent dims is merged only when BOTH operands are stride
    contiguous across them, so self and target always share the same collapsed
    rank and the same index decomposition in the kernel.
    """
    dims = []
    for sz, ss, ts in zip(shape, s_strides, t_strides):
        sz = int(sz)
        if sz == 1:
            continue
        dims.append((sz, int(ss), int(ts)))
    out = []
    for sz, ss, ts in dims:
        if out:
            psz, pss, pts = out[-1]
            if pss == sz * ss and pts == sz * ts:
                out[-1] = (psz * sz, ss, ts)
                continue
        out.append((sz, ss, ts))
    if not out:
        out = [(1, 0, 0)]
    rank = len(out)
    if rank > _MAX_RANK:
        raise RuntimeError("kl_div: symbolic rank %d exceeds supported max" % rank)
    pad = _MAX_RANK - rank
    d = [o[0] for o in out] + [1] * pad
    ss = [o[1] for o in out] + [0] * pad
    ts = [o[2] for o in out] + [0] * pad
    return rank, d, ss, ts


def _flat_ok(t):
    return t.is_contiguous()


def _kl_div_impl(self, target, reduction=1, *, log_target=False):
    if not isinstance(self, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("kl_div expects torch.Tensor arguments")

    s, t = self, target
    if s.shape != t.shape:
        s, t = torch.broadcast_tensors(s, t)

    out_dtype = torch.promote_types(s.dtype, t.dtype)
    if out_dtype == torch.float64:
        out_dtype = torch.float32
    device = s.device
    red = _normalize_reduction(reduction)
    log_target = bool(log_target)
    n = s.numel()
    shape = tuple(s.shape)

    if red == 0:
        if n == 0:
            return torch.empty(shape, dtype=out_dtype, device=device)
        out = torch.empty(shape, dtype=out_dtype, device=device)
        if _flat_ok(s) and _flat_ok(t):
            grid = ((n + _NONE_BLOCK - 1) // _NONE_BLOCK,)
            _kl_none_kernel[grid](
                s,
                t,
                out,
                n,
                log_target=log_target,
                BLOCK=_NONE_BLOCK,
                num_warps=4,
            )
        else:
            rank, dims, sstrides, tstrides = _strided_pair(
                shape, s.stride(), t.stride()
            )
            grid = ((n + _NONE_BLOCK - 1) // _NONE_BLOCK,)
            _kl_none_strided_kernel[grid](
                s,
                t,
                out,
                n,
                sstrides[0],
                sstrides[1],
                sstrides[2],
                sstrides[3],
                tstrides[0],
                tstrides[1],
                tstrides[2],
                tstrides[3],
                dims[1],
                dims[2],
                dims[3],
                RANK=rank,
                log_target=log_target,
                BLOCK=_NONE_BLOCK,
                num_warps=4,
            )
        return out

    # reduced modes ---------------------------------------------------------
    if red == 1:
        scale = 1.0 / n if n > 0 else float("nan")
    elif red == 3:
        first = shape[0] if len(shape) > 0 else 1
        scale = 1.0 / first if first > 0 else float("nan")
    else:
        scale = 1.0

    if n == 0:
        if red == 2:
            return torch.zeros((), dtype=out_dtype, device=device)
        return torch.full((), float("nan"), dtype=out_dtype, device=device)

    out = torch.empty((), dtype=out_dtype, device=device)
    flat = _flat_ok(s) and _flat_ok(t)

    if n <= _REDUCE_BLOCK:
        # single program writes the scalar directly (saves one launch)
        if flat:
            _kl_reduce_kernel[(1,)](
                s,
                t,
                out,
                n,
                scale,
                1,
                1,
                log_target=log_target,
                BLOCK=_REDUCE_BLOCK,
                SINGLE=True,
                num_warps=_REDUCE_WARPS,
            )
        else:
            rank, dims, sstrides, tstrides = _strided_pair(
                shape, s.stride(), t.stride()
            )
            _kl_reduce_strided_kernel[(1,)](
                s,
                t,
                out,
                n,
                scale,
                1,
                1,
                sstrides[0],
                sstrides[1],
                sstrides[2],
                sstrides[3],
                tstrides[0],
                tstrides[1],
                tstrides[2],
                tstrides[3],
                dims[1],
                dims[2],
                dims[3],
                RANK=rank,
                log_target=log_target,
                BLOCK=_REDUCE_BLOCK,
                SINGLE=True,
                num_warps=_REDUCE_WARPS,
            )
        return out

    n_progs = min(_MAX_PROGRAMS, (n + _PROG_UNIT - 1) // _PROG_UNIT)
    if n_progs < 1:
        n_progs = 1
    partials = torch.empty((n_progs,), dtype=torch.float32, device=device)

    if flat:
        if n <= 65536:
            bucket = 0
        elif n <= 262144:
            bucket = 1
        elif n <= 4194304:
            bucket = 2
        else:
            bucket = 3
        _kl_reduce_partial_kernel[(n_progs,)](
            s,
            t,
            partials,
            n,
            log_target=log_target,
            BUCKET=bucket,
        )
    else:
        n_blocks = (n + _REDUCE_BLOCK - 1) // _REDUCE_BLOCK
        iters = (n_blocks + n_progs - 1) // n_progs
        rank, dims, sstrides, tstrides = _strided_pair(shape, s.stride(), t.stride())
        _kl_reduce_strided_kernel[(n_progs,)](
            s,
            t,
            partials,
            n,
            scale,
            iters,
            n_progs,
            sstrides[0],
            sstrides[1],
            sstrides[2],
            sstrides[3],
            tstrides[0],
            tstrides[1],
            tstrides[2],
            tstrides[3],
            dims[1],
            dims[2],
            dims[3],
            RANK=rank,
            log_target=log_target,
            BLOCK=_REDUCE_BLOCK,
            SINGLE=False,
            num_warps=_REDUCE_WARPS,
        )

    block_mid = triton.next_power_of_2(n_progs)
    _kl_finalize_kernel[(1,)](
        partials, out, n_progs, scale, BLOCK=block_mid, num_warps=1
    )
    return out


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def kl_div(self=None, target=None, reduction=1, *, log_target=False, input=None):
    logger.debug("GEMS_MTHREADS KL_DIV")

    if self is None:
        self = input
    if self is None or target is None:
        raise TypeError("run() requires both self and target tensors")
    if not _use_triton_kernel(self) or not _use_triton_kernel(target):
        return default_kl_div(self, target, reduction, log_target=log_target)
    with torch_device_fn.device(self.device):
        return _kl_div_impl(self, target, reduction, log_target=log_target)


run = kl_div

__all__ = ["run", "kl_div"]
