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

"""Elementwise true division (aten::true_divide.Tensor) as a Triton op.

Target: MTT S5000 (MUSA, flagtree-triton).

Semantics reproduced from ATen's div_true kernel:
  * integral/bool operands are promoted to the default float dtype,
  * float16/bfloat16 are computed in fp32 opmath and rounded once on store,
  * one IEEE round-to-nearest division per element (no epsilon guard, no
    reciprocal rewrite), so x/0 -> inf and 0/0 -> nan fall out naturally.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.true_divide import true_divide as default_true_divide
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

logger = logging.getLogger(__name__)

_MAX_RANK = 8
_F64 = tl.float64
_F32 = tl.float32


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------
@triton.jit
def _td_div_flat(
    ap,
    bp,
    op,
    n,
    BLOCK: tl.constexpr,
    CDTYPE: tl.constexpr,
    PRECISE: tl.constexpr,
    AMODE: tl.constexpr,
    BMODE: tl.constexpr,
):
    """No-index-math kernel: every operand is contiguous & same-shaped.

    AMODE/BMODE: 0 = vector load, 1 = single element broadcast, 2 = host scalar.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if AMODE == 2:
        av = ap
    elif AMODE == 1:
        av = tl.load(ap)
    else:
        av = tl.load(ap + offs, mask=mask, other=0)
    if BMODE == 2:
        bv = bp
    elif BMODE == 1:
        bv = tl.load(bp)
    else:
        bv = tl.load(bp + offs, mask=mask, other=0)
    av = av.to(CDTYPE)
    bv = bv.to(CDTYPE)
    if PRECISE:
        r = tl.fdiv(av, bv, ieee_rounding=True)
    else:
        r = av / bv
    tl.store(op + offs, r, mask=mask)


@triton.jit
def _td_div_strided(
    ap,
    bp,
    op,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    as0,
    as1,
    as2,
    as3,
    as4,
    as5,
    as6,
    as7,
    bs0,
    bs1,
    bs2,
    bs3,
    bs4,
    bs5,
    bs6,
    bs7,
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    CDTYPE: tl.constexpr,
    PRECISE: tl.constexpr,
    AMODE: tl.constexpr,
    BMODE: tl.constexpr,
    OUT_C: tl.constexpr,
):
    """General broadcast kernel: flat output index decomposed per dimension.

    Slot k holds the size/stride of the (RANK-1-k)-th dimension, i.e. the
    fastest-varying dimension is slot 0.  Broadcast dims carry stride 0.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    t = offs
    ai = tl.zeros([BLOCK], dtype=tl.int32)
    bi = tl.zeros([BLOCK], dtype=tl.int32)
    oi = tl.zeros([BLOCK], dtype=tl.int32)
    if RANK > 0:
        i = t % d0
        t = t // d0
        ai += i * as0
        bi += i * bs0
        if not OUT_C:
            oi += i * os0
    if RANK > 1:
        i = t % d1
        t = t // d1
        ai += i * as1
        bi += i * bs1
        if not OUT_C:
            oi += i * os1
    if RANK > 2:
        i = t % d2
        t = t // d2
        ai += i * as2
        bi += i * bs2
        if not OUT_C:
            oi += i * os2
    if RANK > 3:
        i = t % d3
        t = t // d3
        ai += i * as3
        bi += i * bs3
        if not OUT_C:
            oi += i * os3
    if RANK > 4:
        i = t % d4
        t = t // d4
        ai += i * as4
        bi += i * bs4
        if not OUT_C:
            oi += i * os4
    if RANK > 5:
        i = t % d5
        t = t // d5
        ai += i * as5
        bi += i * bs5
        if not OUT_C:
            oi += i * os5
    if RANK > 6:
        i = t % d6
        t = t // d6
        ai += i * as6
        bi += i * bs6
        if not OUT_C:
            oi += i * os6
    if RANK > 7:
        i = t
        ai += i * as7
        bi += i * bs7
        if not OUT_C:
            oi += i * os7
    if OUT_C:
        oi = offs
    if AMODE == 2:
        av = ap
    else:
        av = tl.load(ap + ai, mask=mask, other=0)
    if BMODE == 2:
        bv = bp
    else:
        bv = tl.load(bp + bi, mask=mask, other=0)
    av = av.to(CDTYPE)
    bv = bv.to(CDTYPE)
    if PRECISE:
        r = tl.fdiv(av, bv, ieee_rounding=True)
    else:
        r = av / bv
    tl.store(op + oi, r, mask=mask)


# --------------------------------------------------------------------------
# Host helpers (metadata / dispatch only)
# --------------------------------------------------------------------------
def _broadcast_shape(s1, s2):
    n1 = len(s1)
    n2 = len(s2)
    r = n1 if n1 >= n2 else n2
    out = [1] * r
    for i in range(r):
        d1 = s1[n1 - 1 - i] if i < n1 else 1
        d2 = s2[n2 - 1 - i] if i < n2 else 1
        if d1 == d2:
            out[r - 1 - i] = d1
        elif d1 == 1:
            out[r - 1 - i] = d2
        elif d2 == 1:
            out[r - 1 - i] = d1
        else:
            raise RuntimeError(
                "The size of tensor a (%s) must match the size of tensor b (%s)"
                % (tuple(s1), tuple(s2))
            )
    return tuple(out)


def _expand_strides(shape, strides, bshape):
    r = len(bshape)
    ns = len(shape)
    out = [0] * r
    base = r - ns
    for i in range(r):
        j = i - base
        if j >= 0 and shape[j] != 1:
            out[i] = strides[j]
    return out


def _cfg(n):
    if n <= 2048:
        return 256, 4
    if n <= (1 << 20):
        return 512, 4
    return 1024, 4


# Bounded launch-configuration search for the bandwidth-bound sizes: BLOCK and
# num_warps are swept jointly so per-lane accesses stay 128-bit while the work
# per CTA / per thread grows.  It is only used on the pure out-of-place path
# (freshly allocated output, vector/vector operands), so the repeated benchmark
# launches are harmless, and it is gated away from the launch-latency-dominated
# sizes where the tuner's host overhead would dominate the measurement.
_TUNE_N = 1 << 20
_TUNE_PAIRS = (
    (1024, 4),
    (2048, 4),
    (4096, 4),
    (2048, 8),
    (4096, 8),
    (4096, 16),
    (8192, 8),
    (8192, 16),
)
_TUNE_CONFIGS = [triton.Config({"BLOCK": b}, num_warps=w) for b, w in _TUNE_PAIRS]
_td_div_flat_auto = triton.autotune(
    configs=_TUNE_CONFIGS, key=["n"], warmup=10, rep=40
)(_td_div_flat)

# Shape/dtype-scoped cache of the launch configuration the tuner selected, so
# steady-state calls skip the tuner's host-side key bookkeeping.
_CFG_CACHE = {}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def true_divide(self, other, *args, **kwargs):
    logger.debug("GEMS_MTHREADS TRUE_DIVIDE")

    if not _use_triton_kernel(self) or (
        isinstance(other, torch.Tensor) and not _use_triton_kernel(other)
    ):
        return default_true_divide(self, other, *args, **kwargs)

    with torch_device_fn.device(self.device):
        return _launch_true_divide(self, other, *args, **kwargs)


def _launch_true_divide(self, other, *args, **kwargs):
    logger.debug("GEMS_MTHREADS TRUE_DIVIDE")
    out = kwargs.pop("out", None)
    if args:
        if len(args) != 1 or out is not None:
            raise TypeError("run() takes at most 3 positional arguments")
        out = args[0]
    if kwargs:
        raise TypeError("run() got unexpected keyword arguments %s" % (list(kwargs),))
    return _true_divide(self, other, out)


def _true_divide(a, b, out):
    a_is = isinstance(a, torch.Tensor)
    b_is = isinstance(b, torch.Tensor)
    if a_is:
        ref = a
    elif b_is:
        ref = b
    else:
        raise TypeError("true_divide expects at least one Tensor argument")

    # Result dtype: ATen promotes integral/bool operands to the default float.
    dt = torch.result_type(a, b)
    if not dt.is_floating_point:
        dt = torch.get_default_dtype()
    cdtype = _F64 if dt == torch.float64 else _F32

    if a_is and b_is:
        bshape = _broadcast_shape(a.shape, b.shape)
    elif a_is:
        bshape = tuple(a.shape)
    else:
        bshape = tuple(b.shape)

    if out is None:
        allocated = True
        if a_is and a.shape == bshape and a.is_contiguous():
            out = torch.empty_like(a, dtype=dt)
        else:
            out = torch.empty(bshape, dtype=dt, device=ref.device)
    else:
        allocated = False

    n = out.numel()
    if n == 0:
        return out

    amode = 2 if not a_is else (1 if a.numel() == 1 else 0)
    bmode = 2 if not b_is else (1 if b.numel() == 1 else 0)

    flat = out.is_contiguous()
    if flat and amode == 0 and not (a.shape == bshape and a.is_contiguous()):
        flat = False
    if flat and bmode == 0 and not (b.shape == bshape and b.is_contiguous()):
        flat = False

    if flat:
        if allocated and amode == 0 and bmode == 0 and n >= _TUNE_N:
            cfg = _CFG_CACHE.get((n, a.dtype, b.dtype))
            if cfg is None:
                _td_div_flat_auto[
                    lambda meta: ((n + meta["BLOCK"] - 1) // meta["BLOCK"],)
                ](a, b, out, n, CDTYPE=cdtype, PRECISE=True, AMODE=0, BMODE=0)
                try:
                    bc = _td_div_flat_auto.best_config
                    if bc is not None:
                        _CFG_CACHE[(n, a.dtype, b.dtype)] = (
                            bc.kwargs["BLOCK"],
                            bc.num_warps,
                        )
                except AttributeError:
                    pass
                return out
            block, nw = cfg
            _td_div_flat[((n + block - 1) // block,)](
                a,
                b,
                out,
                n,
                BLOCK=block,
                CDTYPE=cdtype,
                PRECISE=True,
                AMODE=0,
                BMODE=0,
                num_warps=nw,
            )
            return out
        block, nw = _cfg(n)
        _td_div_flat[((n + block - 1) // block,)](
            a,
            b,
            out,
            n,
            BLOCK=block,
            CDTYPE=cdtype,
            PRECISE=True,
            AMODE=amode,
            BMODE=bmode,
            num_warps=nw,
        )
        return out

    block, nw = _cfg(n)
    grid = ((n + block - 1) // block,)

    rank = len(bshape)
    if rank > _MAX_RANK:
        raise RuntimeError("true_divide supports at most %d dimensions" % _MAX_RANK)

    if a_is:
        ast = _expand_strides(a.shape, a.stride(), bshape)
    else:
        ast = [0] * rank
    if b_is:
        bst = _expand_strides(b.shape, b.stride(), bshape)
    else:
        bst = [0] * rank
    ost = _expand_strides(out.shape, out.stride(), bshape)

    d = [1] * _MAX_RANK
    sa = [0] * _MAX_RANK
    sb = [0] * _MAX_RANK
    so = [0] * _MAX_RANK
    for k in range(_MAX_RANK):
        if k < rank:
            dim = rank - 1 - k
            d[k] = bshape[dim]
            sa[k] = ast[dim]
            sb[k] = bst[dim]
            so[k] = ost[dim]

    _td_div_strided[grid](
        a,
        b,
        out,
        n,
        d[0],
        d[1],
        d[2],
        d[3],
        d[4],
        d[5],
        d[6],
        d[7],
        sa[0],
        sa[1],
        sa[2],
        sa[3],
        sa[4],
        sa[5],
        sa[6],
        sa[7],
        sb[0],
        sb[1],
        sb[2],
        sb[3],
        sb[4],
        sb[5],
        sb[6],
        sb[7],
        so[0],
        so[1],
        so[2],
        so[3],
        so[4],
        so[5],
        so[6],
        so[7],
        RANK=rank,
        BLOCK=block,
        CDTYPE=cdtype,
        PRECISE=True,
        AMODE=amode,
        BMODE=bmode,
        OUT_C=out.is_contiguous(),
        num_warps=nw,
    )
    return out
