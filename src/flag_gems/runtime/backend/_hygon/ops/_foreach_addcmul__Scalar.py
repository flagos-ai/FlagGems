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

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_RANK = 8
_FUSED_MAX_K = 16
_FUSED_MAX_ELEMS = 1 << 22

# Every tensor-count parameter of the fused kernel is passed at runtime, so the
# compiled launcher signature must never depend on its value.
_FUSED_COUNT_NAMES = ["n%d" % i for i in range(_FUSED_MAX_K)]


@triton.jit
def _cm_body(a_ptr, b_ptr, c_ptr, base, n, lane, value, BLOCK: tl.constexpr):
    offs = base * BLOCK + lane
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    c = tl.load(c_ptr + offs, mask=mask, other=0.0)
    res = a.to(tl.float32) + value * b.to(tl.float32) * c.to(tl.float32)
    tl.store(a_ptr + offs, res.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _addcmul_flat_kernel(a_ptr, b_ptr, c_ptr, n, value, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    _cm_body(a_ptr, b_ptr, c_ptr, pid, n, tl.arange(0, BLOCK), value, BLOCK)


@triton.jit(do_not_specialize=_FUSED_COUNT_NAMES)
def _addcmul_fused_kernel(
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
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    s1 = (n0 + BLOCK - 1) // BLOCK
    s2 = s1 + (n1 + BLOCK - 1) // BLOCK
    s3 = s2 + (n2 + BLOCK - 1) // BLOCK
    s4 = s3 + (n3 + BLOCK - 1) // BLOCK
    s5 = s4 + (n4 + BLOCK - 1) // BLOCK
    s6 = s5 + (n5 + BLOCK - 1) // BLOCK
    s7 = s6 + (n6 + BLOCK - 1) // BLOCK
    s8 = s7 + (n7 + BLOCK - 1) // BLOCK
    s9 = s8 + (n8 + BLOCK - 1) // BLOCK
    s10 = s9 + (n9 + BLOCK - 1) // BLOCK
    s11 = s10 + (n10 + BLOCK - 1) // BLOCK
    s12 = s11 + (n11 + BLOCK - 1) // BLOCK
    s13 = s12 + (n12 + BLOCK - 1) // BLOCK
    s14 = s13 + (n13 + BLOCK - 1) // BLOCK
    s15 = s14 + (n14 + BLOCK - 1) // BLOCK
    if pid < s1:
        _cm_body(a0, b0, c0, pid, n0, lane, value, BLOCK)
    elif pid < s2:
        _cm_body(a1, b1, c1, pid - s1, n1, lane, value, BLOCK)
    elif pid < s3:
        _cm_body(a2, b2, c2, pid - s2, n2, lane, value, BLOCK)
    elif pid < s4:
        _cm_body(a3, b3, c3, pid - s3, n3, lane, value, BLOCK)
    elif pid < s5:
        _cm_body(a4, b4, c4, pid - s4, n4, lane, value, BLOCK)
    elif pid < s6:
        _cm_body(a5, b5, c5, pid - s5, n5, lane, value, BLOCK)
    elif pid < s7:
        _cm_body(a6, b6, c6, pid - s6, n6, lane, value, BLOCK)
    elif pid < s8:
        _cm_body(a7, b7, c7, pid - s7, n7, lane, value, BLOCK)
    elif pid < s9:
        _cm_body(a8, b8, c8, pid - s8, n8, lane, value, BLOCK)
    elif pid < s10:
        _cm_body(a9, b9, c9, pid - s9, n9, lane, value, BLOCK)
    elif pid < s11:
        _cm_body(a10, b10, c10, pid - s10, n10, lane, value, BLOCK)
    elif pid < s12:
        _cm_body(a11, b11, c11, pid - s11, n11, lane, value, BLOCK)
    elif pid < s13:
        _cm_body(a12, b12, c12, pid - s12, n12, lane, value, BLOCK)
    elif pid < s14:
        _cm_body(a13, b13, c13, pid - s13, n13, lane, value, BLOCK)
    elif pid < s15:
        _cm_body(a14, b14, c14, pid - s14, n14, lane, value, BLOCK)
    else:
        _cm_body(a15, b15, c15, pid - s15, n15, lane, value, BLOCK)


@triton.jit
def _addcmul_strided_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n,
    value,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
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
    sc0,
    sc1,
    sc2,
    sc3,
    sc4,
    sc5,
    sc6,
    sc7,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    oa = tl.zeros([BLOCK], dtype=tl.int64)
    ob = tl.zeros([BLOCK], dtype=tl.int64)
    oc = tl.zeros([BLOCK], dtype=tl.int64)
    rem = offs

    i = rem % d7
    rem = rem // d7
    oa += i * sa7
    ob += i * sb7
    oc += i * sc7
    i = rem % d6
    rem = rem // d6
    oa += i * sa6
    ob += i * sb6
    oc += i * sc6
    i = rem % d5
    rem = rem // d5
    oa += i * sa5
    ob += i * sb5
    oc += i * sc5
    i = rem % d4
    rem = rem // d4
    oa += i * sa4
    ob += i * sb4
    oc += i * sc4
    i = rem % d3
    rem = rem // d3
    oa += i * sa3
    ob += i * sb3
    oc += i * sc3
    i = rem % d2
    rem = rem // d2
    oa += i * sa2
    ob += i * sb2
    oc += i * sc2
    i = rem % d1
    rem = rem // d1
    oa += i * sa1
    ob += i * sb1
    oc += i * sc1
    i = rem % d0
    oa += i * sa0
    ob += i * sb0
    oc += i * sc0

    a = tl.load(a_ptr + oa, mask=mask, other=0.0)
    b = tl.load(b_ptr + ob, mask=mask, other=0.0)
    c = tl.load(c_ptr + oc, mask=mask, other=0.0)
    res = a.to(tl.float32) + value * b.to(tl.float32) * c.to(tl.float32)
    tl.store(a_ptr + oa, res.to(a_ptr.dtype.element_ty), mask=mask)


# Compiled fused kernel per (dtype, device); the launcher is independent of the
# tensor counts and element shapes because all of them are runtime arguments.
_FUSED_KERNELS = {}


def _fused_kernel(dtype, device):
    key = (dtype, device)
    ck = _FUSED_KERNELS.get(key)
    if ck is None:
        mock = torch.empty(0, dtype=dtype, device=device)
        wargs = [mock] * (3 * _FUSED_MAX_K) + [1] * _FUSED_MAX_K + [1.0]
        ck = _addcmul_fused_kernel.warmup(
            *wargs, grid=(1,), BLOCK=_BLOCK, num_warps=_NUM_WARPS
        )
        _FUSED_KERNELS[key] = ck
    return ck


def _as_list(x):
    if isinstance(x, torch.Tensor):
        return [x]
    return list(x)


def _pad_rank(x, rank, fill=0):
    vals = list(x) + [fill] * (rank - len(x))
    return vals


def _foreach_addcmul__Scalar(self, tensor1, tensor2, value=1):
    logger.debug("GEMS_HYGON _FOREACH_ADDCMUL__SCALAR")
    if isinstance(value, torch.Tensor):
        value = value.item()
    v = float(value)

    selfs = _as_list(self)
    t1s = _as_list(tensor1)
    t2s = _as_list(tensor2)

    K = len(selfs)
    if len(t1s) != K or len(t2s) != K:
        raise RuntimeError(
            "_foreach_addcmul_: expects the same number of tensors in each list"
        )

    # ---- fused single-launch path for short lists (launch-latency dominated)
    if 2 <= K <= _FUSED_MAX_K:
        dtype = selfs[0].dtype
        total = 0
        blocks = 0
        ns = []
        args = []
        ext = args.extend
        fast = True
        for a, b, c in zip(selfs, t1s, t2s):
            if (
                a.dtype != dtype
                or b.dtype != dtype
                or c.dtype != dtype
                or a.shape != b.shape
                or a.shape != c.shape
                or not a.is_contiguous()
                or not b.is_contiguous()
                or not c.is_contiguous()
            ):
                fast = False
                break
            n = a.numel()
            if n == 0:
                fast = False
                break
            blocks += (n + _BLOCK - 1) // _BLOCK
            total += n
            ns.append(n)
            ext((a, b, c))
            if total > _FUSED_MAX_ELEMS:
                fast = False
                break
        if fast:
            da, db, dc = selfs[0], t1s[0], t2s[0]
            ext((da, db, dc) * (_FUSED_MAX_K - K))
            ext(ns)
            ext((0,) * (_FUSED_MAX_K - K))
            args.append(v)
            runner = _fused_kernel(dtype, da.device)[(blocks, 1, 1)]
            # This backend forwards constexpr values to the generated launcher
            # as trailing positional arguments (BLOCK is the last parameter).
            runner(*args, _BLOCK)
            return None

    # ---- per-tensor fallback (long lists, large lists, strided/broadcast)
    for a, b, c in zip(selfs, t1s, t2s):
        n = a.numel()
        if n == 0:
            continue
        shape = a.shape
        flat = (
            a.is_contiguous()
            and b.shape == shape
            and c.shape == shape
            and b.is_contiguous()
            and c.is_contiguous()
        )
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        if flat:
            _addcmul_flat_kernel[grid](
                a, b, c, n, v, BLOCK=_BLOCK, num_warps=_NUM_WARPS
            )
        else:
            if b.shape != shape:
                b = b.expand(shape)
            if c.shape != shape:
                c = c.expand(shape)
            rank = len(shape)
            if rank == 0:
                _addcmul_flat_kernel[(1,)](
                    a, b, c, 1, v, BLOCK=_BLOCK, num_warps=_NUM_WARPS
                )
                continue
            if rank > _MAX_RANK:
                raise NotImplementedError(
                    "_foreach_addcmul_: rank > %d not supported" % _MAX_RANK
                )
            dims = _pad_rank(list(shape), _MAX_RANK, 1)
            sa = _pad_rank([int(s) for s in a.stride()], _MAX_RANK)
            sb = _pad_rank([int(s) for s in b.stride()], _MAX_RANK)
            sc = _pad_rank([int(s) for s in c.stride()], _MAX_RANK)
            _addcmul_strided_kernel[grid](
                a, b, c, n, v, *dims, *sa, *sb, *sc, BLOCK=_BLOCK, num_warps=_NUM_WARPS
            )
    return None
