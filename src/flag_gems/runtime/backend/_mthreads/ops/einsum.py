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

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)
# ---------------------------------------------------------------------------
# Plan cache.  Keyed on the equation text plus every operand's shape, strides
# and dtype, so no address metadata can go stale when an input changes.
# ---------------------------------------------------------------------------
_CACHE = {}
_VIEW = {}
_MISS = object()

_DTYPE_RANK = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
}


def _promote(dtypes):
    best = None
    for d in dtypes:
        r = _DTYPE_RANK.get(d)
        if r is None:
            return torch.float32
        if best is None or r > _DTYPE_RANK[best]:
            best = d
    return best if best is not None else torch.float32


def _split_terms(s):
    out = []
    for t in s.split(","):
        if "..." in t:
            a, b = t.split("...")
            if "..." in b:
                raise ValueError("einsum: multiple ellipses in one operand")
            out.append((a, b))
        else:
            out.append((t, None))
    return out


class _Parsed(object):
    __slots__ = (
        "per_operand",
        "sizes",
        "out_labels",
        "reduced",
        "out_shape",
        "numel_out",
        "out_div",
        "red_div",
        "eff",
    )


def _parse(equation, operands):
    s = equation.replace(" ", "").replace("\n", "").replace("…", "...")
    if "->" in s:
        lhs, rhs = s.split("->")
        explicit = True
    else:
        lhs, rhs = s, None
        explicit = False

    terms = _split_terms(lhs)
    if len(terms) != len(operands):
        raise ValueError(
            "einsum: number of operands (%d) does not match equation (%d)"
            % (len(operands), len(terms))
        )

    ell = []
    maxe = 0
    for i, (pre, post) in enumerate(terms):
        nd = operands[i].dim()
        e = nd - len(pre) - (0 if post is None else len(post))
        if e < 0:
            raise ValueError("einsum: operand rank smaller than subscripts")
        ell.append(e)
        if e > maxe:
            maxe = e

    per_operand = []
    for i, (pre, post) in enumerate(terms):
        e = ell[i]
        base = maxe - e
        labs = [c for c in pre]
        labs += ["*%d" % (base + j) for j in range(e)]
        if post is not None:
            labs += [c for c in post]
        per_operand.append(labs)

    sizes = {}
    for i, labs in enumerate(per_operand):
        shp = operands[i].shape
        for ax, lab in enumerate(labs):
            d = shp[ax]
            if lab in sizes:
                cur = sizes[lab]
                if cur != d:
                    if d == 1:
                        continue
                    if cur == 1:
                        sizes[lab] = d
                    else:
                        raise ValueError("einsum: dim size mismatch for %r" % lab)
            else:
                sizes[lab] = d

    count = {}
    for labs in per_operand:
        for lab in labs:
            count[lab] = count.get(lab, 0) + 1

    if explicit:
        rterms = _split_terms(rhs)
        if len(rterms) != 1:
            raise ValueError("einsum: output subscripts must be a single term")
        pre, post = rterms[0]
        if post is None:
            if maxe:
                raise ValueError("einsum: output is missing ellipsis")
            out_labels = [c for c in pre]
        else:
            out_labels = [c for c in pre]
            out_labels += ["*%d" % j for j in range(maxe)]
            out_labels += [c for c in post]
    else:
        out_labels = ["*%d" % j for j in range(maxe)]
        out_labels += sorted(
            c for c in count if not c.startswith("*") and count[c] == 1
        )

    seen = []
    for labs in per_operand:
        for lab in labs:
            if lab not in seen:
                seen.append(lab)

    outset = set(out_labels)
    if len(outset) != len(out_labels):
        raise ValueError("einsum: repeated output subscript")
    reduced = [lab for lab in seen if lab not in outset]

    p = _Parsed()
    p.per_operand = per_operand
    p.sizes = sizes
    p.out_labels = out_labels
    p.reduced = reduced
    p.out_shape = tuple(sizes.get(lab, 1) for lab in out_labels)
    n = 1
    for d in p.out_shape:
        n *= int(d)
    p.numel_out = n

    out_div = [1] * len(out_labels)
    acc = 1
    for i in range(len(out_labels) - 1, -1, -1):
        out_div[i] = acc
        acc *= int(p.out_shape[i])
    p.out_div = out_div

    red_div = [1] * len(reduced)
    acc = 1
    for i in range(len(reduced) - 1, -1, -1):
        red_div[i] = acc
        acc *= int(sizes.get(reduced[i], 1))
    p.red_div = red_div

    eff = []
    for i, labs in enumerate(per_operand):
        st = operands[i].stride()
        shp = operands[i].shape
        d = {}
        for ax, lab in enumerate(labs):
            s_ax = st[ax]
            if shp[ax] == 1 and sizes.get(lab, 1) != 1:
                s_ax = 0
            d[lab] = d.get(lab, 0) + s_ax
        eff.append(d)
    p.eff = eff
    return p


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@triton.jit
def _k_elem(
    out_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    meta_ptr,
    numel,
    BLOCK: tl.constexpr,
    NL: tl.constexpr,
    NOP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    acc = tl.full([BLOCK], 1.0, tl.float32)
    for p in tl.static_range(NOP):
        if p == 0:
            ptr = a_ptr
        elif p == 1:
            ptr = b_ptr
        elif p == 2:
            ptr = c_ptr
        else:
            ptr = d_ptr
        off = tl.zeros([BLOCK], tl.int32)
        for lab in tl.static_range(NL):
            dim = tl.load(meta_ptr + lab)
            div = tl.load(meta_ptr + NL + lab)
            st = tl.load(meta_ptr + 2 * NL + p * NL + lab)
            off += ((offs // div) % dim) * st
        v = tl.load(ptr + off, mask=mask, other=1.0)
        acc = acc * v.to(tl.float32)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _k_red_scalar(
    out_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    meta_ptr,
    k_total,
    k_per,
    BK: tl.constexpr,
    NLR: tl.constexpr,
    NOP: tl.constexpr,
):
    pid = tl.program_id(0)
    k_lo = pid * k_per
    k_hi = tl.minimum(k_lo + k_per, k_total)
    offs = tl.arange(0, BK)
    acc = tl.zeros([BK], tl.float32)
    for k0 in range(k_lo, k_hi, BK):
        kk = k0 + offs
        k_mask = kk < k_hi
        prod = tl.full([BK], 1.0, tl.float32)
        for p in tl.static_range(NOP):
            if p == 0:
                ptr = a_ptr
            elif p == 1:
                ptr = b_ptr
            elif p == 2:
                ptr = c_ptr
            else:
                ptr = d_ptr
            koff = tl.zeros([BK], tl.int32)
            for lab in tl.static_range(NLR):
                dim = tl.load(meta_ptr + lab)
                div = tl.load(meta_ptr + NLR + lab)
                st = tl.load(meta_ptr + 2 * NLR + p * NLR + lab)
                koff += ((kk // div) % dim) * st
            v = tl.load(ptr + koff, mask=k_mask, other=1.0)
            prod = prod * v.to(tl.float32)
        acc += tl.where(k_mask, prod, 0.0)
    tl.store(out_ptr + pid, tl.sum(acc).to(out_ptr.dtype.element_ty))


@triton.jit
def _k_red_scalar_aff(
    out_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    k_total,
    k_per,
    BK: tl.constexpr,
    NOP: tl.constexpr,
):
    """Scalar reduction whose reduced axes are a contiguous block in every
    operand, so the address of element k is simply base + k.  Expressing it as
    `ptr + kk` lets the compiler emit wide vector loads instead of a computed
    per-lane gather."""
    pid = tl.program_id(0)
    k_lo = pid * k_per
    k_hi = tl.minimum(k_lo + k_per, k_total)
    offs = tl.arange(0, BK)
    acc = tl.zeros([BK], tl.float32)
    for k0 in range(k_lo, k_hi, BK):
        kk = k0 + offs
        k_mask = kk < k_hi
        prod = tl.full([BK], 1.0, tl.float32)
        for p in tl.static_range(NOP):
            if p == 0:
                ptr = a_ptr
            elif p == 1:
                ptr = b_ptr
            elif p == 2:
                ptr = c_ptr
            else:
                ptr = d_ptr
            v = tl.load(ptr + kk, mask=k_mask, other=1.0)
            prod = prod * v.to(tl.float32)
        acc += tl.where(k_mask, prod, 0.0)
    tl.store(out_ptr + pid, tl.sum(acc).to(out_ptr.dtype.element_ty))


@triton.jit
def _ooff_of(meta_ptr, offs_o, NLO: tl.constexpr, P: tl.constexpr, BO: tl.constexpr):
    """Output-side offset vector of one operand (independent of k)."""
    ooff = tl.zeros([BO], tl.int32)
    for lab in tl.static_range(NLO):
        dim = tl.load(meta_ptr + lab)
        div = tl.load(meta_ptr + NLO + lab)
        st = tl.load(meta_ptr + 2 * NLO + P * NLO + lab)
        ooff += ((offs_o // div) % dim) * st
    return ooff


@triton.jit
def _k_red(
    out_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    meta_ptr,
    numel_out,
    k_total,
    k_per,
    BO: tl.constexpr,
    BK: tl.constexpr,
    NLO: tl.constexpr,
    NLR: tl.constexpr,
    NOP: tl.constexpr,
):
    pid_o = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_o = pid_o * BO + tl.arange(0, BO)
    o_mask = offs_o < numel_out
    k_lo = pid_k * k_per
    k_hi = tl.minimum(k_lo + k_per, k_total)
    offs_k = tl.arange(0, BK)
    rbase = 2 * NLO + NOP * NLO
    # The output-side offsets do not depend on k, so build them once.
    ooff0 = _ooff_of(meta_ptr, offs_o, NLO, 0, BO)
    if NOP > 1:
        ooff1 = _ooff_of(meta_ptr, offs_o, NLO, 1, BO)
    if NOP > 2:
        ooff2 = _ooff_of(meta_ptr, offs_o, NLO, 2, BO)
    if NOP > 3:
        ooff3 = _ooff_of(meta_ptr, offs_o, NLO, 3, BO)
    acc = tl.zeros([BO], tl.float32)
    for k0 in range(k_lo, k_hi, BK):
        kk = k0 + offs_k
        k_mask = kk < k_hi
        prod = tl.full([BO, BK], 1.0, tl.float32)
        for p in tl.static_range(NOP):
            if p == 0:
                ptr = a_ptr
                ooff = ooff0
            elif p == 1:
                ptr = b_ptr
                ooff = ooff1
            elif p == 2:
                ptr = c_ptr
                ooff = ooff2
            else:
                ptr = d_ptr
                ooff = ooff3
            koff = tl.zeros([BK], tl.int32)
            for lab in tl.static_range(NLR):
                dim = tl.load(meta_ptr + rbase + lab)
                div = tl.load(meta_ptr + rbase + NLR + lab)
                st = tl.load(meta_ptr + rbase + 2 * NLR + p * NLR + lab)
                koff += ((kk // div) % dim) * st
            v = tl.load(
                ptr + ooff[:, None] + koff[None, :],
                mask=o_mask[:, None] & k_mask[None, :],
                other=1.0,
            )
            prod = prod * v.to(tl.float32)
        acc += tl.sum(tl.where(k_mask[None, :], prod, 0.0), axis=1)
    tl.store(
        out_ptr + pid_k * numel_out + offs_o,
        acc.to(out_ptr.dtype.element_ty),
        mask=o_mask,
    )


@triton.jit
def _k_final(out_ptr, part_ptr, numel_out, nk, BO: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_o = pid * BO + tl.arange(0, BO)
    o_mask = offs_o < numel_out
    offs_k = tl.arange(0, BK)
    acc = tl.zeros([BO], tl.float32)
    for k0 in range(0, nk, BK):
        kk = k0 + offs_k
        m = (kk < nk)[:, None] & o_mask[None, :]
        v = tl.load(
            part_ptr + kk[:, None] * numel_out + offs_o[None, :], mask=m, other=0.0
        )
        acc += tl.sum(v, axis=0)
    tl.store(out_ptr + offs_o, acc.to(out_ptr.dtype.element_ty), mask=o_mask)


@triton.jit
def _k_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    sa_m,
    sa_k,
    sb_k,
    sb_n,
    sc_m,
    sc_n,
    a_b0,
    a_b1,
    b_b0,
    b_b1,
    c_b0,
    c_b1,
    bd1,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NB: tl.constexpr,
    GM: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid = tl.program_id(1)
    num_pid_m = (M + BM - 1) // BM
    num_pid_n = (N + BN - 1) // BN
    if GM > 1:
        num_in_group = GM * num_pid_n
        gid = pid // num_in_group
        first_m = gid * GM
        gsize = min(num_pid_m - first_m, GM)
        pid_m = first_m + ((pid % num_in_group) % gsize)
        pid_n = (pid % num_in_group) // gsize
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    if NB == 0:
        off_a = 0
        off_b = 0
        off_c = 0
    elif NB == 1:
        off_a = pid_b * a_b0
        off_b = pid_b * b_b0
        off_c = pid_b * c_b0
    else:
        i0 = pid_b // bd1
        i1 = pid_b - i0 * bd1
        off_a = i0 * a_b0 + i1 * a_b1
        off_b = i0 * b_b0 + i1 * b_b1
        off_c = i0 * c_b0 + i1 * c_b1

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + off_a + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
    b_ptrs = b_ptr + off_b + offs_k[:, None] * sb_k + offs_n[None, :] * sb_n
    acc = tl.zeros([BM, BN], tl.float32)
    if EVEN:
        # Every extent is an exact multiple of the tile, so neither the inner
        # loop nor the epilogue needs predication: unpredicated loads can use
        # wide vector accesses and the store needs no mask.
        for k0 in range(0, K // BK):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, input_precision="ieee")
            a_ptrs += BK * sa_k
            b_ptrs += BK * sb_k
        tl.store(
            c_ptr + off_c + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n,
            acc.to(c_ptr.dtype.element_ty),
        )
    else:
        m_mask = offs_m < M
        n_mask = offs_n < N
        for k0 in range(0, (K + BK - 1) // BK):
            k_mask = (k0 * BK + offs_k) < K
            a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            acc = tl.dot(a, b, acc, input_precision="ieee")
            a_ptrs += BK * sa_k
            b_ptrs += BK * sb_k
        tl.store(
            c_ptr + off_c + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n,
            acc.to(c_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------
def _meta(vals, dev):
    return torch.tensor(vals, dtype=torch.int32, device=dev)


def _align(n, gran):
    """Round n up to a multiple of gran (>= 1)."""
    g = gran if gran > 0 else 1
    return ((int(n) + g - 1) // g) * g


def _pad(ops):
    n = len(ops)
    if n == 1:
        return ops[0], ops[0], ops[0], ops[0]
    if n == 2:
        return ops[0], ops[1], ops[1], ops[1]
    if n == 3:
        return ops[0], ops[1], ops[2], ops[2]
    return ops[0], ops[1], ops[2], ops[3]


def _elem_entry(p, ops, out_shape, out_dtype, dev):
    nop = len(ops)
    out_labels = p.out_labels
    nl = len(out_labels)
    vals = list(p.out_shape)
    vals += list(p.out_div)
    for q in range(nop):
        e = p.eff[q]
        vals += [int(e.get(lab, 0)) for lab in out_labels]
    meta = _meta(vals, dev)
    numel = p.numel_out
    block = 1024
    grid = (triton.cdiv(numel, block),)

    def launch(operands):
        out = torch.empty(out_shape, dtype=out_dtype, device=dev)
        a, b, c, d = _pad(operands)
        _k_elem[grid](
            out, a, b, c, d, meta, numel, BLOCK=block, NL=nl, NOP=nop, num_warps=4
        )
        return out

    return launch


def _red_entry(p, ops, out_shape, out_dtype, dev):
    nop = len(ops)
    reduced = p.reduced
    nlr = len(reduced)
    k_total = 1
    for lab in reduced:
        k_total *= int(p.sizes.get(lab, 1))
    out_labels = p.out_labels
    nlo = len(out_labels)
    numel = p.numel_out

    # Whether every operand's reduced axes form one contiguous block laid out
    # in exactly the reduced-label order.  When that holds the gather index is
    # simply k, which lets the compiler emit wide vector loads instead of a
    # per-lane computed gather.
    affine = True
    for q in range(nop):
        e = p.eff[q]
        for lab in reduced:
            if lab not in e or int(e[lab]) != int(p.red_div[reduced.index(lab)]):
                affine = False
                break
        if not affine:
            break

    if numel == 1:
        bk = 1024
        nk = min(1024, max(240, triton.cdiv(k_total, 4 * bk)))
        nk = max(1, min(nk, triton.cdiv(k_total, bk)))
        k_per = _align(triton.cdiv(k_total, nk), 16)
        nk = max(1, triton.cdiv(k_total, k_per))
        grid = (nk,)
        direct = nk == 1
        fbo = 2
        fbk = min(1024, max(2, triton.next_power_of_2(nk)))
        meta = None
        if not affine:
            vals = [int(p.sizes.get(lab, 1)) for lab in reduced]
            vals += list(p.red_div)
            for q in range(nop):
                e = p.eff[q]
                vals += [int(e.get(lab, 0)) for lab in reduced]
            meta = _meta(vals, dev)

        def launch(operands):
            out = torch.empty(out_shape, dtype=out_dtype, device=dev)
            a, b, c, d = _pad(operands)
            if direct:
                dst = out
            else:
                dst = torch.empty((nk,), dtype=torch.float32, device=dev)
            if affine:
                _k_red_scalar_aff[grid](
                    dst,
                    a,
                    b,
                    c,
                    d,
                    k_total,
                    k_per,
                    BK=bk,
                    NOP=nop,
                    num_warps=8,
                    num_stages=4,
                )
            else:
                _k_red_scalar[grid](
                    dst,
                    a,
                    b,
                    c,
                    d,
                    meta,
                    k_total,
                    k_per,
                    BK=bk,
                    NLR=nlr,
                    NOP=nop,
                    num_warps=8,
                    num_stages=4,
                )
            if not direct:
                _k_final[(1,)](out, dst, 1, nk, BO=fbo, BK=fbk, num_warps=4)
            return out

        return launch

    if numel >= 4096:
        bo = 128
    elif numel >= 1024:
        bo = 64
    elif numel >= 64:
        bo = 32
    elif numel >= 16:
        bo = 16
    else:
        bo = 8
    bk = 64
    n_ob = triton.cdiv(numel, bo)
    nk = max(1, min(triton.cdiv(1024, n_ob), triton.cdiv(k_total, bk)))
    k_per = _align(triton.cdiv(k_total, nk), 16)
    nk = max(1, triton.cdiv(k_total, k_per))
    direct = nk == 1

    vals = list(p.out_shape)
    vals += list(p.out_div)
    for q in range(nop):
        e = p.eff[q]
        vals += [int(e.get(lab, 0)) for lab in out_labels]
    vals += [int(p.sizes.get(lab, 1)) for lab in reduced]
    vals += list(p.red_div)
    for q in range(nop):
        e = p.eff[q]
        vals += [int(e.get(lab, 0)) for lab in reduced]
    meta = _meta(vals, dev)
    grid = (n_ob, nk)
    fbo = min(512, triton.next_power_of_2(numel))
    fbk = max(2, min(128, triton.next_power_of_2(nk), 4096 // fbo))
    fgrid = (triton.cdiv(numel, fbo),)

    def launch(operands):
        out = torch.empty(out_shape, dtype=out_dtype, device=dev)
        a, b, c, d = _pad(operands)
        if direct:
            dst = out
        else:
            dst = torch.empty((nk, numel), dtype=torch.float32, device=dev)
        _k_red[grid](
            dst,
            a,
            b,
            c,
            d,
            meta,
            numel,
            k_total,
            k_per,
            BO=bo,
            BK=bk,
            NLO=nlo,
            NLR=nlr,
            NOP=nop,
            num_warps=8,
            num_stages=3,
        )
        if not direct:
            _k_final[fgrid](out, dst, numel, nk, BO=fbo, BK=fbk, num_warps=4)
        return out

    return launch


def _try_gemm(p, ops):
    if len(ops) != 2:
        return None
    a_labs = p.per_operand[0]
    b_labs = p.per_operand[1]
    if len(set(a_labs)) != len(a_labs) or len(set(b_labs)) != len(b_labs):
        return None
    sa = set(a_labs)
    sb = set(b_labs)
    red = p.reduced
    if len(red) != 1:
        return None
    lab_k = red[0]
    if lab_k not in sa or lab_k not in sb:
        return None
    batch = [lab for lab in p.out_labels if lab in sa and lab in sb]
    mm = [lab for lab in p.out_labels if lab in sa and lab not in sb]
    nn = [lab for lab in p.out_labels if lab in sb and lab not in sa]
    if len(mm) != 1 or len(nn) != 1 or len(batch) > 2:
        return None
    if set(p.out_labels) - sa - sb:
        return None
    if sa - set(batch) - set(mm) - set(red):
        return None
    if sb - set(batch) - set(nn) - set(red):
        return None
    dt = ops[0].dtype
    if dt != ops[1].dtype:
        return None
    if dt not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    M = int(p.sizes[mm[0]])
    N = int(p.sizes[nn[0]])
    K = int(p.sizes[lab_k])
    if M < 16 or N < 16 or K < 16:
        return None
    return {
        "M": M,
        "N": N,
        "K": K,
        "sa_m": int(p.eff[0].get(mm[0], 0)),
        "sa_k": int(p.eff[0][lab_k]),
        "sb_k": int(p.eff[1][lab_k]),
        "sb_n": int(p.eff[1].get(nn[0], 0)),
        "batch": batch,
        "m": mm[0],
        "n": nn[0],
    }


def _gemm_entry(p, ops, out_shape, out_dtype, dev, g):
    M, N, K = g["M"], g["N"], g["K"]
    batch = g["batch"]
    nb = len(batch)
    nbatch = 1
    for lab in batch:
        nbatch *= int(p.sizes[lab])

    a_b0 = int(p.eff[0].get(batch[0], 0)) if nb > 0 else 0
    a_b1 = int(p.eff[0].get(batch[1], 0)) if nb > 1 else 0
    b_b0 = int(p.eff[1].get(batch[0], 0)) if nb > 0 else 0
    b_b1 = int(p.eff[1].get(batch[1], 0)) if nb > 1 else 0
    acc = M * N
    cst = [0] * nb
    for i in range(nb - 1, -1, -1):
        cst[i] = acc
        acc *= int(p.sizes[batch[i]])
    c_b0 = cst[0] if nb > 0 else 0
    c_b1 = cst[1] if nb > 1 else 0
    bd1 = int(p.sizes[batch[1]]) if nb > 1 else 0

    # Static occupancy-aware tile choice.  The measured round-2 profile shows a
    # 512^3 GEMM launching only 64 blocks of 4 warps over 60 SMs (~1 resident
    # block per MP, ~4% compute and ~2% memory throughput, 99.95% L2 hit): the
    # kernel is latency exposed, so resident warps per MP are what matter, and
    # redundant tile loads are absorbed by L2.  Take the largest tile that
    # still yields at least ~2 blocks per MP and widen the block to 8 warps.
    cands = ((64, 64), (64, 32), (32, 32))
    target = 120
    bm, bn = cands[-1]
    for tm, tn in cands:
        if nbatch * triton.cdiv(M, tm) * triton.cdiv(N, tn) >= target:
            bm, bn = tm, tn
            break
    bk = 32
    nwarps = 8 if bm * bn >= 2048 else 4
    even = (M % bm == 0) and (N % bn == 0) and (K % bk == 0)

    grid = (nbatch, triton.cdiv(M, bm) * triton.cdiv(N, bn))
    sa_m = g["sa_m"]
    sa_k = g["sa_k"]
    sb_k = g["sb_k"]
    sb_n = g["sb_n"]
    sc_m = N
    sc_n = 1

    def launch(operands):
        out = torch.empty(out_shape, dtype=out_dtype, device=dev)
        _k_gemm[grid](
            operands[0],
            operands[1],
            out,
            M,
            N,
            K,
            sa_m,
            sa_k,
            sb_k,
            sb_n,
            sc_m,
            sc_n,
            a_b0,
            a_b1,
            b_b0,
            b_b1,
            c_b0,
            c_b1,
            bd1,
            BM=bm,
            BN=bn,
            BK=bk,
            NB=nb,
            GM=8,
            EVEN=even,
            num_warps=nwarps,
            num_stages=3,
        )
        return out

    return launch


def _try_layout_view(p, ops):
    """Zero-copy result for pure-permutation and pure-diagonal equations.

    A single operand whose labels are all kept (nothing reduced) needs no
    arithmetic at all: the answer is a strided view of that operand, which is
    exactly how the reference answers these equations.  Materialising such a
    result can never be competitive (it turns a metadata operation into a full
    HBM round trip), so express it with the permitted zero-copy layout
    operations only.  An identity permutation would alias the operand itself,
    so that case keeps the materialising kernel.
    """
    if len(ops) != 1 or p.reduced:
        return None
    x = ops[0]
    labs = p.per_operand[0]
    if len(labs) != x.dim():
        return None
    outset = set(p.out_labels)
    if len(outset) != len(p.out_labels):
        return None
    for lab in labs:
        if lab not in outset:
            return None
    counts = {}
    for lab in labs:
        counts[lab] = counts.get(lab, 0) + 1
    if len(counts) != len(outset):
        return None
    diag = None
    for lab, c in counts.items():
        if c > 2:
            return None
        if c == 2:
            if diag is not None:
                return None
            diag = lab

    if diag is None:
        perm = [labs.index(lab) for lab in p.out_labels]
        if perm == list(range(len(perm))):
            return None

        def launch(operands):
            return operands[0].permute(perm)

        return launch

    ax1 = labs.index(diag)
    ax2 = labs.index(diag, ax1 + 1)
    after = [lab for lab in labs if lab != diag] + [diag]
    perm = [after.index(lab) for lab in p.out_labels]
    ident = perm == list(range(len(perm)))

    def launch(operands):
        y = operands[0].diagonal(0, ax1, ax2)
        if ident:
            return y
        return y.permute(perm)

    return launch


def _make(equation, operands):
    p = _parse(equation, operands)
    ops = list(operands)
    dev = ops[0].device
    out_dtype = _promote([o.dtype for o in ops])
    out_shape = p.out_shape
    v = _try_layout_view(p, ops)
    if v is not None:
        return v
    g = _try_gemm(p, ops)
    if g is not None:
        return _gemm_entry(p, ops, out_shape, out_dtype, dev, g)
    if not p.reduced:
        return _elem_entry(p, ops, out_shape, out_dtype, dev)
    return _red_entry(p, ops, out_shape, out_dtype, dev)


def einsum(equation, *operands, **kwargs):
    logger.debug("GEMS_MTHREADS EINSUM")
    if not isinstance(equation, str):
        raise TypeError("einsum: equation must be a string")
    if not operands:
        raise ValueError("einsum: at least one operand is required")
    if len(operands) == 1:
        vk = (equation, operands[0].dim())
        ent = _VIEW.get(vk, _MISS)
        if ent is _MISS:
            ent = _try_layout_view(_parse(equation, operands), list(operands))
            _VIEW[vk] = ent
        if ent is not None:
            return ent(operands)
    key = (
        equation,
        tuple([(o.shape, o.stride(), o.dtype) for o in operands]),
        operands[0].device,
    )
    entry = _CACHE.get(key)
    if entry is None:
        entry = _make(equation, operands)
        _CACHE[key] = entry
    return entry(operands)
