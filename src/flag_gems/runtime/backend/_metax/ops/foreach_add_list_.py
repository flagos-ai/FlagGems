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

"""Fused Triton implementation of ``aten::_foreach_add_.List`` (in place).

Semantics: for every i, ``self[i].add_(other[i], alpha=alpha)`` computed in the
opmath type of ``self[i]`` (float32 for fp16/bf16), rounded once on store.
``self`` keeps its storage, strides and dtype; the call returns ``None``.

Hot path: one launch covers up to four (self, other) pairs.  The tensors are
passed as real (16-byte aligned) pointer arguments and the block -> tensor
mapping is a branch on the cumulative block counts, so the kernel needs no
device side metadata table and Triton can vectorize both accesses.

Because the op is a pure streaming kernel, the per-call host cost dominates the
small-shape latency.  For the common float case the launcher therefore reuses
an already compiled kernel (and its launcher closure) instead of re-running
Triton's per-call signature binding, specialization and cache-key lookup.  The
cache key is a superset of Triton's own key (device, dtype, BLOCK, num_warps,
the exact integer arguments and a 16-byte pointer alignment flag), so a reused
binary is always the binary Triton would have selected itself.
"""

import numbers

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

# BLOCK / num_warps per dtype for the fused contiguous kernel, for tensors of
# at most _HUGE_ELEMS elements.  Larger tensors use the smaller _HUGE_CFG tile:
# the two size classes measured consistently different optima (the 16.7M-element
# class prefers the larger tile, the 1e9-element class the smaller one).
_BLOCK_CFG = {
    torch.float32: (2048, 4),
    torch.float16: (4096, 4),
    torch.bfloat16: (4096, 4),
    torch.float64: (1024, 4),
    torch.int8: (8192, 4),
    torch.uint8: (8192, 4),
    torch.int16: (4096, 4),
    torch.int32: (2048, 4),
    torch.int64: (1024, 4),
    torch.bool: (8192, 4),
}

_HUGE_CFG = {
    torch.float32: (512, 4),
    torch.float16: (1024, 4),
    torch.bfloat16: (1024, 4),
    torch.float64: (256, 4),
    torch.int8: (1024, 4),
    torch.uint8: (1024, 4),
    torch.int16: (1024, 4),
    torch.int32: (512, 4),
    torch.int64: (256, 4),
    torch.bool: (1024, 4),
}

# Above this element count per tensor the smaller tile is measurably better.
_HUGE_ELEMS = 1 << 26

# (device, dtype, BLOCK, num_warps, numels/cums...) -> launcher closure
_LAUNCH_CACHE = {}


# ---------------------------------------------------------------------------
# elementwise body: fully typed from the pointer arguments themselves
# ---------------------------------------------------------------------------
@triton.jit
def _elem_body(sp, op, n_elem, blk, alpha, BLOCK: tl.constexpr):
    ST = sp.dtype.element_ty
    offs = blk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    if ST == tl.float32 or ST == tl.float16 or ST == tl.bfloat16:
        sv = tl.load(sp + offs, mask=mask, other=0.0).to(tl.float32)
        ov = tl.load(op + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(sp + offs, (sv + alpha * ov).to(ST), mask=mask)
    elif ST == tl.float64:
        sv = tl.load(sp + offs, mask=mask, other=0.0)
        ov = tl.load(op + offs, mask=mask, other=0.0).to(tl.float64)
        tl.store(sp + offs, sv + alpha.to(tl.float64) * ov, mask=mask)
    elif ST == tl.int1:
        sv = tl.load(sp + offs, mask=mask, other=0)
        ov = tl.load(op + offs, mask=mask, other=0)
        tl.store(sp + offs, sv | (ov & (alpha != 0)), mask=mask)
    else:
        ai = alpha.to(tl.int64)
        sv = tl.load(sp + offs, mask=mask, other=0).to(tl.int64)
        ov = tl.load(op + offs, mask=mask, other=0).to(tl.int64)
        tl.store(sp + offs, (sv + ai * ov).to(ST), mask=mask)


@triton.jit
def _fast4(
    s0,
    o0,
    n0,
    c1,
    s1,
    o1,
    n1,
    c2,
    s2,
    o2,
    n2,
    c3,
    s3,
    o3,
    n3,
    alpha,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < c1:
        _elem_body(s0, o0, n0, pid, alpha, BLOCK)
    elif pid < c2:
        _elem_body(s1, o1, n1, pid - c1, alpha, BLOCK)
    elif pid < c3:
        _elem_body(s2, o2, n2, pid - c2, alpha, BLOCK)
    else:
        _elem_body(s3, o3, n3, pid - c3, alpha, BLOCK)


# ---------------------------------------------------------------------------
# generic kernel: one launch per tensor (strided / broadcast / scalar operand)
# ---------------------------------------------------------------------------
@triton.jit
def _generic_body(
    sp, op, meta, alpha, ov, BLOCK: tl.constexpr, OTHER_SCALAR: tl.constexpr
):
    ST = sp.dtype.element_ty
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    numel = tl.load(meta + 0)
    mask = idx < numel

    rem = idx
    soff = tl.zeros([BLOCK], dtype=tl.int64)
    ooff = tl.zeros([BLOCK], dtype=tl.int64)
    for k in tl.static_range(8):
        slot = 7 - k
        ssz = tl.load(meta + 1 + slot)
        sst = tl.load(meta + 9 + slot)
        coord = rem % ssz
        rem = rem // ssz
        soff += coord * sst
        if not OTHER_SCALAR:
            osz = tl.load(meta + 17 + slot)
            ost = tl.load(meta + 25 + slot)
            ooff += tl.where(osz == ssz, coord, 0) * ost

    if ST == tl.float32 or ST == tl.float16 or ST == tl.bfloat16:
        sv = tl.load(sp + soff, mask=mask, other=0.0).to(tl.float32)
        if OTHER_SCALAR:
            val = sv + alpha * ov
        else:
            val = sv + alpha * tl.load(op + ooff, mask=mask, other=0.0).to(tl.float32)
        tl.store(sp + soff, val.to(ST), mask=mask)
    elif ST == tl.float64:
        sv = tl.load(sp + soff, mask=mask, other=0.0)
        if OTHER_SCALAR:
            val = sv + alpha.to(tl.float64) * ov.to(tl.float64)
        else:
            val = sv + alpha.to(tl.float64) * tl.load(
                op + ooff, mask=mask, other=0.0
            ).to(tl.float64)
        tl.store(sp + soff, val, mask=mask)
    elif ST == tl.int1:
        sv = tl.load(sp + soff, mask=mask, other=0)
        if OTHER_SCALAR:
            val = sv | ((ov != 0) & (alpha != 0))
        else:
            val = sv | (tl.load(op + ooff, mask=mask, other=0) & (alpha != 0))
        tl.store(sp + soff, val, mask=mask)
    else:
        ai = alpha.to(tl.int64)
        sv = tl.load(sp + soff, mask=mask, other=0).to(tl.int64)
        if OTHER_SCALAR:
            val = sv + ai * ov.to(tl.int64)
        else:
            val = sv + ai * tl.load(op + ooff, mask=mask, other=0).to(tl.int64)
        tl.store(sp + soff, val.to(ST), mask=mask)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------
def _launch_fast(device, dtype, ss, oo, alpha):
    fast_launch = dtype.is_floating_point
    alpha_num = float(alpha) if fast_launch else int(alpha)
    n = len(ss)
    for start in range(0, n, 4):
        s0 = ss[start]
        o0 = oo[start]
        n0 = s0.numel()
        rem = n - start
        if rem > 1:
            s1 = ss[start + 1]
            o1 = oo[start + 1]
            n1 = s1.numel()
        else:
            s1 = s0
            o1 = o0
            n1 = 0
        if rem > 2:
            s2 = ss[start + 2]
            o2 = oo[start + 2]
            n2 = s2.numel()
        else:
            s2 = s0
            o2 = o0
            n2 = 0
        if rem > 3:
            s3 = ss[start + 3]
            o3 = oo[start + 3]
            n3 = s3.numel()
        else:
            s3 = s0
            o3 = o0
            n3 = 0
        if (
            n0 <= _HUGE_ELEMS
            and n1 <= _HUGE_ELEMS
            and n2 <= _HUGE_ELEMS
            and n3 <= _HUGE_ELEMS
        ):
            block, warps = _BLOCK_CFG[dtype]
        else:
            block, warps = _HUGE_CFG[dtype]
        b0 = (n0 + block - 1) // block
        b1 = (n1 + block - 1) // block
        b2 = (n2 + block - 1) // block
        b3 = (n3 + block - 1) // block
        c1 = b0
        c2 = c1 + b1
        c3 = c2 + b2
        total = c3 + b3
        if total <= 0:
            continue
        args = (s0, o0, n0, c1, s1, o1, n1, c2, s2, o2, n2, c3, s3, o3, n3, alpha_num)
        if fast_launch:
            addr = s0.data_ptr() | o0.data_ptr()
            if rem > 1:
                addr |= s1.data_ptr() | o1.data_ptr()
            if rem > 2:
                addr |= s2.data_ptr() | o2.data_ptr()
            if rem > 3:
                addr |= s3.data_ptr() | o3.data_ptr()
            if addr & 15 == 0:
                key = (device, dtype, block, warps, n0, c1, n1, c2, n2, c3, n3)
                entry = _LAUNCH_CACHE.get(key)
                if entry is None:
                    compiled = _fast4.run(
                        *args, grid=(1, 1, 1), warmup=True, BLOCK=block, num_warps=warps
                    )
                    # loads the binary through the supported path once
                    compiled[(total, 1, 1)]
                    dev_idx = device.index
                    if dev_idx is None:
                        dev_idx = _driver.active.get_current_device()
                    entry = (
                        compiled.run,
                        compiled.function,
                        compiled.packed_metadata,
                        dev_idx,
                    )
                    _LAUNCH_CACHE[key] = entry
                run_fn, fn, packed, dev_idx = entry
                run_fn(
                    total,
                    1,
                    1,
                    _driver.active.get_current_stream(dev_idx),
                    fn,
                    packed,
                    None,
                    triton.knobs.runtime.launch_enter_hook,
                    triton.knobs.runtime.launch_exit_hook,
                    *args,
                    block,
                )
                continue
        _fast4[(total,)](*args, BLOCK=block, num_warps=warps)


def _launch_generic(tensor, other, is_scalar, alpha, scalar_value):
    if tensor.dtype not in _BLOCK_CFG:
        raise RuntimeError("unsupported dtype %s" % (tensor.dtype,))
    numel = tensor.numel()
    if numel == 0:
        return
    is_float = tensor.dtype.is_floating_point
    ndim = tensor.dim()
    sizes = [1] * 8
    sstr = [0] * 8
    for d in range(ndim):
        sizes[8 - ndim + d] = tensor.shape[d]
        sstr[8 - ndim + d] = tensor.stride(d)
    o_sizes = [1] * 8
    o_str = [0] * 8
    alpha_arg = float(alpha) if is_float else int(alpha)
    if is_scalar:
        op_arg = tensor
        ov_arg = float(scalar_value) if is_float else int(scalar_value)
    else:
        op_arg = other
        ov_arg = 0.0
        odim = other.dim()
        for d in range(odim):
            o_sizes[8 - odim + d] = other.shape[d]
            o_str[8 - odim + d] = other.stride(d)
    meta = [numel] + sizes + sstr + o_sizes + o_str
    tab = torch.tensor(meta, dtype=torch.int64, device=tensor.device)
    block = 1024
    _generic_body[(triton.cdiv(numel, block),)](
        tensor,
        op_arg,
        tab,
        alpha_arg,
        ov_arg,
        BLOCK=block,
        OTHER_SCALAR=is_scalar,
        num_warps=4,
    )


def _run_pairs(selfs, others, alpha, n):
    """Common case: ``other`` is a full-length tensor list."""
    s0 = selfs[0]
    try:
        dev = s0.device
        dt = s0.dtype
        ok = dt in _BLOCK_CFG
        if ok:
            for i in range(n):
                s = selfs[i]
                o = others[i]
                if (
                    s.dtype is not dt
                    or o.dtype is not dt
                    or s.device != dev
                    or not s.is_contiguous()
                    or not o.is_contiguous()
                    or s.shape != o.shape
                ):
                    ok = False
                    break
    except AttributeError:
        ok = False
    if ok:
        _launch_fast(dev, dt, selfs, others, alpha)
        return None
    return _run_mixed(selfs, others, alpha, n)


def _run_scalars(selfs, values, alpha, n):
    generic = []
    for i in range(n):
        s = selfs[i]
        val = values[i]
        _check_scalar_dtype(s, val, alpha)
        generic.append((s, s, True, alpha, val))
    for s, o, is_scalar, a, v in generic:
        _launch_generic(s, o, is_scalar, a, v)
    return None


def _run_mixed(selfs, others, alpha, n):
    fast = {}
    generic = []
    for i in range(n):
        s = selfs[i]
        o = others[i]
        if not isinstance(s, torch.Tensor) or not isinstance(o, torch.Tensor):
            raise RuntimeError("mixed Tensor / Scalar lists are not supported")
        if o.device != s.device:
            raise RuntimeError("all tensors must be on the same device")
        sd = s.dtype
        od = o.dtype
        if sd != od:
            promote = torch.promote_types(sd, od)
            if not torch.can_cast(promote, sd):
                raise RuntimeError(
                    "result type %s can't be cast to the desired output type %s"
                    % (promote, sd)
                )
        elif (
            sd in _BLOCK_CFG
            and s.is_contiguous()
            and o.is_contiguous()
            and s.shape == o.shape
        ):
            key = (s.device, sd)
            slot = fast.get(key)
            if slot is None:
                slot = ([], [])
                fast[key] = slot
            slot[0].append(s)
            slot[1].append(o)
            continue
        generic.append((s, o, False, alpha, 0.0))

    for (device, dtype), (ss, oo) in fast.items():
        _launch_fast(device, dtype, ss, oo, alpha)
    for s, o, is_scalar, a, v in generic:
        _launch_generic(s, o, is_scalar, a, v)
    return None


def foreach_add_list_(self, other=None, alpha=1, *args, **kwargs):
    if kwargs:
        if "alpha" in kwargs:
            alpha = kwargs.pop("alpha")
        if "self" in kwargs:
            self = kwargs.pop("self")
        if "other" in kwargs:
            other = kwargs.pop("other")
        if kwargs:
            raise TypeError("unexpected keyword arguments: %s" % (tuple(kwargs),))
    if args:
        if len(args) > 1:
            raise TypeError("run() takes at most 3 positional arguments")
        alpha = args[0]

    t = type(self)
    selfs = self if (t is list or t is tuple) else list(self)
    n = len(selfs)
    if n == 0:
        return None

    t = type(other)
    if t is list or t is tuple:
        if len(other) != n:
            raise RuntimeError(
                "_foreach_add_ expected the same number of tensors in the two "
                "lists, got %d and %d" % (n, len(other))
            )
        first = other[0]
        if isinstance(first, torch.Tensor):
            return _run_pairs(selfs, other, alpha, n)
        if isinstance(first, numbers.Number) and not isinstance(first, complex):
            return _run_scalars(selfs, other, alpha, n)
        raise RuntimeError("unsupported element type in other")
    if isinstance(other, torch.Tensor):
        return _run_pairs(selfs, [other] * n, alpha, n)
    if isinstance(other, numbers.Number) and not isinstance(other, complex):
        return _run_scalars(selfs, [other] * n, alpha, n)
    raise RuntimeError("unsupported other argument")


def _check_scalar_dtype(tensor, value, alpha):
    dt = tensor.dtype
    if dt not in _BLOCK_CFG:
        raise RuntimeError("unsupported dtype %s" % (dt,))
    if dt.is_floating_point:
        return
    if not _is_integral(alpha):
        raise RuntimeError(
            "alpha must not be a floating point number for integral inputs"
        )
    if not _is_integral(value):
        raise RuntimeError(
            "other must not be a floating point number for integral inputs"
        )


def _is_integral(value):
    if isinstance(value, bool):
        return True
    if isinstance(value, numbers.Integral):
        return True
    if isinstance(value, numbers.Real):
        return float(value).is_integer()
    return False
