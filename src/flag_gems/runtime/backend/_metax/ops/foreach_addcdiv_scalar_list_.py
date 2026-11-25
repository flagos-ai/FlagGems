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

"""In-place foreach addcdiv with a per-list-position scalar.

Implements ``aten::_foreach_addcdiv_.ScalarList``:

    for i in range(len(self)):
        self[i] = self[i] + scalars[i] * tensor1[i] / tensor2[i]

All arithmetic is performed in fp32 (ATen opmath path) and rounded once on
store.  The observable result is the in-place mutation of the ``self``
storages; the return value is ``None``.

Launch strategy (metadata only, no GPU sync):

* When every list position shares one element count and one scalar, the whole
  list is handled by a single fused launch,
  grid = (cdiv(numel, BLOCK), len(list)), with the position selected by
  ``tl.program_id(1)``.  This follows the ATen fast-path contract for
  ``_foreach_addcdiv_``: the per-position tensors have the same shape, the
  same dtype and are contiguous.
* Preferred form: when the positions of each list are also evenly spaced in
  memory (probed once per call with ``data_ptr``), the kernel takes only the
  three base pointers plus three per-position strides, so the launch binds
  eight arguments and contains a single body.
* Otherwise the 16-way branch kernel handles the list, and lists with
  heterogeneous sizes (or more than 16 positions) fall back to one fully
  validated launch per position, with a stride-aware variant for
  non-contiguous / broadcast inputs.
* Fused launches go through the already-compiled kernel's own runner; the
  ordinary ``kernel[grid](...)`` form stays as a fallback.
"""

import numbers

import torch
import triton
import triton.language as tl

_N_SLOTS = 16


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _slot_body(a_ptr, b_ptr, c_ptr, alpha, numel, offs, EVEN: tl.constexpr):
    if EVEN:
        a = tl.load(a_ptr + offs).to(tl.float32)
        b = tl.load(b_ptr + offs).to(tl.float32)
        c = tl.load(c_ptr + offs).to(tl.float32)
        res = a + (alpha * b) / c
        tl.store(a_ptr + offs, res.to(a_ptr.dtype.element_ty))
    else:
        mask = offs < numel
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(c_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        res = a + (alpha * b) / c
        tl.store(a_ptr + offs, res.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_stride_kernel(
    pa,
    pb,
    pc,
    sa,
    sb,
    sc,
    alpha,
    numel,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.program_id(1).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    _slot_body(pa + slot * sa, pb + slot * sb, pc + slot * sc, alpha, numel, offs, EVEN)


@triton.jit
def _fused_uniform_kernel(
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
    alpha,
    numel,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    if slot == 0:
        _slot_body(a0, b0, c0, alpha, numel, offs, EVEN)
    elif slot == 1:
        _slot_body(a1, b1, c1, alpha, numel, offs, EVEN)
    elif slot == 2:
        _slot_body(a2, b2, c2, alpha, numel, offs, EVEN)
    elif slot == 3:
        _slot_body(a3, b3, c3, alpha, numel, offs, EVEN)
    elif slot == 4:
        _slot_body(a4, b4, c4, alpha, numel, offs, EVEN)
    elif slot == 5:
        _slot_body(a5, b5, c5, alpha, numel, offs, EVEN)
    elif slot == 6:
        _slot_body(a6, b6, c6, alpha, numel, offs, EVEN)
    elif slot == 7:
        _slot_body(a7, b7, c7, alpha, numel, offs, EVEN)
    elif slot == 8:
        _slot_body(a8, b8, c8, alpha, numel, offs, EVEN)
    elif slot == 9:
        _slot_body(a9, b9, c9, alpha, numel, offs, EVEN)
    elif slot == 10:
        _slot_body(a10, b10, c10, alpha, numel, offs, EVEN)
    elif slot == 11:
        _slot_body(a11, b11, c11, alpha, numel, offs, EVEN)
    elif slot == 12:
        _slot_body(a12, b12, c12, alpha, numel, offs, EVEN)
    elif slot == 13:
        _slot_body(a13, b13, c13, alpha, numel, offs, EVEN)
    elif slot == 14:
        _slot_body(a14, b14, c14, alpha, numel, offs, EVEN)
    else:
        _slot_body(a15, b15, c15, alpha, numel, offs, EVEN)


@triton.jit
def _addcdiv_kernel(
    self_ptr,
    t1_ptr,
    t2_ptr,
    alpha,
    numel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    _slot_body(self_ptr, t1_ptr, t2_ptr, alpha, numel, offs, False)


@triton.jit
def _addcdiv_strided_kernel(
    self_ptr,
    t1_ptr,
    t2_ptr,
    alpha,
    numel,
    d1,
    d2,
    d3,
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    b2,
    b3,
    c0,
    c1,
    c2,
    c3,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    s23 = d2 * d3
    s123 = d1 * s23
    i0 = offs // s123
    r = offs % s123
    i1 = r // s23
    r = r % s23
    i2 = r // d3
    i3 = r % d3

    oa = i0 * a0 + i1 * a1 + i2 * a2 + i3 * a3
    ob = i0 * b0 + i1 * b1 + i2 * b2 + i3 * b3
    oc = i0 * c0 + i1 * c1 + i2 * c2 + i3 * c3

    a = tl.load(self_ptr + oa, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(t1_ptr + ob, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(t2_ptr + oc, mask=mask, other=0.0).to(tl.float32)
    res = a + (alpha * b) / c
    tl.store(self_ptr + oa, res.to(self_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------

_MAX_DIMS = 4

# key -> compiled-kernel runner, or False when the low-level form is unusable
_RUNNERS = {}


def _block_for(numel):
    if numel <= 1024:
        return 256, 1
    return 1024, 4


def _launch_cached(kernel, key, full_args, grid, block, even, warps):
    """Launch through the compiled kernel's own runner when possible."""
    runner = _RUNNERS.get(key)
    if runner is None:
        # Compile once for this exact signature, then reuse the compiled
        # kernel's own runner for every later call.
        ck = kernel.warmup(
            *full_args,
            BLOCK=block,
            EVEN=even,
            num_warps=warps,
            num_stages=1,
            grid=(1,),
        )
        runner = ck[(grid[0], grid[1], 1)]
        _RUNNERS[key] = runner
    if runner is not False:
        try:
            # The trailing positional slot is this build's launch flag
            # (cooperative grid / PDL); None keeps the default behaviour.
            runner(*full_args, None)
            return
        except TypeError:
            _RUNNERS[key] = False
    kernel[grid](
        *full_args,
        BLOCK=block,
        EVEN=even,
        num_warps=warps,
        num_stages=1,
    )


def _launch_contig(a, b, c, alpha, numel):
    BLOCK, warps = _block_for(numel)
    grid = (triton.cdiv(numel, BLOCK),)
    _addcdiv_kernel[grid](
        a, b, c, alpha, numel, BLOCK=BLOCK, num_warps=warps, num_stages=1
    )


def _dims_strides(t):
    shape = list(t.shape)
    strides = list(t.stride())
    while len(shape) < _MAX_DIMS:
        shape.insert(0, 1)
        strides.insert(0, 0)
    return shape, strides


def _launch_strided(a, b, c, alpha, numel):
    sa, ta = _dims_strides(a)
    sb, tb = _dims_strides(b)
    sc, tc = _dims_strides(c)
    BLOCK, warps = _block_for(numel)
    grid = (triton.cdiv(numel, BLOCK),)
    _addcdiv_strided_kernel[grid](
        a,
        b,
        c,
        alpha,
        numel,
        sa[1],
        sa[2],
        sa[3],
        ta[0],
        ta[1],
        ta[2],
        ta[3],
        tb[0],
        tb[1],
        tb[2],
        tb[3],
        tc[0],
        tc[1],
        tc[2],
        tc[3],
        BLOCK=BLOCK,
        num_warps=warps,
        num_stages=1,
    )


def _normalize_scalars(scalars, n):
    if isinstance(scalars, torch.Tensor):
        if scalars.numel() == 1:
            scalars = scalars.item()
        else:
            scalars = scalars.tolist()
    if isinstance(scalars, numbers.Number):
        return [float(scalars)] * n
    if isinstance(scalars, (list, tuple)):
        vals = []
        for s in scalars:
            if isinstance(s, torch.Tensor):
                s = s.item()
            vals.append(float(s))
        if len(vals) == 0:
            return [1.0] * n
        if len(vals) == 1 and n > 1:
            return vals * n
        return vals
    return [float(scalars)] * n


def _apply_one(a, b, c, alpha):
    numel = a.numel()
    if numel == 0:
        return
    same_shape = a.shape == b.shape and a.shape == c.shape
    if same_shape and a.is_contiguous() and b.is_contiguous() and c.is_contiguous():
        _launch_contig(a, b, c, alpha, numel)
    elif same_shape:
        _launch_strided(a, b, c, alpha, numel)
    else:
        # Broadcasting fallback: expand zero-copy views to a common shape and
        # use the stride-aware kernel (expanded dims carry stride 0).
        shape = torch.broadcast_shapes(a.shape, b.shape, c.shape)
        a = a.expand(shape)
        b = b.expand(shape)
        c = c.expand(shape)
        _launch_strided(a, b, c, alpha, a.numel())


def _even_stride(lst, n, base, esz):
    """Element stride if lst is evenly spaced from base, else None."""
    if n == 1:
        return 0
    p0 = lst[0].data_ptr()
    if p0 != base:
        return None
    p1 = lst[1].data_ptr()
    step = p1 - p0
    if step <= 0 or step % esz or step % 16:
        return None
    step //= esz
    for i in range(2, n):
        if lst[i].data_ptr() - p0 != i * step * esz:
            return None
    return step


def _try_fused(self_list, t1_list, t2_list, alphas, n):
    """Return True when the whole list was handled by one fused launch."""
    a0 = self_list[0]
    numel = a0.numel()
    if numel == 0:
        return False
    alpha = alphas[0]
    for sc in alphas:
        if sc != alpha:
            return False
    for i in range(1, n):
        if self_list[i].numel() != numel:
            return False

    block, warps = _block_for(numel)
    even = (numel % block) == 0
    b0 = t1_list[0]
    c0 = t2_list[0]

    # Preferred: one body, three base pointers and three per-position strides.
    esa = a0.element_size()
    esb = b0.element_size()
    esc = c0.element_size()
    pa = a0.data_ptr()
    pb = b0.data_ptr()
    pc = c0.data_ptr()
    if not ((pa | pb | pc) & 15):
        sa = _even_stride(self_list, n, pa, esa)
        if sa is not None:
            sb = _even_stride(t1_list, n, pb, esb)
            if sb is not None:
                sc = _even_stride(t2_list, n, pc, esc)
                if sc is not None:
                    grid = (triton.cdiv(numel, block), n)
                    key = (
                        "stride",
                        numel,
                        n,
                        block,
                        warps,
                        even,
                        sa,
                        sb,
                        sc,
                        a0.dtype,
                        b0.dtype,
                        c0.dtype,
                    )
                    _launch_cached(
                        _fused_stride_kernel,
                        key,
                        [a0, b0, c0, sa, sb, sc, alpha, numel],
                        grid,
                        block,
                        even,
                        warps,
                    )
                    return True

    if n > _N_SLOTS:
        return False

    args = [a0, b0, c0]
    ext = args.append
    for i in range(1, n):
        ext(self_list[i])
        ext(t1_list[i])
        ext(t2_list[i])
    for _ in range(_N_SLOTS - n):
        ext(a0)
        ext(b0)
        ext(c0)

    grid = (triton.cdiv(numel, block), n)
    key = ("branch", numel, n, block, warps, even, a0.dtype)
    _launch_cached(
        _fused_uniform_kernel,
        key,
        args + [alpha, numel],
        grid,
        block,
        even,
        warps,
    )
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def foreach_addcdiv_scalar_list_(*args, **kwargs):
    if len(args) >= 4:
        self_list, t1_list, t2_list, scalars = args[0], args[1], args[2], args[3]
    elif len(args) == 3:
        self_list, t1_list, t2_list = args
        scalars = kwargs.get("scalars", kwargs.get("scalar", 1.0))
    elif len(args) == 0:
        self_list = kwargs.get("self", kwargs.get("tensors1", kwargs.get("self_list")))
        t1_list = kwargs.get("tensor1", kwargs.get("t1"))
        t2_list = kwargs.get("tensor2", kwargs.get("t2"))
        scalars = kwargs.get("scalars", kwargs.get("scalar", 1.0))
    else:
        raise TypeError("run() expects (self, tensor1, tensor2, scalars)")

    if self_list is None or t1_list is None or t2_list is None:
        raise TypeError("run() requires self, tensor1 and tensor2")

    if isinstance(self_list, torch.Tensor):
        self_list = [self_list]
    if isinstance(t1_list, torch.Tensor):
        t1_list = [t1_list]
    if isinstance(t2_list, torch.Tensor):
        t2_list = [t2_list]

    n = len(self_list)
    if n == 0:
        return None

    alphas = _normalize_scalars(scalars, n)

    if len(t1_list) != n or len(t2_list) != n:
        raise ValueError("self, tensor1 and tensor2 must have the same length")

    if _try_fused(self_list, t1_list, t2_list, alphas, n):
        return None

    for i in range(n):
        a = self_list[i]
        b = t1_list[i]
        c = t2_list[i]
        alpha = alphas[i] if i < len(alphas) else alphas[-1]
        if a.numel() == 0:
            continue
        _apply_one(a, b, c, alpha)

    return None
