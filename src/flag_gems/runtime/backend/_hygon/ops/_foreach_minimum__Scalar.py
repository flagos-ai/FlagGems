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

"""Triton implementation of ``aten::_foreach_minimum_.Scalar``.

Applies ``min_nan(x, scalar)`` in place to every tensor of the input list
(``torch.minimum`` semantics: NaN propagates from either operand) and returns
``None``.  Storage, strides and the list itself are preserved.

Layout handling (no copies, arbitrary strides):
  * dims of size 1 are dropped;
  * the remaining dims are sorted by stride and merged while dense, which turns
    every dense / uniformly-strided tensor into a single logical 1-D run;
  * otherwise the smallest-stride dim becomes the vector axis and the rest are
    collapsed into a runtime-decomposed row index.

Launch handling: on this target a Triton host dispatch costs tens of
microseconds and grows with the number of kernel arguments, while a short
tensor's device time is a few microseconds.  The launch paths therefore use
dedicated minimal-signature kernels (base pointer, scalar, length), and tiny
uniform tensor lists (same dtype/device/length, contiguous, <= 2M elements) are
served by a single launch whose grid axis 1 selects the tensor.  Bandwidth-bound
tensors use a wider 2048-element tile (16 elements per thread) and a single
NaN-propagating minimum instruction.
"""

import logging

import torch
import triton
import triton.language as tl
from triton import knobs as _knobs
from triton.runtime import driver as _driver

logger = logging.getLogger(__name__)

_MAX_OUTER = 8
_FUSE_MAX_NUMEL = 2097152
_WIDE_MIN_NUMEL = 2097152


# --------------------------------------------------------------------------- #
# kernels (minimal launch signatures)
# --------------------------------------------------------------------------- #
@triton.jit
def _min_body(base, offs, n, s_val):
    mask = offs < n
    x = tl.load(base + offs, mask=mask)
    s = s_val.to(x.dtype)
    res = tl.minimum(x, s, propagate_nan=tl.PropagateNan.ALL)
    tl.store(base + offs, res, mask=mask)


@triton.jit
def _min_c1(x_ptr, s_val, n):
    offs = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    _min_body(x_ptr, offs, n, s_val)


@triton.jit
def _min_c1_wide(x_ptr, s_val, n):
    offs = tl.program_id(0) * 2048 + tl.arange(0, 2048)
    _min_body(x_ptr, offs, n, s_val)


@triton.jit
def _min_c1_i64(x_ptr, s_val, n):
    offs = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024).to(tl.int64)
    _min_body(x_ptr, offs, n, s_val)


@triton.jit
def _min_i1(x_ptr, s_val, n):
    offs = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, tl.minimum(x, s_val.to(x.dtype)), mask=mask)


@triton.jit
def _min_d1(x_ptr, s_ptr, n):
    offs = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024).to(tl.int64)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    s = tl.load(s_ptr)
    res = tl.minimum(x, s, propagate_nan=tl.PropagateNan.ALL)
    tl.store(x_ptr + offs, res, mask=mask)


@triton.jit
def _min_u4(p0, p1, p2, p3, s_val, n):
    """One launch for up to four same-dtype, equal-length contiguous tensors.

    Grid axis 1 selects the tensor; slots beyond the launched grid.y are never
    executed, so padded pointers are inert.
    """
    pid = tl.program_id(0)
    t = tl.program_id(1)
    offs = pid * 1024 + tl.arange(0, 1024)
    if t == 0:
        _min_body(p0, offs, n, s_val)
    elif t == 1:
        _min_body(p1, offs, n, s_val)
    elif t == 2:
        _min_body(p2, offs, n, s_val)
    else:
        _min_body(p3, offs, n, s_val)


# --------------------------------------------------------------------------- #
# kernels (rare strided paths)
# --------------------------------------------------------------------------- #
@triton.jit
def _min_gapped(
    x_ptr,
    s_ptr,
    s_val,
    n,
    g,
    USE_PTR: tl.constexpr,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs * g, mask=mask)
    if USE_PTR:
        s = tl.load(s_ptr)
        res = tl.minimum(x, s, propagate_nan=tl.PropagateNan.ALL)
    else:
        s = s_val.to(x.dtype)
        res = tl.where(x != x, x, tl.where(s != s, s, tl.minimum(x, s)))
    tl.store(x_ptr + offs * g, res, mask=mask)


@triton.jit
def _min_tiled(
    x_ptr,
    s_ptr,
    s_val,
    nrows,
    ncols,
    inner_stride,
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
    N_OUTER: tl.constexpr,
    USE_PTR: tl.constexpr,
    INNER_ONE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    I64: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < nrows
    r = rows
    if I64:
        off = tl.zeros([BLOCK_R], dtype=tl.int64)
    else:
        off = tl.zeros([BLOCK_R], dtype=tl.int32)
    ds = (d0, d1, d2, d3, d4, d5, d6, d7)
    ts = (t0, t1, t2, t3, t4, t5, t6, t7)
    for k in tl.static_range(N_OUTER):
        i = r % ds[k]
        r = r // ds[k]
        if I64:
            off += i.to(tl.int64) * ts[k]
        else:
            off += i * ts[k]
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < ncols
    if INNER_ONE:
        ptrs = x_ptr + off[:, None] + cols[None, :]
    else:
        ptrs = x_ptr + off[:, None] + cols[None, :] * inner_stride
    mask = rmask[:, None] & cmask[None, :]
    x = tl.load(ptrs, mask=mask)
    if USE_PTR:
        s = tl.load(s_ptr)
        res = tl.minimum(x, s, propagate_nan=tl.PropagateNan.ALL)
    else:
        s = s_val.to(x.dtype)
        res = tl.where(x != x, x, tl.where(s != s, s, tl.minimum(x, s)))
    tl.store(ptrs, res, mask=mask)


# --------------------------------------------------------------------------- #
# host helpers
# --------------------------------------------------------------------------- #
def _collapse(shape, strides):
    """Drop size-1 dims, sort by stride, merge dense neighbours."""
    dims = [[int(s), int(st)] for s, st in zip(shape, strides) if int(s) != 1]
    if not dims:
        return dims
    dims = sorted(dims, key=lambda d: d[1])
    merged = [dims[0]]
    for s, st in dims[1:]:
        ps, pst = merged[-1]
        if pst * ps == st:
            merged[-1][0] = ps * s
        else:
            merged.append([s, st])
    return merged


_LAUNCHERS = {}


def _cached_launch(kernel_id, jit_fn, grid3, args, extra_key):
    """Launch through the compiled kernel handle that the first ``kernel[grid]``
    call returns, so repeat calls skip Triton's binder / cache-key / global-check
    path (the dominant host cost here).  The handle is keyed on everything that
    the JIT specialisation would key on: element type, device, the ``n == 1``
    specialisation, base-pointer 16-byte alignment, and the launch grid.

    Repeat calls go straight to the compiled launcher with the same launch hooks
    the runner would pass, skipping the per-call launch-metadata construction;
    any failure falls back to the officially returned runner, and the op is
    idempotent so a retry after a partially executed launch is still correct.
    """
    key = (kernel_id,) + grid3 + extra_key
    entry = _LAUNCHERS.get(key)
    if entry is None:
        compiled = jit_fn[grid3](*args)
        if compiled is None:
            return
        entry = (
            compiled.run,
            compiled.function,
            compiled.packed_metadata,
            compiled[grid3],
        )
        _LAUNCHERS[key] = entry
        return
    launcher, function, packed, runner = entry
    try:
        dev = _driver.active.get_current_device()
        stream = _driver.active.get_current_stream(dev)
        launcher(
            grid3[0],
            grid3[1],
            grid3[2],
            stream,
            function,
            packed,
            None,
            _knobs.runtime.launch_enter_hook,
            _knobs.runtime.launch_exit_hook,
            *args,
        )
    except Exception:  # noqa: BLE001
        runner(*args)


def _launch_single(t, s_val, s_ptr, use_ptr, is_int=False):
    n = t.numel()
    if t.is_contiguous():
        if use_ptr:
            _min_d1[((n + 1023) // 1024,)](t, s_ptr, n, num_warps=4)
        elif is_int:
            _min_i1[((n + 1023) // 1024,)](t, s_val, n, num_warps=4)
        elif n >= 2147483648:
            _min_c1_i64[((n + 1023) // 1024,)](t, s_val, n, num_warps=4)
        elif n > _WIDE_MIN_NUMEL and t.element_size() == 2:
            grid3 = ((n + 2047) // 2048, 1, 1)
            _cached_launch(
                "c1w",
                _min_c1_wide,
                grid3,
                (t, s_val, n),
                (t.dtype, t.get_device(), n == 1, t.data_ptr() % 16 == 0),
            )
        else:
            grid3 = ((n + 1023) // 1024, 1, 1)
            _cached_launch(
                "c1",
                _min_c1,
                grid3,
                (t, s_val, n),
                (t.dtype, t.get_device(), n == 1, t.data_ptr() % 16 == 0),
            )
        return

    shape = t.shape
    strides = t.stride()
    for s, st in zip(shape, strides):
        if int(s) > 1 and int(st) == 0:
            raise RuntimeError(
                "unsupported operation: more than one element of the written-to "
                "tensor refers to a single memory location. Please clone() the "
                "tensor before performing the operation."
            )

    i64 = n >= 2147483648
    block = 1024 if n > 4096 else 256
    warps = 4 if n > 4096 else 2
    merged = _collapse(shape, strides)
    if len(merged) <= 1:
        g = merged[0][1] if merged else 1
        grid = ((n + block - 1) // block,)
        _min_gapped[grid](t, s_ptr, s_val, n, g, use_ptr, block, i64, num_warps=warps)
        return

    inner = merged[0]
    outer = merged[1:]
    if len(outer) > _MAX_OUTER:
        raise RuntimeError("unsupported tensor rank for in-place minimum")
    ncols = inner[0]
    inner_stride = inner[1]
    nrows = 1
    for s, _ in outer:
        nrows *= s

    bcols = min(256, max(16, triton.next_power_of_2(ncols)))
    brows = max(1, 1024 // bcols)
    dims = [1] * _MAX_OUTER
    strs = [0] * _MAX_OUTER
    for k, (s, st) in enumerate(outer):
        dims[k] = s
        strs[k] = st
    grid = ((nrows + brows - 1) // brows, (ncols + bcols - 1) // bcols)
    _min_tiled[grid](
        t,
        s_ptr,
        s_val,
        nrows,
        ncols,
        inner_stride,
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        dims[4],
        dims[5],
        dims[6],
        dims[7],
        strs[0],
        strs[1],
        strs[2],
        strs[3],
        strs[4],
        strs[5],
        strs[6],
        strs[7],
        len(outer),
        use_ptr,
        inner_stride == 1,
        brows,
        bcols,
        i64,
        num_warps=4,
    )


def _launch_uniform(group, s_val, n):
    """One launch for 2..4 same-dtype, equal-length contiguous tensors."""
    g = len(group)
    if g < 4:
        group = group + [group[-1]] * (4 - g)
    nb = (n + 1023) // 1024
    grid3 = (nb, g, 1)
    _cached_launch(
        "u4",
        _min_u4,
        grid3,
        (group[0], group[1], group[2], group[3], s_val, n),
        (
            group[0].dtype,
            group[0].get_device(),
            n == 1,
            group[0].data_ptr() % 16 == 0,
            group[1].data_ptr() % 16 == 0,
            group[2].data_ptr() % 16 == 0,
            group[3].data_ptr() % 16 == 0,
        ),
    )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def _foreach_minimum__Scalar(self, scalar):
    logger.debug("GEMS_HYGON _FOREACH_MINIMUM__SCALAR")
    if isinstance(self, torch.Tensor):
        seq = [self]
    else:
        seq = list(self)
    total = len(seq)
    if total == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    if isinstance(scalar, torch.Tensor):
        sval = scalar.item()
    else:
        sval = scalar

    i = 0
    while i < total:
        t = seq[i]
        if not isinstance(t, torch.Tensor):
            raise TypeError("expected Tensor in the tensor list")
        n = t.numel()
        if n == 0:
            i += 1
            continue
        dt = t.dtype
        if dt.is_floating_point:
            if dt == torch.float64:
                sp = torch.tensor(float(sval), dtype=torch.float64, device=t.device)
                _launch_single(t, 0.0, sp, True)
                i += 1
                continue
            sv = float(sval)
            if t.is_contiguous() and n <= _FUSE_MAX_NUMEL:
                dev = t.get_device()
                g = 1
                while g < 4 and i + g < total:
                    u = seq[i + g]
                    if (
                        not isinstance(u, torch.Tensor)
                        or u.dtype != dt
                        or u.get_device() != dev
                        or not u.is_contiguous()
                        or u.numel() != n
                    ):
                        break
                    g += 1
                if g > 1:
                    _launch_uniform(seq[i : i + g], sv, n)
                    i += g
                    continue
            _launch_single(t, sv, t, False)
        elif dt == torch.bool:
            raise RuntimeError(
                "result type Long can't be cast to the desired output type Bool"
            )
        elif dt.is_complex:
            raise RuntimeError("minimum not implemented for complex tensors")
        else:
            if isinstance(sval, float):
                raise RuntimeError(
                    "result type Float can't be cast to the desired output type "
                    + str(dt).replace("torch.", "")
                )
            _launch_single(t, int(sval), t, False, True)
        i += 1
    return None


# aliases so the injected entry point resolves under any registered name
_foreach_minimum_ = _foreach_minimum__Scalar
foreach_minimum_ = _foreach_minimum__Scalar
foreach_minimum_scalar_ = _foreach_minimum__Scalar
