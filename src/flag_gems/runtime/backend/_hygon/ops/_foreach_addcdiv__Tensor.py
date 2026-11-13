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

_BLOCK = 2048
_WARPS = 4
_MAX_T = 16
# Below this many total elements a call is host-launch bound, so a single fused
# launch wins; above it the per-entry launches stay device bound and are faster.
_FUSED_MAX_TOTAL = 1 << 21

_LEAN_OK = True
_RUNNERS = {}


@triton.jit
def _addcdiv_kernel(
    self_ptr,
    t1_ptr,
    t2_ptr,
    alpha_ptr,
    alpha_val,
    n_elem,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    HAS_ALPHA_PTR: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        a = tl.load(self_ptr + offs)
        b = tl.load(t1_ptr + offs)
        c = tl.load(t2_ptr + offs)
    else:
        mask = offs < n_elem
        a = tl.load(self_ptr + offs, mask=mask, other=0.0)
        b = tl.load(t1_ptr + offs, mask=mask, other=0.0)
        c = tl.load(t2_ptr + offs, mask=mask, other=0.0)

    if HAS_ALPHA_PTR:
        alpha = tl.load(alpha_ptr).to(tl.float32)
    else:
        alpha = alpha_val

    af = a.to(tl.float32)
    bf = b.to(tl.float32)
    cf = c.to(tl.float32)
    out = af + alpha * (bf / cf)
    res = out.to(a.dtype)

    if EVEN:
        tl.store(self_ptr + offs, res)
    else:
        tl.store(self_ptr + offs, res, mask=mask)


_TUNE_BLOCKS = (1024, 2048, 4096, 8192)
_TUNE_CONFIGS = [
    triton.Config({"BLOCK": _b}, num_warps=_w) for _b in _TUNE_BLOCKS for _w in (4, 8)
]


@triton.autotune(
    configs=_TUNE_CONFIGS,
    key=["n_elem", "HAS_ALPHA_PTR"],
    restore_value=["self_ptr"],
)
@triton.jit
def _addcdiv_tuned_kernel(
    self_ptr,
    t1_ptr,
    t2_ptr,
    alpha_ptr,
    alpha_val,
    n_elem,
    BLOCK: tl.constexpr,
    HAS_ALPHA_PTR: tl.constexpr,
):
    # Only routed to sizes divisible by the largest candidate BLOCK, so the
    # tail mask is never needed and every candidate config is in bounds.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(self_ptr + offs)
    b = tl.load(t1_ptr + offs)
    c = tl.load(t2_ptr + offs)

    if HAS_ALPHA_PTR:
        alpha = tl.load(alpha_ptr).to(tl.float32)
    else:
        alpha = alpha_val

    af = a.to(tl.float32)
    bf = b.to(tl.float32)
    cf = c.to(tl.float32)
    out = af + alpha * (bf / cf)
    tl.store(self_ptr + offs, out.to(a.dtype))


@triton.jit
def _fused_addcdiv_kernel(
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
    u0,
    u1,
    u2,
    u3,
    u4,
    u5,
    u6,
    u7,
    u8,
    u9,
    u10,
    u11,
    u12,
    u13,
    u14,
    u15,
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    v8,
    v9,
    v10,
    v11,
    v12,
    v13,
    v14,
    v15,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15,
    n_elem,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.program_id(1)

    sp = s0
    up = u0
    vp = v0
    alpha = a0
    if NT > 1:
        sp = tl.where(tid == 1, s1, sp)
        up = tl.where(tid == 1, u1, up)
        vp = tl.where(tid == 1, v1, vp)
        alpha = tl.where(tid == 1, a1, alpha)
    if NT > 2:
        sp = tl.where(tid == 2, s2, sp)
        up = tl.where(tid == 2, u2, up)
        vp = tl.where(tid == 2, v2, vp)
        alpha = tl.where(tid == 2, a2, alpha)
    if NT > 3:
        sp = tl.where(tid == 3, s3, sp)
        up = tl.where(tid == 3, u3, up)
        vp = tl.where(tid == 3, v3, vp)
        alpha = tl.where(tid == 3, a3, alpha)
    if NT > 4:
        sp = tl.where(tid == 4, s4, sp)
        up = tl.where(tid == 4, u4, up)
        vp = tl.where(tid == 4, v4, vp)
        alpha = tl.where(tid == 4, a4, alpha)
    if NT > 5:
        sp = tl.where(tid == 5, s5, sp)
        up = tl.where(tid == 5, u5, up)
        vp = tl.where(tid == 5, v5, vp)
        alpha = tl.where(tid == 5, a5, alpha)
    if NT > 6:
        sp = tl.where(tid == 6, s6, sp)
        up = tl.where(tid == 6, u6, up)
        vp = tl.where(tid == 6, v6, vp)
        alpha = tl.where(tid == 6, a6, alpha)
    if NT > 7:
        sp = tl.where(tid == 7, s7, sp)
        up = tl.where(tid == 7, u7, up)
        vp = tl.where(tid == 7, v7, vp)
        alpha = tl.where(tid == 7, a7, alpha)
    if NT > 8:
        sp = tl.where(tid == 8, s8, sp)
        up = tl.where(tid == 8, u8, up)
        vp = tl.where(tid == 8, v8, vp)
        alpha = tl.where(tid == 8, a8, alpha)
    if NT > 9:
        sp = tl.where(tid == 9, s9, sp)
        up = tl.where(tid == 9, u9, up)
        vp = tl.where(tid == 9, v9, vp)
        alpha = tl.where(tid == 9, a9, alpha)
    if NT > 10:
        sp = tl.where(tid == 10, s10, sp)
        up = tl.where(tid == 10, u10, up)
        vp = tl.where(tid == 10, v10, vp)
        alpha = tl.where(tid == 10, a10, alpha)
    if NT > 11:
        sp = tl.where(tid == 11, s11, sp)
        up = tl.where(tid == 11, u11, up)
        vp = tl.where(tid == 11, v11, vp)
        alpha = tl.where(tid == 11, a11, alpha)
    if NT > 12:
        sp = tl.where(tid == 12, s12, sp)
        up = tl.where(tid == 12, u12, up)
        vp = tl.where(tid == 12, v12, vp)
        alpha = tl.where(tid == 12, a12, alpha)
    if NT > 13:
        sp = tl.where(tid == 13, s13, sp)
        up = tl.where(tid == 13, u13, up)
        vp = tl.where(tid == 13, v13, vp)
        alpha = tl.where(tid == 13, a13, alpha)
    if NT > 14:
        sp = tl.where(tid == 14, s14, sp)
        up = tl.where(tid == 14, u14, up)
        vp = tl.where(tid == 14, v14, vp)
        alpha = tl.where(tid == 14, a14, alpha)
    if NT > 15:
        sp = tl.where(tid == 15, s15, sp)
        up = tl.where(tid == 15, u15, up)
        vp = tl.where(tid == 15, v15, vp)
        alpha = tl.where(tid == 15, a15, alpha)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        a = tl.load(sp + offs)
        b = tl.load(up + offs)
        c = tl.load(vp + offs)
    else:
        mask = offs < n_elem
        a = tl.load(sp + offs, mask=mask, other=0.0)
        b = tl.load(up + offs, mask=mask, other=0.0)
        c = tl.load(vp + offs, mask=mask, other=0.0)

    af = a.to(tl.float32)
    bf = b.to(tl.float32)
    cf = c.to(tl.float32)
    out = af + alpha * (bf / cf)
    res = out.to(a.dtype)

    if EVEN:
        tl.store(sp + offs, res)
    else:
        tl.store(sp + offs, res, mask=mask)


def _calibrate(kernel, args, grid3, constexprs, key):
    """Build a lean launcher and perform this call's launch with it.

    Returns True when the launch has been performed. The lean launcher skips
    JITFunction.run's per-argument python binding. Some backends append extra
    placeholder arguments to the launcher signature, so the trailing count is
    calibrated by trying counts in order; a wrong count raises before any work
    is enqueued, and the successful attempt is this call's single launch.
    """
    global _LEAN_OK
    try:
        compiled = kernel.warmup(*args, grid=grid3, num_warps=_WARPS, **constexprs)
        if compiled is None:
            _LEAN_OK = False
            return False
        runner = compiled[grid3]
    except Exception:
        _LEAN_OK = False
        return False
    for n_extra in range(5):
        try:
            if n_extra:
                runner(*(list(args) + [0] * n_extra))
            else:
                runner(*args)
        except TypeError:
            continue
        except Exception:
            _LEAN_OK = False
            return False
        _RUNNERS[key] = (runner, [0] * n_extra)
        return True
    _LEAN_OK = False
    return False


def _lean_launch(kernel, args, grid3, constexprs, key):
    """Return True when the lean launcher performed the launch."""
    if _LEAN_OK:
        entry = _RUNNERS.get(key)
        if entry is None:
            return _calibrate(kernel, args, grid3, constexprs, key)
        runner, zeros = entry
        if zeros:
            args.extend(zeros)
        runner(*args)
        return True
    return False


def _normalize_scalars(scalars, n):
    """Return (alphas, all_host_floats) with len(alphas) == n.

    Each entry is either a python float or a 0-dim tensor.
    """
    if isinstance(scalars, torch.Tensor):
        if scalars.dim() == 0:
            return [scalars] * n, False
        if scalars.numel() == 1:
            return [scalars.reshape(-1)[0]] * n, False
        if not scalars.is_cuda:
            vals = scalars.tolist()
            m = len(vals)
            if m == 0:
                return [0.0] * n, True
            if m == n:
                return vals, True
            if m < n:
                return [vals[i % m] for i in range(n)], True
            return vals[:n], True
        if scalars.numel() >= n:
            return [scalars[i] for i in range(n)], False
        m = scalars.numel()
        return [scalars[i % m] for i in range(n)], False
    if isinstance(scalars, (list, tuple)):
        vals = []
        is_float = True
        for s in scalars:
            if isinstance(s, torch.Tensor):
                if s.dim() != 0:
                    s = s.reshape(-1)[0]
                vals.append(s)
                is_float = False
            else:
                vals.append(float(s))
        m = len(vals)
        if m == 0:
            return [0.0] * n, True
        if m == 1 and n != 1:
            return vals * n, is_float
        if m < n:
            return [vals[i % m] for i in range(n)], is_float
        return vals[:n], is_float
    return [float(scalars)] * n, True


def _foreach_addcdiv__Tensor(self, tensor1, tensor2, scalars):
    logger.debug("GEMS_HYGON _FOREACH_ADDCDIV__TENSOR")
    self_list = self
    n = len(self_list)
    if n == 0:
        return None

    alphas = None
    if type(scalars) is torch.Tensor and scalars.dim() == 1 and not scalars.is_cuda:
        # Common case: scalars is a CPU 1-D tensor with one value per entry.
        alphas = scalars.tolist()
        if len(alphas) != n:
            alphas = None
        else:
            host_floats = True
    if alphas is None:
        alphas, host_floats = _normalize_scalars(scalars, n)

    # Fast path: one fused launch for a small homogeneous tensor list.
    if host_floats and 1 < n <= _MAX_T:
        e0 = self_list[0]
        n_elem = e0.numel()
        if 0 < n_elem and n * n_elem <= _FUSED_MAX_TOTAL:
            dt = e0.dtype
            dt1 = tensor1[0].dtype
            dt2 = tensor2[0].dtype
            # Alignment is only used as a cache-key bit, so sample it once per
            # list instead of once per entry.
            addr = e0.data_ptr() | tensor1[0].data_ptr() | tensor2[0].data_ptr()
            fused = True
            for e, b, c in zip(self_list, tensor1, tensor2):
                if (
                    e.numel() != n_elem
                    or e.dtype is not dt
                    or not e.is_contiguous()
                    or b.numel() != n_elem
                    or b.dtype is not dt1
                    or not b.is_contiguous()
                    or c.numel() != n_elem
                    or c.dtype is not dt2
                    or not c.is_contiguous()
                ):
                    fused = False
                    break
            if fused:
                if n == _MAX_T:
                    args = (
                        list(self_list)
                        + list(tensor1)
                        + list(tensor2)
                        + alphas
                        + [n_elem]
                    )
                else:
                    pad = e0
                    args = [self_list[i] if i < n else pad for i in range(_MAX_T)]
                    args += [tensor1[i] if i < n else pad for i in range(_MAX_T)]
                    args += [tensor2[i] if i < n else pad for i in range(_MAX_T)]
                    args += [alphas[i] if i < n else 0.0 for i in range(_MAX_T)]
                    args.append(n_elem)
                even = (n_elem & (_BLOCK - 1)) == 0
                gx = (n_elem + _BLOCK - 1) // _BLOCK
                key = (
                    0,
                    dt,
                    dt1,
                    dt2,
                    n,
                    gx,
                    even,
                    (n_elem & 15) == 0,
                    n_elem == 1,
                    (addr & 15) == 0,
                )
                grid = (gx, n, 1)
                if not _lean_launch(
                    _fused_addcdiv_kernel,
                    args,
                    grid,
                    {"NT": n, "BLOCK": _BLOCK, "EVEN": even},
                    key,
                ):
                    _fused_addcdiv_kernel[grid](
                        *args, num_warps=_WARPS, NT=n, BLOCK=_BLOCK, EVEN=even
                    )
                return None

    for i in range(n):
        out_t = self_list[i]
        n_elem = out_t.numel()
        if n_elem == 0:
            continue
        t1 = tensor1[i]
        t2 = tensor2[i]

        alpha = alphas[i]
        if isinstance(alpha, torch.Tensor) and alpha.is_cuda:
            alpha_ptr = alpha
            alpha_val = 0.0
            has_ptr = True
        else:
            if isinstance(alpha, torch.Tensor):
                alpha_val = float(alpha.item())
            else:
                alpha_val = float(alpha)
            alpha_ptr = out_t
            has_ptr = False

        even = (n_elem & (_BLOCK - 1)) == 0
        gx = (n_elem + _BLOCK - 1) // _BLOCK
        if (n_elem & 8191) == 0:
            # Device-bound sizes: let the bounded autotuner pick BLOCK/warps.
            _addcdiv_tuned_kernel[
                lambda META: ((n_elem + META["BLOCK"] - 1) // META["BLOCK"], 1, 1)
            ](
                out_t,
                t1,
                t2,
                alpha_ptr,
                alpha_val,
                n_elem,
                HAS_ALPHA_PTR=has_ptr,
            )
            continue
        addr = out_t.data_ptr() | t1.data_ptr() | t2.data_ptr() | alpha_ptr.data_ptr()
        key = (
            1,
            out_t.dtype,
            t1.dtype,
            t2.dtype,
            alpha_ptr.dtype,
            has_ptr,
            even,
            gx,
            (n_elem & 15) == 0,
            n_elem == 1,
            (addr & 15) == 0,
        )
        grid = (gx, 1, 1)
        sargs = [out_t, t1, t2, alpha_ptr, alpha_val, n_elem]
        if not _lean_launch(
            _addcdiv_kernel,
            sargs,
            grid,
            {"BLOCK": _BLOCK, "EVEN": even, "HAS_ALPHA_PTR": has_ptr},
            key,
        ):
            _addcdiv_kernel[grid](
                *sargs,
                num_warps=_WARPS,
                BLOCK=_BLOCK,
                EVEN=even,
                HAS_ALPHA_PTR=has_ptr,
            )

    return None
