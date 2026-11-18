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

import glob
import logging
import os

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._kunlunxin.ops.cumsum import cumsum_wrapper

logger = logging.getLogger(__name__)


def _bootstrap_raw_clang():
    """Point tle.raw at the packaged xtdk clang when the env is not set.

    The payload compiler is looked up once at compile time; set (or validate)
    the env var early so the production path never depends on the caller."""
    if "TRITON_XPU_CLANG_PATH" not in os.environ:
        try:
            triton_dir = os.path.dirname(os.path.abspath(triton.__file__))
            roots = [
                triton_dir,
                os.path.dirname(triton_dir),
                "/workspace/triton",
                "/workspace/triton-sdnn-build/python",
            ]
            cands = []
            for root in roots:
                deps = os.path.join(root, "_deps", "xtdk")
                cands += sorted(glob.glob(os.path.join(deps, "*/bin")))
            if cands:
                os.environ["TRITON_XPU_CLANG_PATH"] = cands[-1]
        except Exception:
            pass


_bootstrap_raw_clang()

_HERE = os.path.dirname(os.path.abspath(__file__))


@tle.raw.dialect("xpu3", file=os.path.join(_HERE, "cumprod_backward_raw.xpu"))
def cb_row_f32(grad, out, inp, gi, M, N): ...


@triton.jit(do_not_specialize=["M", "N"])
def _cb_raw_f32_kernel(Grad, Out, Inp, Gi, M, N):
    tle.raw.call(cb_row_f32, (Grad, Out, Inp, Gi, M, N))


# Payload dispatch thresholds (Round 2): tiny/medium rows are launch-bound in
# the Triton path; a single core-strided payload launch removes the per-program
# overhead and the zero-detection round-trip.  The payload handles tails and
# rows with zeros internally, so no host-side flag is needed on this branch.
_RAW_MAX_N = 512
_RAW_MAX_ELEMS = 65536

logger = logging.getLogger(__name__)

# Rows up to this length are served by the single-program per-row kernel (one
# pass, no intermediate buffers); longer rows use the go + cumsum prefix path.
_ROW_MAX_N = 2048
# Prefix-sum elementwise tile width.
_PF_BLOCK = 32768


@triton.jit
def _any_zero_partial_kernel(
    input_ptr, numel, partial_ptr, BLOCK: tl.constexpr, NEED_MASK: tl.constexpr
):
    # Grid-parallel pass 1: each program reduces one chunk and flags whether any
    # element in it is zero.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < numel
        oc = tl.where(mask, offs, 0)
        v = tl.load(input_ptr + oc).to(tl.float32)
        v = tl.where(mask, v, 1.0)
    else:
        v = tl.load(input_ptr + offs).to(tl.float32)
    has_zero = tl.sum((v == 0.0).to(tl.int32), axis=0) > 0
    tl.store(partial_ptr + pid, has_zero.to(tl.int32))


@triton.jit
def _any_zero_final_kernel(partial_ptr, num_partial, out_ptr, BLOCK: tl.constexpr):
    # Pass 2: reduce the per-chunk flags to a single tensor-level flag.  This
    # mirrors aten::cumprod_backward's tensor-level branch: when any zero is
    # present it recomputes the cumprod, otherwise it reuses the forward output.
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for start in range(0, num_partial, BLOCK):
        idx = start + offs
        mask = idx < num_partial
        idx_c = tl.where(mask, idx, 0)
        v = tl.load(partial_ptr + idx_c).to(tl.int32)
        v = tl.where(mask, v, 0)
        acc += v
    has_zero = tl.sum(acc, axis=0) > 0
    tl.store(out_ptr, has_zero.to(tl.int32))


# ---------------------------------------------------------------------------
# Fast path (no zeros anywhere):  grad_input = rev_cumsum(grad * out) / inp.
# ---------------------------------------------------------------------------


@triton.jit
def _bwd_row_kernel(
    grad_ptr,
    input_ptr,
    output_ptr,
    grad_input_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """One row per program, single 1D tile.  rev = total - inclusive_cumsum + go.

    This is the only tl.cumsum pattern this backend lowers correctly for a
    bounded, unmasked tile; no carry loop is used (a same-kernel carry mixing
    tl.sum with tl.cumsum mis-lowers on this backend)."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = pid * N
    if NEED_MASK:
        mask = offs < N
        oc = tl.where(mask, offs, 0)
        g = tl.load(grad_ptr + base + oc, mask=mask, other=0.0).to(tl.float32)
        o = tl.load(output_ptr + base + oc, mask=mask, other=0.0).to(tl.float32)
        inp = tl.load(input_ptr + base + oc, mask=mask, other=0.0).to(tl.float32)
        g = tl.where(mask, g, 0.0)
        o = tl.where(mask, o, 0.0)
        inp = tl.where(mask, inp, 0.0)
    else:
        g = tl.load(grad_ptr + base + offs).to(tl.float32)
        o = tl.load(output_ptr + base + offs).to(tl.float32)
        inp = tl.load(input_ptr + base + offs).to(tl.float32)
        mask = None
    go = g * o
    total = tl.sum(go, axis=0)
    inc = tl.cumsum(go, axis=0)
    rev = total - inc + go
    grad = rev / inp
    if NEED_MASK:
        tl.store(
            grad_input_ptr + base + offs,
            grad.to(grad_input_ptr.dtype.element_ty),
            mask=mask,
        )
    else:
        tl.store(grad_input_ptr + base + offs, grad.to(grad_input_ptr.dtype.element_ty))


@triton.jit
def _bwd_go_kernel(
    grad_ptr,
    output_ptr,
    input_ptr,
    go_ptr,
    zflag_ptr,
    numel,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Fused go = grad * output and per-program zero detection on input."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < numel
        oc = tl.where(mask, offs, 0)
        g = tl.load(grad_ptr + oc).to(tl.float32)
        o = tl.load(output_ptr + oc).to(tl.float32)
        v = tl.load(input_ptr + oc).to(tl.float32)
        tl.store(go_ptr + oc, g * o, mask=mask)
        v = tl.where(mask, v, 1.0)
    else:
        g = tl.load(grad_ptr + offs).to(tl.float32)
        o = tl.load(output_ptr + offs).to(tl.float32)
        v = tl.load(input_ptr + offs).to(tl.float32)
        tl.store(go_ptr + offs, g * o)
    has_z = tl.sum((v == 0.0).to(tl.int32), axis=0) > 0
    tl.store(zflag_ptr + pid, has_z.to(tl.int32))


@triton.jit
def _bwd_apply_kernel(
    go_ptr,
    prefix_ptr,
    input_ptr,
    grad_input_ptr,
    total_ptr,
    N,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """(M, cdiv(N, BLOCK)) grid.  rev = total - prefix + go; grad = rev / inp."""
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    base = pid_r * N
    if NEED_MASK:
        mask = offs < N
        oc = tl.where(mask, offs, 0)
        p = tl.load(prefix_ptr + base + oc).to(tl.float32)
        go = tl.load(go_ptr + base + oc).to(tl.float32)
        inp = tl.load(input_ptr + base + oc).to(tl.float32)
    else:
        mask = None
        p = tl.load(prefix_ptr + base + offs).to(tl.float32)
        go = tl.load(go_ptr + base + offs).to(tl.float32)
        inp = tl.load(input_ptr + base + offs).to(tl.float32)
    # Row total == prefix at the last column: read it directly, no copy.
    total = tl.load(total_ptr + pid_r * N + (N - 1)).to(tl.float32)
    rev = total - p + go
    grad = rev / inp
    if NEED_MASK:
        tl.store(
            grad_input_ptr + base + oc,
            grad.to(grad_input_ptr.dtype.element_ty),
            mask=mask,
        )
    else:
        tl.store(grad_input_ptr + base + offs, grad.to(grad_input_ptr.dtype.element_ty))


def _bwd_fast_2d(grad, input, output, N):
    """No-zero backward on contiguous (M, N) rows along the last dim."""
    M = input.shape[0]
    gi = torch.empty((M, N), dtype=grad.dtype, device=grad.device)
    if N <= _ROW_MAX_N:
        BLOCK = max(16, triton.next_power_of_2(N))
        need_mask = 1 if BLOCK != N else 0
        num_warps = 8 if BLOCK > 2048 else 4
        with torch_device_fn.device(input.device):
            _bwd_row_kernel[(M,)](
                grad,
                input,
                output,
                gi,
                N=N,
                BLOCK=BLOCK,
                NEED_MASK=need_mask,
                num_warps=num_warps,
                buffer_size_limit=2048,
            )
        return gi
    numel = M * N
    need_mask_g = 1 if numel % _PF_BLOCK else 0
    go = torch.empty((M, N), dtype=torch.float32, device=input.device)
    prefix = torch.empty((M, N), dtype=torch.float32, device=input.device)
    num_chunks = triton.cdiv(numel, _PF_BLOCK)
    zflag = torch.empty(num_chunks, dtype=torch.int32, device=input.device)
    with torch_device_fn.device(input.device):
        _bwd_go_kernel[(num_chunks,)](
            grad,
            output,
            input,
            go,
            zflag,
            numel,
            BLOCK=_PF_BLOCK,
            NEED_MASK=need_mask_g,
            num_warps=8,
            buffer_size_limit=2048,
        )
        # Inclusive prefix sum via the production cumsum kernels (matmul
        # prefix-sum / row / group paths, all proven on this backend).
        cumsum_wrapper(go, 1, out=prefix)
        ab = _PF_BLOCK
        if N % ab:
            ab = triton.next_power_of_2(N)
        need_mask_a = 1 if N % ab else 0
        _bwd_apply_kernel[(M, triton.cdiv(N, ab))](
            go,
            prefix,
            input,
            gi,
            prefix,
            N,
            BLOCK=ab,
            NEED_MASK=need_mask_a,
            num_warps=8,
            buffer_size_limit=2048,
        )
    return gi


# ---------------------------------------------------------------------------
# Zero-handling path (HAS_ZERO=1): mirror aten's tensor-level branch.  When any
# zero exists the whole tensor is recomputed from the forward-cumprod semantics.
# ---------------------------------------------------------------------------


@triton.jit
def _cumprod_backward_zero_kernel(
    grad_ptr,
    input_ptr,
    output_ptr,
    grad_input_ptr,
    has_zero_ptr,
    N,
    stride_d,
    stride_b,
    BLOCK: tl.constexpr,
):
    """One program per row (grid == batch size); single 1D tile.  This is the
    proven zero-handling kernel (it mirrors aten's tensor-level branch); all
    scans live outside any runtime if."""
    pid = tl.program_id(0)
    base = pid * stride_b

    offs = tl.arange(0, BLOCK)
    mask = offs < N
    # Clamp masked lanes to index 0 (this backend ignores tl.load(other=) and
    # would otherwise read neighboring memory into the reductions below).
    offs_clamped = tl.where(mask, offs, 0)

    g = tl.load(grad_ptr + base + offs_clamped * stride_d, cache_modifier=".ca").to(
        tl.float32
    )
    g = tl.where(mask, g, 0.0)
    inp = tl.load(input_ptr + base + offs_clamped * stride_d, cache_modifier=".ca").to(
        tl.float32
    )
    inp = tl.where(mask, inp, 0.0)
    out_bf16 = tl.load(
        output_ptr + base + offs_clamped * stride_d, cache_modifier=".ca"
    ).to(tl.float32)
    out_bf16 = tl.where(mask, out_bf16, 0.0)

    # Locate the first zero in this row (k = index, or 2147483647 if none).
    is_zero = (inp == 0.0) & mask
    large_val = 2147483647
    zero_indices = tl.where(is_zero, offs.to(tl.int32), large_val)
    k = tl.min(zero_indices, axis=0)

    has_zero = tl.load(has_zero_ptr).to(tl.int32)
    out_fp32 = tl.cumprod(inp, axis=0)
    out = tl.where(has_zero != 0, out_fp32, out_bf16)

    go = g * out

    total_go = tl.sum(go, axis=0)
    inc_cs = tl.cumsum(go, axis=0)
    rev_cs = total_go - inc_cs + go

    safe_inp = tl.where(inp == 0.0, 1.0, inp)
    grad_input = tl.where(mask, rev_cs / safe_inp, 0.0)
    grad_input = tl.where(inp == 0.0, 0.0, grad_input)

    # Zero handling is computed unconditionally (a scan must never sit inside a
    # runtime scf.if on this backend).  When there is no zero, k == large_val so
    # `offs == k` is false everywhere and the extra results are never selected.
    inp_mod = tl.where(offs.to(tl.int32) == k, 1.0, inp)
    inp_mod = tl.where(mask, inp_mod, 1.0)
    cumprod_mod = tl.cumprod(inp_mod, axis=0)

    gcm = g * cumprod_mod
    zero_val = tl.sum(tl.where(offs.to(tl.int32) >= k, gcm, 0.0), axis=0)

    grad_input = tl.where(offs.to(tl.int32) > k, 0.0, grad_input)
    grad_input = tl.where(offs.to(tl.int32) == k, zero_val, grad_input)

    tl.store(
        grad_input_ptr + base + offs * stride_d,
        grad_input.to(tl.float32),
        mask=mask,
        cache_modifier=".cs",
    )


def _bwd_zero_2d(grad, input, output, N, has_zero_buf):
    """Zero-handling path on normalized (M, N) contiguous rows."""
    M = input.shape[0]
    gi = torch.empty((M, N), dtype=grad.dtype, device=grad.device)
    BLOCK = 16
    while BLOCK < N:
        BLOCK *= 2
    if BLOCK <= 32:
        num_warps = 1
    elif BLOCK <= 64:
        num_warps = 2
    elif BLOCK <= 128:
        num_warps = 4
    else:
        num_warps = 8
    with torch_device_fn.device(input.device):
        _cumprod_backward_zero_kernel[(M,)](
            grad,
            input,
            output,
            gi,
            has_zero_buf,
            N,
            1,
            N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return gi


def cumprod_backward(grad, input, dim, output):
    logger.debug("GEMS_KUNLUNXIN CUMPROD_BACKWARD")
    ndim = input.ndim
    dim = int(dim)
    if dim < 0:
        dim += ndim

    N = input.shape[dim]

    grad_input = torch.empty_like(grad)
    if grad.numel() == 0 or input.numel() == 0:
        return grad_input

    # Tensor-level zero detection (mirrors aten::cumprod_backward's branch).
    has_zero_buf = torch.empty(1, dtype=torch.int32, device=input.device)
    grad2 = grad.movedim(dim, -1).reshape(-1, N)
    inp2 = input.movedim(dim, -1).reshape(-1, N)
    out2 = output.movedim(dim, -1).reshape(-1, N)
    if not grad2.is_contiguous():
        grad2 = grad2.contiguous()
    if not inp2.is_contiguous():
        inp2 = inp2.contiguous()
    if not out2.is_contiguous():
        out2 = out2.contiguous()

    M2 = grad2.shape[0]
    if N <= _RAW_MAX_N and M2 * N <= _RAW_MAX_ELEMS:
        gi2 = torch.empty((M2, N), dtype=grad.dtype, device=grad.device)
        used = False
        with torch_device_fn.device(input.device):
            # Only fp32 uses the payload for now: for bf16/f16 the tiny-N
            # zero-handling difference vs the fp64 reference stays within the
            # verified Triton kernels, and the fp32 payload is bit-level
            # validated (incl. zero rows).
            if grad.dtype is torch.float32:
                _cb_raw_f32_kernel[(1,)](grad2, out2, inp2, gi2, M2, N)
                used = True
        if used:
            gi = gi2.reshape([input.shape[i] for i in range(ndim) if i != dim] + [N])
            gi = gi.movedim(-1, dim)
            if not gi.is_contiguous():
                gi = gi.contiguous()
            return gi

    # Dedicated tensor-level zero detection (the proven two-kernel reduction).
    inp_flat = inp2.reshape(-1)
    numel = inp_flat.numel()
    RED_BLOCK = 16384 if numel >= 8 * 1024 * 1024 else 4096
    num_blocks = (numel + RED_BLOCK - 1) // RED_BLOCK
    partial = torch.empty(num_blocks, dtype=torch.int32, device=input.device)
    need_mask = 1 if numel % RED_BLOCK else 0
    with torch_device_fn.device(input.device):
        _any_zero_partial_kernel[(num_blocks,)](
            inp_flat,
            numel,
            partial,
            BLOCK=RED_BLOCK,
            NEED_MASK=need_mask,
        )
        _any_zero_final_kernel[(1,)](partial, num_blocks, has_zero_buf, BLOCK=1024)
    if int(has_zero_buf.item()) != 0:
        gi2 = _bwd_zero_2d(grad2, inp2, out2, N, has_zero_buf)
    else:
        gi2 = _bwd_fast_2d(grad2, inp2, out2, N)

    gi = gi2.reshape([input.shape[i] for i in range(ndim) if i != dim] + [N])
    gi = gi.movedim(-1, dim)
    if not gi.is_contiguous():
        gi = gi.contiguous()
    return gi
