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
#
# Kunlunxin XPU3 specialization of the Moore-Penrose pseudoinverse.
#
# HEAD (generic ``flag_gems/ops/linalg_pinv.py``) is functionally correct on
# this backend: for square ``m == n <= 8`` it runs the register-resident
# one-sided Jacobi kernel (``_pinv_reg_kernel``) and contracts with
# ``torch.matmul``; larger / rectangular inputs take the blocked path.  Two
# measured properties motivate this override:
#
#   * the generic register path hard-codes 10 sweeps for ``C <= 4``, but on
#     XPU3 four cyclic sweeps already converge for 4x4 (a conditioning /
#     batch / seed sweep from cond 1 to 1e6 shows the 4-sweep result is never
#     worse than the 10-sweep one -- extra sweeps only add fp32 drift), and
#   * the pinv contraction can be folded into the same kernel launch: once the
#     rotated columns ``w_j`` and the rotation matrix ``V`` are in registers,
#     ``pinv[i, l] = sum_j V[i, j] * w_j[l] / sigma_j^2`` is a handful of 1-D
#     FMAs, so no second launch (and no host-side ``torch.matmul``) is needed.
#
# Both together cut a 4x4 call from ~40/50/118 us (batch 1/4/32) to
# ~23/29/67 us.  The 8x8 case is deliberately NOT specialized: it needs 6-8
# sweeps for cond >= 10 and the fused contraction is marginally slower there,
# so it keeps the generic implementation unchanged.

import importlib
import linecache
import logging

import torch
import triton

logger = logging.getLogger(__name__)

# The generic entry point is resolved from the ``flag_gems.ops`` *package*
# attribute: it is bound once by ``ops/__init__.py``'s
# ``from flag_gems.ops.linalg_pinv import linalg_pinv`` and is left untouched by
# the module-attribute rebind at the tail of this file.  The submodule
# attribute, by contrast, may already have been rebound by an earlier import of
# this backend module -- it is loaded under both ``_kunlunxin.ops.linalg_pinv``
# and ``flag_gems.runtime.backend._kunlunxin.ops.linalg_pinv``.
_OPS_PACKAGE = importlib.import_module("flag_gems.ops")
_GENERIC_MODULE = importlib.import_module("flag_gems.ops.linalg_pinv")
_GENERIC_PINV = getattr(_OPS_PACKAGE, "linalg_pinv", None)
if not (
    callable(_GENERIC_PINV)
    and getattr(_GENERIC_PINV, "__module__", "") == "flag_gems.ops.linalg_pinv"
):
    _GENERIC_PINV = _GENERIC_MODULE.linalg_pinv

# Largest square order handled by the fused kernel. Larger / rectangular
# matrices -- and 8x8, which is not converged after 4 sweeps -- fall through to
# the generic implementation.
_FUSED_MAX_N = 4
# Cyclic one-sided Jacobi on a 4x4 Gram matrix converges in 3-4 sweeps; four is
# the measured fixed point (see the module docstring).
_FUSED_SWEEPS = 4

_KERNELS = {}


def _kernel_source(C, sweeps):
    """Generate the fused Jacobi + contraction kernel for a C x C batch element.

    Triton has no list indexing / ``__setitem__`` over tensors, so the column
    variables ``w0..w{C-1}`` and the rotation rows ``v0..v{C-1}`` are emitted as
    plain registers and the whole sweep is unrolled.  Every reduction runs on a
    1-D register vector: the backend rejects ``tl.reduce`` over a 2-D tile
    (``arith.cmpi`` type errors / uni_sram exhaustion).
    """
    BLOCK = triton.next_power_of_2(C)
    lines = []

    def emit(line=""):
        lines.append(line)

    emit("import triton")
    emit("import triton.language as tl")
    emit("")
    emit("")
    emit("@triton.jit")
    emit("def _pinv_square_fused_kernel(A_ptr, OUT_ptr, atol_val, rtol_val):")
    emit("    pid = tl.program_id(0)")
    emit(f"    base = pid * {C * C}")
    emit(f"    rs = tl.arange(0, {BLOCK})")
    emit(f"    rmask = rs < {C}")
    # W holds the columns of A; column j is A[:, j] = base + r * C + j.
    for j in range(C):
        emit(f"    w{j} = tl.load(A_ptr + base + rs * {C} + {j}, mask=rmask, other=0.0)")
    for j in range(C):
        emit(f"    v{j} = tl.where(rs == {j}, 1.0, 0.0)")
        emit(f"    sig{j} = tl.zeros([], dtype=tl.float32)")

    # Fixed sweep count: no device-to-host sync per sweep.  sig_j is refreshed
    # at the end of every sweep because a reduction inside the runtime loop is
    # legal, while reducing a loop-carried value afterwards is not.
    emit(f"    for _sweep in tl.range(0, {sweeps}):")
    for p in range(C):
        for q in range(p + 1, C):
            emit(f"        alpha = tl.sum(w{p} * w{p})")
            emit(f"        beta = tl.sum(w{q} * w{q})")
            emit(f"        gamma = tl.sum(w{p} * w{q})")
            emit("        scale = tl.maximum(alpha, beta) + 1e-30")
            emit("        rot = tl.abs(gamma) > 1e-10 * scale")
            emit("        tau = (beta - alpha) / (2.0 * gamma + 1e-30)")
            emit("        sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)")
            emit("        t = sign_tau / (tl.abs(tau) + tl.sqrt(tau * tau + 1.0))")
            emit("        c = 1.0 / tl.sqrt(1.0 + t * t)")
            emit("        s = t * c")
            emit("        c = tl.where(rot, c, 1.0)")
            emit("        s = tl.where(rot, s, 0.0)")
            for prefix in ("w", "v"):
                emit(f"        cur_p = {prefix}{p}")
                emit(f"        cur_q = {prefix}{q}")
                emit(f"        {prefix}{p} = c * cur_p - s * cur_q")
                emit(f"        {prefix}{q} = s * cur_p + c * cur_q")
    for j in range(C):
        emit(f"        sig{j} = tl.sqrt(tl.sum(w{j} * w{j}))")

    # Singular-value cutoff, matching torch semantics:
    #   cutoff = max(atol, rtol * sigma_max)
    emit("    sigma_max = sig0")
    for j in range(1, C):
        emit(f"    sigma_max = tl.maximum(sigma_max, sig{j})")
    emit("    thr = tl.maximum(atol_val, sigma_max * rtol_val)")
    for j in range(C):
        emit(f"    inv{j} = tl.where(sig{j} > thr, 1.0 / (sig{j} * sig{j} + 1e-38), 0.0)")

    # pinv[i, l] = sum_j V[i, j] * w_j[l] / sigma_j^2.  Lane i of V[:, j] is
    # extracted with a 1-D masked reduction, then accumulated as C FMAs per
    # output row -- no 2-D tile, no second launch.
    for i in range(C):
        terms = []
        for j in range(C):
            emit(f"    v{i}_{j} = tl.sum(tl.where(rs == {i}, v{j}, 0.0)) * inv{j}")
            terms.append(f"v{i}_{j} * w{j}")
        emit(f"    row{i} = " + " + ".join(terms))
        emit(f"    tl.store(OUT_ptr + base + {i * C} + rs, row{i}, mask=rmask)")

    return "\n".join(lines) + "\n"


def _fused_kernel(C):
    kernel = _KERNELS.get(C)
    if kernel is None:
        source = _kernel_source(C, _FUSED_SWEEPS)
        name = f"<flag_gems_kunlunxin_pinv_square_{C}>"
        linecache.cache[name] = (len(source), None, source.splitlines(True), name)
        namespace = {}
        exec(compile(source, name, "exec"), namespace)
        kernel = namespace["_pinv_square_fused_kernel"]
        _KERNELS[C] = kernel
    return kernel


def _fused_pinv(A_flat, batch, n, atol_val, rtol_val):
    out = torch.empty(batch, n, n, dtype=torch.float32, device=A_flat.device)
    _fused_kernel(n)[(batch,)](A_flat, out, atol_val, rtol_val)
    return out


def _resolve_tolerances(A, m, n, atol, rtol):
    atol_val = 0.0 if atol is None else float(atol)
    if rtol is None:
        eps = torch.finfo(A.dtype).eps
        rtol_val = 0.0 if atol_val > 0.0 else float(max(m, n)) * eps
    else:
        rtol_val = float(rtol)
    return atol_val, rtol_val


def linalg_pinv(
    A: torch.Tensor,
    *,
    atol: float = None,
    rtol: float = None,
    hermitian: bool = False,
) -> torch.Tensor:
    """Moore-Penrose pseudoinverse (Kunlunxin fast path for small square input)."""
    logger.debug("GEMS_KUNLUNXIN LINALG_PINV")

    assert A.dtype in (
        torch.float32,
        torch.float64,
    ), f"linalg_pinv: expected float32 or float64, got {A.dtype}"

    if not A.is_contiguous():
        A = A.contiguous()

    orig_shape = A.shape
    m = orig_shape[-2]
    n = orig_shape[-1]

    if A.dtype == torch.float32 and m == n and 0 < m <= _FUSED_MAX_N:
        batch = 1
        for d in orig_shape[:-2]:
            batch *= d
        if batch > 0:
            A_flat = A.reshape(batch, m, n)
            atol_val, rtol_val = _resolve_tolerances(A, m, n, atol, rtol)
            out = _fused_pinv(A_flat, batch, n, atol_val, rtol_val)
            return out.reshape(list(orig_shape[:-2]) + [n, m])

    # Rectangular, larger, fp64, or empty inputs: delegate to the generic
    # implementation (same Jacobi algorithm, pure FlagGems ops).
    return _GENERIC_PINV(A, atol=atol, rtol=rtol, hermitian=hermitian)


# ``benchmark/test_linalg_pinv.py`` binds its candidate with
# ``from flag_gems.ops.linalg_pinv import linalg_pinv``, which reads the generic
# module attribute and therefore bypasses both the SpecOpRegistrar rewrite of
# ``flag_gems.<op>`` and the ATen registration.  Rebind the module attribute so
# the benchmark (and any other explicit generic-module import) picks up this
# backend implementation; this is the same pattern other vendors use
# (``_enflame/gcu300/ops/div.py``).
if _GENERIC_MODULE.linalg_pinv is not linalg_pinv:
    _GENERIC_MODULE.linalg_pinv = linalg_pinv
