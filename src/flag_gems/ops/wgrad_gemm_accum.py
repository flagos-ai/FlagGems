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

"""Weight-gradient GEMM with in-place accumulation (Apex-aligned).

Matches Apex ``fused_weight_gradient_mlp_cuda`` semantics used by Megatron
``LinearWithGradAccumulationAndAsyncCommunication`` when
``gradient_accumulation_fusion`` is enabled.

Each update performs ``main_grad += grad_output.T @ input`` (after collapsing
leading dimensions).

``wgrad_gemm_accum_fp32`` (including half/bf16 activations into fp32
``main_grad``) calls ``cublasGemmEx`` with the same layout / dtype / algo as
Apex, via a compiled CUDA extension (FlagGems ``c_operators`` when present,
otherwise ``torch.utils.cpp_extension`` JIT).  If that path is unavailable,
Torch fp32 matmul is used as a **loud** fallback (``logger.error`` +
``warnings.warn`` once).  A single GemmEx *runtime* failure only falls back
for that call and retries next time; permanent fallback needs consecutive
failures (default 3, overridable via ``FLAGGEMS_WGRAD_GEMMEX_FAIL_LIMIT``).
Set ``FLAGGEMS_WGRAD_REQUIRE_GEMMEX=1`` to raise instead.

Non-contiguous ``main_grad``: a transpose view of a contiguous buffer is a
fast path (no densify).  General NC still densifies then ``copy_`` and warns
once.  Set ``FLAGGEMS_WGRAD_REQUIRE_CONTIGUOUS_MAIN_GRAD=1`` to reject any
non-``is_contiguous()`` ``main_grad``.  Training callers should prefer
``ensure_contiguous_main_grad`` once when allocating / binding ``main_grad``,
then reuse the contiguous buffer every step.

TF32 note: like Apex, ``allow_tf32=False`` does not force full-fp32 GemmEx.
Test-only CPU-fp64-matched checks use ``wgrad_gemm_accum_fp32_strict_cpu_ref``
(not a kwarg on the training API).

``wgrad_gemm_accum_fp16`` uses ``torch.addmm`` (cuBLAS) for same-dtype
accumulation.
"""

from __future__ import annotations

import logging
import os
import warnings
from functools import lru_cache
from pathlib import Path

import torch

import flag_gems

logger = logging.getLogger(__name__)

# Permanent GemmEx disable (missing ext, or too many consecutive runtime fails).
_WGRAD_EXT_RUNTIME_OK = True
_WGRAD_FALLBACK_ACTIVE = False
_WGRAD_FALLBACK_REASON: str | None = None
_WGRAD_FALLBACK_WARNED = False
# Consecutive GemmEx runtime failures; reset on the next successful call.
_WGRAD_RUNTIME_FAIL_STREAK = 0
_DEFAULT_GEMMEX_FAIL_LIMIT = 3
_WGRAD_NC_MAIN_WARNED = False


def _env_flag_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "on", "yes")


def _require_gemmex() -> bool:
    """Hard-fail instead of Torch fallback when env is set.

    ``FLAGGEMS_WGRAD_REQUIRE_GEMMEX=1|true|on|yes`` is for training / bench /
    local strict runs so a missing GemmEx path cannot silently become a slow,
    non-Apex-aligned Torch matmul.
    """
    return _env_flag_true("FLAGGEMS_WGRAD_REQUIRE_GEMMEX")


def _require_contiguous_main_grad() -> bool:
    """Reject any non-``is_contiguous()`` ``main_grad`` when env is set.

    Fail-fast for training jobs that want Apex-style contiguous buffers only
    (including rejecting transpose views).  Set
    ``FLAGGEMS_WGRAD_REQUIRE_CONTIGUOUS_MAIN_GRAD=1``.
    """
    return _env_flag_true("FLAGGEMS_WGRAD_REQUIRE_CONTIGUOUS_MAIN_GRAD")


def ensure_contiguous_main_grad(main_grad: torch.Tensor) -> torch.Tensor:
    """Return a contiguous ``main_grad`` for Apex-like wgrad speed.

    Training-side helper: if ``main_grad`` is already contiguous, return it
    unchanged; otherwise return ``main_grad.contiguous()`` (a dense copy with
    the same values).  Callers that own the buffer (e.g. Megatron
    ``weight.main_grad``) should rebind to the returned tensor once and reuse
    it on later steps so each ``wgrad_gemm_accum_*`` call stays on the fast
    contiguous path instead of densify+``copy_`` every step.

    This does not mutate a non-contiguous view in place; rebinding is required
    for the speedup to stick.
    """
    if main_grad.is_contiguous():
        return main_grad
    return main_grad.contiguous()


def _is_transpose_contiguous_2d(t: torch.Tensor) -> bool:
    """True if ``t`` is a 2D transpose view whose ``t.T`` is contiguous."""
    return (
        t.dim() == 2
        and (not t.is_contiguous())
        and t.transpose(0, 1).is_contiguous()
    )


def _is_row_major_padded_2d(t: torch.Tensor) -> bool:
    """True if ``t`` is row-major with padded leading stride.

    Accepts layouts like ``pad[:, :cols]`` where ``stride(-1)==1`` and
    ``stride(0)>=cols``. This is a regular strided 2D layout that can be used
    as ``torch.addmm(..., out=t)`` without a densify+copy roundtrip.
    """
    return (
        t.dim() == 2
        and (not t.is_contiguous())
        and t.stride(1) == 1
        and t.stride(0) >= t.size(1)
    )


def _check_main_grad_contiguity(
    main_grad: torch.Tensor, *, allow_row_major_padded_fastpath: bool = False
) -> None:
    """Raise or warn when ``main_grad`` layout is not strict-contiguous.

    ``REQUIRE_CONTIGUOUS=1`` rejects every non-``is_contiguous()`` tensor
    (including transpose views).  Otherwise only the slow general-NC densify
    path warns; contiguous and transpose-contiguous are silent.
    """
    global _WGRAD_NC_MAIN_WARNED

    if main_grad.is_contiguous():
        return
    if _require_contiguous_main_grad():
        raise RuntimeError(
            "wgrad_gemm_accum: main_grad must be contiguous when "
            "FLAGGEMS_WGRAD_REQUIRE_CONTIGUOUS_MAIN_GRAD=1 "
            "(prefer ensure_contiguous_main_grad / contiguous main_grad "
            "for Apex-like speed)"
        )
    if _is_transpose_contiguous_2d(main_grad):
        return
    if allow_row_major_padded_fastpath and _is_row_major_padded_2d(main_grad):
        return
    if _WGRAD_NC_MAIN_WARNED:
        return
    _WGRAD_NC_MAIN_WARNED = True
    msg = (
        "wgrad_gemm_accum: general non-contiguous main_grad triggers "
        "densify+copy; correct but slower. Transpose views of a contiguous "
        "buffer are already optimized. Prefer "
        "ensure_contiguous_main_grad(main_grad) when binding the buffer, or "
        "set FLAGGEMS_WGRAD_REQUIRE_CONTIGUOUS_MAIN_GRAD=1 to fail hard."
    )
    logger.warning(msg)
    warnings.warn(msg, UserWarning, stacklevel=3)


def _gemmex_fail_limit() -> int:
    """How many consecutive GemmEx runtime failures before permanent fallback."""
    raw = os.environ.get("FLAGGEMS_WGRAD_GEMMEX_FAIL_LIMIT", "").strip()
    if not raw:
        return _DEFAULT_GEMMEX_FAIL_LIMIT
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_GEMMEX_FAIL_LIMIT


def _note_gemmex_success() -> None:
    global _WGRAD_RUNTIME_FAIL_STREAK
    _WGRAD_RUNTIME_FAIL_STREAK = 0


def _activate_torch_fallback(reason: str) -> None:
    """Record permanent fallback; optionally raise; always make the first drop loud."""
    global _WGRAD_EXT_RUNTIME_OK, _WGRAD_FALLBACK_ACTIVE
    global _WGRAD_FALLBACK_REASON, _WGRAD_FALLBACK_WARNED

    _WGRAD_EXT_RUNTIME_OK = False
    _WGRAD_FALLBACK_ACTIVE = True
    if _WGRAD_FALLBACK_REASON is None:
        _WGRAD_FALLBACK_REASON = reason

    if _require_gemmex():
        raise RuntimeError(
            "wgrad_gemm_accum_fp32: GemmEx required "
            f"(FLAGGEMS_WGRAD_REQUIRE_GEMMEX=1) but unavailable: {reason}"
        )

    if _WGRAD_FALLBACK_WARNED:
        return
    _WGRAD_FALLBACK_WARNED = True
    msg = (
        "wgrad_gemm_accum_fp32: GemmEx path unavailable; using Torch fp32 "
        f"matmul fallback. Reason: {reason}. This path is NOT Apex-aligned "
        "for performance/numerics. Set FLAGGEMS_WGRAD_REQUIRE_GEMMEX=1 to "
        "fail hard instead of falling back."
    )
    logger.error(msg)
    warnings.warn(msg, UserWarning, stacklevel=3)


def _handle_gemmex_runtime_failure(exc: BaseException) -> bool:
    """Handle a GemmEx call failure.

    Returns True if permanent fallback was activated (caller should use Torch and
    stay on Torch). Returns False if this was treated as transient: caller should
    use Torch for **this call only**, then retry GemmEx on the next call.
    """
    global _WGRAD_RUNTIME_FAIL_STREAK

    reason = f"GemmEx runtime failed: {exc}"
    if _require_gemmex():
        raise RuntimeError(
            "wgrad_gemm_accum_fp32: GemmEx required "
            f"(FLAGGEMS_WGRAD_REQUIRE_GEMMEX=1) but unavailable: {reason}"
        ) from exc

    _WGRAD_RUNTIME_FAIL_STREAK += 1
    limit = _gemmex_fail_limit()
    if _WGRAD_RUNTIME_FAIL_STREAK >= limit:
        _activate_torch_fallback(
            f"{reason} (permanent after {_WGRAD_RUNTIME_FAIL_STREAK} "
            "consecutive failures)"
        )
        return True

    logger.warning(
        "wgrad_gemm_accum_fp32: %s; using Torch for this call only "
        "(%d/%d consecutive). Will retry GemmEx on the next call.",
        reason,
        _WGRAD_RUNTIME_FAIL_STREAK,
        limit,
    )
    return False


def wgrad_using_torch_fallback() -> bool:
    """True after fp32 path has permanently dropped to Torch matmul (not GemmEx)."""
    return _WGRAD_FALLBACK_ACTIVE


def wgrad_fallback_reason() -> str | None:
    """First reason that activated permanent Torch fallback, or ``None``."""
    return _WGRAD_FALLBACK_REASON


def _collapse_to_2d(input: torch.Tensor, grad_output: torch.Tensor):
    if input.dim() > 2:
        input_2d = input.reshape(-1, input.size(-1))
    else:
        input_2d = input

    if grad_output.dim() > 2:
        grad_output_2d = grad_output.reshape(-1, grad_output.size(-1))
    else:
        grad_output_2d = grad_output

    if input_2d.size(0) != grad_output_2d.size(0):
        raise RuntimeError(
            "input and grad_output must have the same number of rows after collapse"
        )

    return input_2d, grad_output_2d


def _validate_device(*tensors: torch.Tensor) -> None:
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise RuntimeError("All tensors must be on the same device")
    device = devices.pop()
    if device.type != flag_gems.device:
        raise RuntimeError(
            f"Expected tensors on {flag_gems.device}, but got {device.type}"
        )


@lru_cache(None)
def _load_wgrad_gemm_ext():
    """Return a module exposing ``wgrad_gemm_accum_fp32``, or ``None``.

    Prefer the official FlagGems C extension when installed; otherwise try to
    JIT-compile ``flag_gems/csrc/wgrad_gemm_accum.cpp`` (kernel body is shared
    with ``cpp/lib`` via ``wgrad_gemm_accum_kernel.h``).  Returns ``None`` if
    neither path works (e.g. CI runners without nvcc) so callers can fall back
    to a Torch addmm path for correctness tests.
    """
    try:
        from flag_gems import c_operators

        if hasattr(c_operators, "wgrad_gemm_accum_fp32"):
            logger.debug("wgrad_gemm_accum_fp32: using flag_gems.c_operators")
            return c_operators
    except ImportError:
        pass

    try:
        from torch.ops import flag_gems as flag_gems_ops

        if hasattr(flag_gems_ops, "wgrad_gemm_accum_fp32"):
            logger.debug("wgrad_gemm_accum_fp32: using torch.ops.flag_gems")
            return flag_gems_ops
    except (ImportError, AttributeError):
        pass

    from torch.utils.cpp_extension import load

    src = Path(__file__).resolve().parent.parent / "csrc" / "wgrad_gemm_accum.cpp"
    if not src.is_file():
        logger.debug("wgrad_gemm_accum_fp32: missing GemmEx source at %s", src)
        return None

    # Do NOT link -lcublas here. The extension must call the same libcublas that
    # owns PyTorch's BLAS handle (via dlsym); a second copy -> INVALID_VALUE.
    logger.info(
        "wgrad_gemm_accum_fp32: JIT-compiling CUDA extension from %s (first call only)",
        src,
    )
    try:
        return load(
            name="flag_gems_wgrad_gemm_accum",
            sources=[str(src)],
            extra_ldflags=["-ldl"],
            with_cuda=True,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - caller decides fallback vs hard-fail
        logger.debug("wgrad_gemm_accum_fp32: JIT extension unavailable (%s)", exc)
        return None


def _torch_wgrad_gemm_accum_fp32(
    input_2d: torch.Tensor,
    grad_output_2d: torch.Tensor,
    main_grad: torch.Tensor,
) -> None:
    """Torch fallback: ``main_grad += grad_output.T @ input`` in fp32 math."""
    input_c = input_2d.contiguous().float()
    grad_c = grad_output_2d.contiguous().float()
    if main_grad.is_contiguous():
        main_grad.add_(grad_c.t() @ input_c)
        return
    # Transpose-contiguous: write W.T += X.T @ G directly (no densify+copy).
    if _is_transpose_contiguous_2d(main_grad):
        main_t = main_grad.transpose(0, 1)
        main_t.add_(input_c.t() @ grad_c)
        return
    weight = main_grad.contiguous()
    weight.add_(grad_c.t() @ input_c)
    main_grad.copy_(weight)


def _wgrad_fp32_strict_accum(
    grad_output_2d: torch.Tensor,
    input_2d: torch.Tensor,
    main_grad: torch.Tensor,
) -> None:
    """fp32 activations with TF32 off: match CPU fp64 reference semantics.

    GemmEx + DEFAULT_TENSOR_OP may still use TF32 tensor cores on Ampere+ even
    when ``allow_tf32=False``.  Mirror ``_ref_wgrad_gemm_accum_fp32_cpu``:
    fp64 GEMM -> cast to fp32 -> add into main_grad.
    """
    input_c = input_2d.contiguous()
    grad_c = grad_output_2d.contiguous()
    if main_grad.is_contiguous():
        wgrad_fp32 = (grad_c.t().contiguous().double() @ input_c.double()).float()
        main_grad.add_(wgrad_fp32)
        return
    if _is_transpose_contiguous_2d(main_grad):
        main_t = main_grad.transpose(0, 1)
        wgrad_t = (input_c.double().t().contiguous() @ grad_c.double()).float()
        main_t.add_(wgrad_t)
        return
    wgrad_fp32 = (grad_c.t().contiguous().double() @ input_c.double()).float()
    weight = main_grad.contiguous()
    weight.add_(wgrad_fp32)
    main_grad.copy_(weight)


def _cublas_wgrad_gemm_accum_fp32(
    input_2d: torch.Tensor,
    grad_output_2d: torch.Tensor,
    main_grad: torch.Tensor,
    *,
    strict_cpu_ref: bool = False,
) -> None:
    """Apex ``wgrad_gemm_accum_fp32_cuda`` layout via compiled ``cublasGemmEx``."""
    if main_grad.dtype != torch.float32:
        raise RuntimeError("main_grad must be float32 for GemmEx fp32-accum path")
    if input_2d.dtype != grad_output_2d.dtype:
        raise RuntimeError(
            "input and grad_output dtype must match, "
            f"got {input_2d.dtype} vs {grad_output_2d.dtype}"
        )

    # Test-only: fp64 GEMM to match CPU reference when TF32 is off.
    # Default (bench / training): GemmEx like Apex regardless of allow_tf32 default.
    if strict_cpu_ref and input_2d.dtype == torch.float32:
        _wgrad_fp32_strict_accum(grad_output_2d, input_2d, main_grad)
        return

    # Prefer GemmEx. Missing extension -> permanent loud fallback.
    # Runtime failure -> Torch for this call only until consecutive-fail limit.
    if _WGRAD_EXT_RUNTIME_OK:
        ext = _load_wgrad_gemm_ext()
        if ext is not None:
            try:
                ext.wgrad_gemm_accum_fp32(input_2d, grad_output_2d, main_grad)
                _note_gemmex_success()
                return
            except Exception as exc:  # noqa: BLE001
                _handle_gemmex_runtime_failure(exc)
                _torch_wgrad_gemm_accum_fp32(input_2d, grad_output_2d, main_grad)
                return
        _activate_torch_fallback(
            "compiled GemmEx extension unavailable (c_operators / JIT)"
        )

    _torch_wgrad_gemm_accum_fp32(input_2d, grad_output_2d, main_grad)


def wgrad_gemmex_available() -> bool:
    """True if the compiled GemmEx path loads and runs a tiny smoke call."""
    if not torch.cuda.is_available():
        return False
    if _WGRAD_FALLBACK_ACTIVE:
        return False
    ext = _load_wgrad_gemm_ext()
    if ext is None:
        return False
    try:
        device = flag_gems.device
        inp = torch.randn(4, 8, device=device, dtype=torch.float16)
        gout = torch.randn(4, 16, device=device, dtype=torch.float16)
        main = torch.zeros(16, 8, device=device, dtype=torch.float32)
        ext.wgrad_gemm_accum_fp32(inp, gout, main)
        return True
    except Exception:  # noqa: BLE001
        return False


def _matmul_operands(
    grad_output_2d: torch.Tensor, input_2d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(grad_output.T, input)`` for same-dtype ``torch.addmm``.

    Densify first, then take a transpose view so contiguous / non-contiguous
    callers share one cuBLAS ``OP_T`` path.
    """
    if not grad_output_2d.is_contiguous():
        grad_output_2d = grad_output_2d.contiguous()
    if not input_2d.is_contiguous():
        input_2d = input_2d.contiguous()
    return grad_output_2d.t(), input_2d


def _fused_addmm_cublas(
    main_grad: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> None:
    """Same-dtype fused ``main_grad += mat1 @ mat2`` via PyTorch cuBLAS addmm.

    ``mat1``/``mat2`` are ``(grad_output.T, input)``.  For transpose-contiguous
    ``main_grad``, rewrite as ``main_grad.T += input.T @ grad_output``.
    """
    if main_grad.is_contiguous():
        torch.addmm(main_grad, mat1, mat2, beta=1, alpha=1, out=main_grad)
        return
    if _is_transpose_contiguous_2d(main_grad):
        # mat1 = grad.T, mat2 = input  =>  W += grad.T @ input
        # W.T += input.T @ grad  with grad = mat1.T, input = mat2
        main_t = main_grad.transpose(0, 1)
        torch.addmm(main_t, mat2.t(), mat1.t(), beta=1, alpha=1, out=main_t)
        return
    if _is_row_major_padded_2d(main_grad):
        # PoC fast path for regular padded row-major NC (e.g. pad[:, :in]):
        # write directly into strided out, skip densify+copy.
        torch.addmm(main_grad, mat1, mat2, beta=1, alpha=1, out=main_grad)
        return
    # General non-contiguous out: densify, compute, copy back.
    weight = main_grad.contiguous()
    torch.addmm(weight, mat1, mat2, beta=1, alpha=1, out=weight)
    main_grad.copy_(weight)


def _accum_wgrad(
    grad_output_2d: torch.Tensor,
    input_2d: torch.Tensor,
    main_grad: torch.Tensor,
    *,
    fp32_accum: bool,
    strict_cpu_ref: bool = False,
) -> None:
    # K==0: empty product, leave main_grad unchanged (same as a no-op GEMM).
    if input_2d.size(0) == 0:
        return
    # Zero M/N (out/in features): reject like Apex/cublasGemmEx.
    if input_2d.size(1) == 0 or grad_output_2d.size(1) == 0:
        raise RuntimeError(
            "wgrad_gemm_accum: in_features and out_features must be > 0 "
            f"(got in={input_2d.size(1)}, out={grad_output_2d.size(1)}); "
            "Apex/cublasGemmEx also reject zero M/N"
        )

    _check_main_grad_contiguity(
        main_grad, allow_row_major_padded_fastpath=(not fp32_accum)
    )

    if fp32_accum:
        # Match Apex fused_weight_gradient path (half/bf16/fp32 -> fp32 C).
        _cublas_wgrad_gemm_accum_fp32(
            input_2d, grad_output_2d, main_grad, strict_cpu_ref=strict_cpu_ref
        )
        return

    grad_output_T, input_c = _matmul_operands(grad_output_2d, input_2d)
    _fused_addmm_cublas(main_grad, grad_output_T, input_c)


def wgrad_gemm_accum_fp32(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    main_grad: torch.Tensor,
) -> None:
    """Accumulate weight gradient into ``main_grad`` using fp32 storage.

    Production / training path mirrors Apex: ``cublasGemmEx`` with
    ``CUBLAS_GEMM_DEFAULT_TENSOR_OP``.  Setting
    ``torch.backends.cuda.matmul.allow_tf32 = False`` does **not** force a
    full-fp32 math path here (same footgun as Apex on Ampere+).

    Non-contiguous ``main_grad``: a transpose view of a contiguous buffer is
    handled without densify+``copy_`` (write through ``main_grad.T``).  General
    NC still densifies; prefer ``ensure_contiguous_main_grad`` when binding, or
    set ``FLAGGEMS_WGRAD_REQUIRE_CONTIGUOUS_MAIN_GRAD=1`` to fail hard.

    For test-only CPU-fp64-matched checks under TF32-off, call
    ``wgrad_gemm_accum_fp32_strict_cpu_ref`` instead — do not use that helper
    in training or benchmarks (slower and not Apex-aligned).
    """
    logger.debug("GEMS WGRAD_GEMM_ACCUM_FP32")

    _validate_device(input, grad_output, main_grad)

    if main_grad.dtype != torch.float32:
        raise RuntimeError(
            "main_grad must be float32 for wgrad_gemm_accum_fp32, "
            f"but got {main_grad.dtype}"
        )
    if input.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise RuntimeError(
            "Unsupported input dtype for wgrad_gemm_accum_fp32: " f"{input.dtype}"
        )
    if grad_output.dtype != input.dtype:
        raise RuntimeError(
            "grad_output dtype must match input dtype, "
            f"but got {grad_output.dtype} vs {input.dtype}"
        )

    input_2d, grad_output_2d = _collapse_to_2d(input, grad_output)
    out_dim = grad_output_2d.size(-1)
    in_dim = input_2d.size(-1)
    if main_grad.shape != (out_dim, in_dim):
        raise RuntimeError(
            "main_grad shape mismatch: expected "
            f"({out_dim}, {in_dim}), got {tuple(main_grad.shape)}"
        )

    _accum_wgrad(
        grad_output_2d,
        input_2d,
        main_grad,
        fp32_accum=True,
        strict_cpu_ref=False,
    )


def wgrad_gemm_accum_fp32_strict_cpu_ref(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    main_grad: torch.Tensor,
) -> None:
    """TEST-ONLY: fp32 activations via fp64 GEMM then fp32 add into ``main_grad``.

    Use this for TF32-off math checks against a CPU fp64 reference.  It is
    intentionally **not** on the public training API so production code cannot
    accidentally pass ``strict_cpu_ref=True`` and silently lose Apex alignment
    / speed.  Only ``torch.float32`` activations are supported.
    """
    logger.debug("GEMS WGRAD_GEMM_ACCUM_FP32_STRICT_CPU_REF (test-only)")

    _validate_device(input, grad_output, main_grad)

    if main_grad.dtype != torch.float32:
        raise RuntimeError(
            "main_grad must be float32 for wgrad_gemm_accum_fp32_strict_cpu_ref, "
            f"but got {main_grad.dtype}"
        )
    if input.dtype != torch.float32 or grad_output.dtype != torch.float32:
        raise RuntimeError(
            "wgrad_gemm_accum_fp32_strict_cpu_ref requires float32 activations, "
            f"got input={input.dtype}, grad_output={grad_output.dtype}"
        )

    input_2d, grad_output_2d = _collapse_to_2d(input, grad_output)
    out_dim = grad_output_2d.size(-1)
    in_dim = input_2d.size(-1)
    if main_grad.shape != (out_dim, in_dim):
        raise RuntimeError(
            "main_grad shape mismatch: expected "
            f"({out_dim}, {in_dim}), got {tuple(main_grad.shape)}"
        )

    _accum_wgrad(
        grad_output_2d,
        input_2d,
        main_grad,
        fp32_accum=True,
        strict_cpu_ref=True,
    )


def wgrad_gemm_accum_fp16(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    main_grad: torch.Tensor,
) -> None:
    """Accumulate weight gradient into ``main_grad`` using fp16/bf16 storage.

    Non-contiguous ``main_grad``: transpose-contiguous is a fast path. Regular
    row-major padded 2D layouts (``stride(-1)==1``) also have a direct out path.
    Other general NC layouts densify then ``copy_`` (same caveat as fp32 path).
    """
    logger.debug("GEMS WGRAD_GEMM_ACCUM_FP16")

    _validate_device(input, grad_output, main_grad)

    if main_grad.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "main_grad must be float16 or bfloat16 for wgrad_gemm_accum_fp16, "
            f"but got {main_grad.dtype}"
        )
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "Unsupported input dtype for wgrad_gemm_accum_fp16: " f"{input.dtype}"
        )
    if grad_output.dtype != input.dtype:
        raise RuntimeError(
            "grad_output dtype must match input dtype, "
            f"but got {grad_output.dtype} vs {input.dtype}"
        )
    if main_grad.dtype != input.dtype:
        raise RuntimeError(
            "main_grad dtype must match input dtype, "
            f"but got {main_grad.dtype} vs {input.dtype}"
        )

    input_2d, grad_output_2d = _collapse_to_2d(input, grad_output)
    out_dim = grad_output_2d.size(-1)
    in_dim = input_2d.size(-1)
    if main_grad.shape != (out_dim, in_dim):
        raise RuntimeError(
            "main_grad shape mismatch: expected "
            f"({out_dim}, {in_dim}), got {tuple(main_grad.shape)}"
        )

    _accum_wgrad(
        grad_output_2d,
        input_2d,
        main_grad,
        fp32_accum=False,
    )
