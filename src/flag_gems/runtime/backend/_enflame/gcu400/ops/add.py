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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# ---------------------------------------------------------------------------
# torch.randn complex-on-gcu shim (torch_gcu P21 defect workaround)
#
# torch_gcu's topsatenNormal aborts the whole process for complex dtypes
# (TOPSATEN_STATUS_BAD_PARAM -> C++ terminate), which kills any test whose
# input construction calls torch.randn(dtype=complex, device=gcu) before
# flag_gems code even runs.  Complex tensors themselves are fully usable on
# gcu400 once created: CPU->GCU copy, native elementwise add, and
# view_as_real/view_as_complex all work (verified on L600).  So route
# device-side complex randn through CPU generation + H2D copy; every other
# call shape (non-complex dtype, cpu/unset device, explicit generator)
# passes through to the original implementation untouched.  Narrow,
# value-preserving, and safe to drop once topsatenNormal supports
# ComplexFloat.
# ---------------------------------------------------------------------------
_TORCH_COMPLEX_DTYPES = (torch.complex32, torch.complex64, torch.complex128)

# fp8 dtypes for the isclose shim below (getattr-guarded across torch
# versions; e4m3fnuz/e5m2fnuz are byte-swapped twins of the two base kinds).
_FP8_COMPARE_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
        getattr(torch, "float8_e8m0fnu", None),
    )
    if dtype is not None
)


def _install_complex_randn_shim():
    current = torch.randn
    if getattr(current, "_flaggems_gcu_complex_safe", False):
        return

    def _complex_safe_randn(*args, **kwargs):
        dtype = kwargs.get("dtype")
        if dtype not in _TORCH_COMPLEX_DTYPES:
            return current(*args, **kwargs)
        device = kwargs.get("device")
        if device is None or kwargs.get("generator") is not None:
            # Default device (cpu) is complex-safe; a caller-supplied
            # generator keeps original semantics.
            return current(*args, **kwargs)
        try:
            target = torch.device(device)
        except (TypeError, ValueError):
            return current(*args, **kwargs)
        if target.type != "gcu":
            return current(*args, **kwargs)
        kw = dict(kwargs)
        kw["device"] = "cpu"
        out = kwargs.get("out")
        if out is not None:
            kw.pop("out")
            out.copy_(current(*args, **kw))
            return out
        return current(*args, **kw).to(target)

    _complex_safe_randn._flaggems_gcu_complex_safe = True
    torch.randn = _complex_safe_randn


def _install_complex_isclose_shim():
    # Same defect family on the comparison side: torch.isclose on complex
    # gcu tensors aborts the process (complex sub/abs are unimplemented in
    # topsaten, P21), which kills torch.testing.assert_close for complex
    # results even when the op itself is correct.  Route complex device
    # comparisons through CPU (value-identical, returned on the original
    # device); every other call passes through untouched.  Note: only the
    # functional form is shimmed (torch.testing uses exactly that); the
    # Tensor.isclose method keeps original behavior.
    #
    # fp8 operands take the same CPU route *with an fp32 upcast* (P21
    # flaggems__conj_physical.md sub-item 5): topsatenEq has no fp8 dtype
    # either, so the on-device compare aborts the whole pytest process, and
    # CPU isclose itself raises NotImplementedError for fp8
    # ("mul_cpu_reduced_float").  The upcast is exact for the tolerance-free
    # comparisons torch.testing performs on 1-byte floats (isclose(0,0) == eq
    # and fp8 -> fp32 is injective); no in-repo caller passes nonzero
    # tolerances with fp8 operands.
    current = torch.isclose
    if getattr(current, "_flaggems_gcu_complex_safe", False):
        return

    def _complex_safe_isclose(input, other, *args, **kwargs):
        if isinstance(input, torch.Tensor) and input.device.type == "gcu":
            if input.is_complex():
                target = input.device
                other_cpu = other.cpu() if isinstance(other, torch.Tensor) else other
                return current(input.cpu(), other_cpu, *args, **kwargs).to(target)
            if input.dtype in _FP8_COMPARE_DTYPES:
                target = input.device
                input_fp32 = input.cpu().to(torch.float32)
                other_fp32 = (
                    other.cpu().to(torch.float32)
                    if isinstance(other, torch.Tensor)
                    else other
                )
                return current(input_fp32, other_fp32, *args, **kwargs).to(target)
        return current(input, other, *args, **kwargs)

    _complex_safe_isclose._flaggems_gcu_complex_safe = True
    torch.isclose = _complex_safe_isclose


_install_complex_randn_shim()
_install_complex_isclose_shim()

# GCU400 override of the vendored pointwise_dynamic path for the hot
# same-shape contiguous float case.  The pointwise_dynamic codegen emits an
# ND-tile block-pointer kernel (tile 65536, grid<=48, num_warps=4) whose
# block-pointer lowering plus 2D/3D tile quantization leaves 13% on the table
# at the 16.7M core shape (48.7 vs 41.8us native) and 3% at 1Gi.  The flat
# grid-stride kernel below treats the tensors as one dense 1D range (both
# operands are contiguous with identical shape) and keeps the exact
# `x + y * alpha` body of the generated kernel, so numerics are bit-identical.
#
# Config ladder measured on L600 with the core benchmark protocol
# (do_bench kernel median, warmup 1000ms / rep 100ms), 2-load flat kernel:
#   - the sweet spot keeps 256KB of load tiles per program (2 x BLOCK x
#     itemsize), i.e. BLOCK = 131072 // itemsize, inside the 448KB
#     double-program DSM budget so grid=48 runs 48-way: fp16/bf16 B65536/nw2
#     hits 2.96TB/s at 1Gi (2174.9us, 1.045x vs native) and 41.0us at 16.7M
#     (1.019x); fp32 B32768/nw2 hits 2.95TB/s at 1Gi (4363.4us, 1.070x) and
#     74.4us at 16.7M (1.038x).  num_warps=2 beats 4 at the big-block tier
#     (B65536-nw4 2430us) while B8192 wants nw4; num_stages>=2 is neutral to
#     harmful (1Gi fp16 2174.9 -> 2265.6us).
#   - tiny shapes (<=4096) are launch-floor bound (~3.3us); one CTA with the
#     exact next_pow2 tile matches the floor, and the sub-2M band keeps
#     B8192/nw4 for CTA count (add3/special_exp2 ladder precedent).
#
# Everything the fast guard rejects (broadcast shapes, dtype promotion,
# non-contiguous layouts, non-float dtypes, 0-dim complex inputs) keeps the
# existing pointwise_dynamic arms unchanged.


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def add_flat_kernel(
    x_ptr, y_ptr, out_ptr, alpha, N_total, BLOCK: tl.constexpr, EXACT: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EXACT:
            x = tl.load(x_ptr + off)
            y = tl.load(y_ptr + off)
            tl.store(out_ptr + off, x + y * alpha)
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask)
            y = tl.load(y_ptr + off, mask=mask)
            tl.store(out_ptr + off, x + y * alpha, mask=mask)


def _choose_block_and_warps(N_total, itemsize):
    if N_total <= 4096:
        return max(triton.next_power_of_2(N_total), 256), 2
    if N_total <= 65536:
        return 2048, 4
    if N_total <= (1 << 21):
        return 8192, 4
    return 131072 // itemsize, 2


def _add_flat_applicable(A, B):
    return (
        A.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and B.dtype == A.dtype
        and B.device == A.device
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    )


def _launch_add_flat(A, B, out, alpha):
    N_total = A.numel()
    if out is None:
        out = torch.empty_like(A)
    if N_total == 0:
        return out
    BLOCK, num_warps = _choose_block_and_warps(N_total, A.element_size())
    grid_size = min(-(-N_total // BLOCK), NUM_SIPS * 2)
    exact = N_total % BLOCK == 0
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic; converterResult=-2 family, see P02); retrying the
    # launch recompiles and succeeds (sinh.py / atan2_.py precedent).
    for attempt in range(3):
        try:
            with torch_device_fn.device(A.device):
                add_flat_kernel[(grid_size,)](
                    A,
                    B,
                    out,
                    alpha,
                    N_total,
                    BLOCK=BLOCK,
                    EXACT=exact,
                    num_warps=num_warps,
                    num_stages=1,
                )
            return out
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "add flat kernel launch failed (attempt %d): %s", attempt + 1, exc
            )


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_tensor_scalar(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_scalar_tensor(x, y, alpha):
    return x + y * alpha


# ---------------------------------------------------------------------------
# Complex support (parity with the generic add, adapted for torch_gcu).
#
# torch_gcu's topsatenCopy aborts on on-device complex dtype casts
# (float/int -> complex, P21), so the generic decomposition
# (`B.to(A.dtype)` before view_as_real) cannot be reused verbatim.  Instead
# everything is computed in the float "pair" domain: complex operands go
# through view_as_real (a metadata view), real tensor operands are stacked
# into (real, 0) pairs, and scalars become a broadcast [re, im] pair.  The
# pair-domain add reuses the flat fast path when applicable, and the result
# is reassembled with view_as_complex (also metadata-only).  alpha must be
# real, which covers every aten add.Tensor complex use in the test suite.
# ---------------------------------------------------------------------------
_COMPLEX_FLOAT_VIEW = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


def _real_to_pair(t):
    return torch.stack((t, torch.zeros_like(t)), dim=-1)


def _pair_add(Ar, Br, alpha):
    if _add_flat_applicable(Ar, Br):
        return _launch_add_flat(Ar, Br, None, alpha)
    return add_func(Ar, Br, alpha)


def _add_complex(A, B, alpha):
    if isinstance(alpha, complex):
        raise NotImplementedError(
            "add with complex alpha is not supported on the enflame backend"
        )
    if (
        isinstance(A, torch.Tensor)
        and isinstance(B, torch.Tensor)
        and B.device != A.device
    ):
        B = B.to(A.device)
    result_dtype = torch.result_type(A, B)
    common = _COMPLEX_FLOAT_VIEW.get(result_dtype)
    if common is None:
        raise RuntimeError(
            f"unexpected non-complex result dtype {result_dtype} in complex add"
        )
    if common == torch.float64:
        raise NotImplementedError(
            "complex128 add is not supported (fp64 disabled on this backend)"
        )

    def _pair_of(x, other):
        if isinstance(x, torch.Tensor):
            pair = torch.view_as_real(x) if x.is_complex() else _real_to_pair(x)
            return pair.to(common) if pair.dtype != common else pair
        if isinstance(x, complex):
            values = (x.real, x.imag)
        else:
            values = (float(x), 0.0)
        return torch.tensor(values, dtype=common, device=other.device).broadcast_to(
            other.shape + (2,)
        )

    Ar = _pair_of(A, B)
    Br = _pair_of(B, A)
    return torch.view_as_complex(_pair_add(Ar, Br, alpha))


def add(A, B, *, alpha=1):
    logger.debug("GEMS_ENFLAME ADD")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.is_complex() or B.is_complex():
            return _add_complex(A, B, alpha)
        if _add_flat_applicable(A, B):
            return _launch_add_flat(A, B, None, alpha)
        return add_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        if A.is_complex() or isinstance(B, complex):
            return _add_complex(A, B, alpha)
        return add_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        if B.is_complex() or isinstance(A, complex):
            return _add_complex(A, B, alpha)
        return add_func_scalar_tensor(A, B, alpha)
    else:
        return torch.tensor(A + B * alpha)


def add_(A, B, *, alpha=1):
    logger.debug("GEMS_ENFLAME ADD_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.is_complex() or B.is_complex():
            A.copy_(_add_complex(A, B, alpha))
            return A
        if _add_flat_applicable(A, B):
            _launch_add_flat(A, B, A, alpha)
            return A
        return add_func(A, B, alpha, out0=A)
    elif isinstance(A, torch.Tensor):
        if A.is_complex() or isinstance(B, complex):
            A.copy_(_add_complex(A, B, alpha))
            return A
        return add_func_tensor_scalar(A, B, alpha, out0=A)
    elif isinstance(B, torch.Tensor):
        if B.is_complex() or isinstance(A, complex):
            return _add_complex(A, B, alpha)
        return add_func_scalar_tensor(A, B, alpha, out0=A)
    else:
        raise ValueError("Unreachable.")
