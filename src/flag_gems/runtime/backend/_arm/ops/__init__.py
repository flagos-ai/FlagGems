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

from .addmm import addmm, addmm_out
from .all import all
from .any import any
from .argmax import argmax
from .attention import scaled_dot_product_attention
from .bmm import bmm
from .cumsum import cumsum
from .div import (  # noqa: F401
    div_mode,
    div_mode_,
    floor_divide,
    floor_divide_,
    remainder,
    remainder_,
    true_divide,
    true_divide_,
)
from .exponential_ import exponential_
from .full import full
from .gather import gather
from .index_select import index_select
from .isin import isin
from .lt import lt
from .masked_fill import masked_fill
from .min import min
from .mm import mm, mm_out
from .multinomial import multinomial
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .quantile import quantile
from .scatter import scatter
from .sort import sort
from .sub import sub
from .topk import topk
from .where import where_self_out

__all__ = [
    "addmm",
    "addmm_out",
    "all",
    "any",
    "argmax",
    "bmm",
    "cumsum",
    "div_mode",
    "div_mode_",
    "exponential_",
    "floor_divide",
    "floor_divide_",
    "full",
    "gather",
    "index_select",
    "isin",
    "lt",
    "masked_fill",
    "min",
    "mm",
    "mm_out",
    "multinomial",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "quantile",
    "remainder",
    "remainder_",
    "scaled_dot_product_attention",
    "scatter",
    "sort",
    "sub",
    "topk",
    "where_self_out",
    "apply_arm_overrides",
]

import logging as _logging  # noqa: E402

import torch as _torch  # noqa: E402

# ---------------------------------------------------------------------------
# ARM CPU op overrides (torch.library impls + flag_gems / F.* monkeypatches).
#
# These are NOT standard aten_lib ops registered through the FlagGems Registrar,
# so they cannot ride enable()/only_enable()'s _FULL_CONFIG include/exclude. They
# are collected in a name-keyed registry and applied through a single idempotent
# entry point, apply_arm_overrides(include=, exclude=), so callers can select a
# subset (e.g. exclude {"mm"} to avoid its prefill regression on prefill-heavy
# workloads).
# ---------------------------------------------------------------------------

# torch.library handles must stay alive — GC would revoke the registration.
_argmax_aten_lib = None
_mm_aten_lib = None


def _register_quantized_linear_dynamic():
    # Triton-CPU INT8 GEMM for quantized::linear_dynamic (quantized:: namespace).
    from .quantized_linear_dynamic import register as _reg

    _reg()


def _register_int_mm():
    # Triton-CPU INT8 GEMM for aten::_int_mm (enables torchao INT8 paths).
    from .int_mm import register as _reg

    _reg()


def _register_argmax():
    # FlagGems argmax for aten::argmax (decode lm_head: 2.2x faster for [1,151936]).
    global _argmax_aten_lib
    if _argmax_aten_lib is not None:
        return
    from .argmax import argmax as _fg_argmax

    _argmax_aten_lib = _torch.library.Library("aten", "IMPL")
    _argmax_aten_lib.impl("argmax", _fg_argmax, "CPU", allow_override=True)
    _logging.getLogger(__name__).debug(
        "FlagGems ARM: registered Triton-CPU argmax for aten::argmax"
    )


def _override_fused_add_rms_norm_with_arm():
    # Standalone rms_norm override dropped: no measurable benefit vs ATen native
    # (Qwen3-1.7B INT8 decode A/B 3 rounds: 9.93 vs 9.97 tok/s, within noise).
    # fused_add_rms_norm kept: saves a residual-add memory roundtrip (vLLM path).
    import flag_gems as _fg

    from .rms_norm import fused_add_rms_norm as _arm_fused_add_rms_norm

    _fg.fused_add_rms_norm = _arm_fused_add_rms_norm
    _logging.getLogger(__name__).debug(
        "FlagGems ARM: overrode flag_gems.fused_add_rms_norm with ARM Triton kernel"
    )


def _override_rope_with_arm():
    # Generic fused/rotary_embedding.py uses @libentry() → DEVICE_COUNT crash on CPU.
    import flag_gems as _fg
    import flag_gems.fused as _fg_fused

    from .rope import arm_apply_rotary_pos_emb as _arm_rope

    _fg.apply_rotary_pos_emb = _arm_rope
    _fg_fused.apply_rotary_pos_emb = _arm_rope
    _logging.getLogger(__name__).debug(
        "FlagGems ARM: overrode flag_gems.apply_rotary_pos_emb with pure-PyTorch"
    )


def _override_silu_and_mul_with_arm():
    # Generic fused/silu_and_mul.py uses @pointwise_dynamic → @libentry() → crash.
    import flag_gems as _fg
    import flag_gems.fused as _fg_fused

    from .silu_and_mul import arm_silu_and_mul as _arm_sam
    from .silu_and_mul import arm_silu_and_mul_out as _arm_sam_out

    _fg.silu_and_mul = _arm_sam
    _fg.silu_and_mul_out = _arm_sam_out
    _fg_fused.silu_and_mul = _arm_sam
    _fg_fused.silu_and_mul_out = _arm_sam_out
    _logging.getLogger(__name__).debug(
        "FlagGems ARM: overrode flag_gems.silu_and_mul / silu_and_mul_out"
    )


def _register_mm():
    # aten::mm BF16 override. M=1 decode: 2-5x faster than ATen (unoptimized GEMV).
    # M=64 prefill: 2-3x slower (ATen uses native BF16 BFMMLA) — exclude "mm" for
    # prefill-heavy workloads. _mm_aten_lib must stay alive.
    global _mm_aten_lib
    if _mm_aten_lib is not None:
        return
    from .mm import mm as _fg_mm

    _mm_aten_lib = _torch.library.Library("aten", "IMPL")
    _mm_aten_lib.impl("mm", _fg_mm, "CPU", allow_override=True)
    _logging.getLogger(__name__).debug(
        "FlagGems ARM: registered Triton-CPU mm for aten::mm"
    )


def _register_sdpa():
    # Triton-CPU Flash Attention for F.scaled_dot_product_attention (prefill 4-5x;
    # decode/other cases fall back to ATen).
    #
    # Intentionally a monkey-patch, NOT torch.library: a Library("aten","IMPL")
    # override would route the ATen fallback inside our own wrapper back to us
    # (infinite recursion). The monkey-patch leaves the original C++ dispatch
    # reachable via _aten_sdpa captured at import in attention.py.
    import torch.nn.functional as _F

    from .attention import scaled_dot_product_attention as _fg_sdpa

    _F.scaled_dot_product_attention = _fg_sdpa
    _logging.getLogger(__name__).debug(
        "FlagGems ARM: monkey-patched F.scaled_dot_product_attention "
        "with Triton Flash Attention (prefill 4-5x speedup)"
    )


# Name → applier. Names match the aten/flag_gems op they override so callers can
# select with the same vocabulary as only_enable(include=[...]).
_ARM_OVERRIDE_REGISTRY = {
    "quantized_linear_dynamic": _register_quantized_linear_dynamic,
    "_int_mm": _register_int_mm,
    "argmax": _register_argmax,
    "fused_add_rms_norm": _override_fused_add_rms_norm_with_arm,
    "apply_rotary_pos_emb": _override_rope_with_arm,
    "silu_and_mul": _override_silu_and_mul_with_arm,
    "mm": _register_mm,
    "scaled_dot_product_attention": _register_sdpa,
}

_ARM_OVERRIDES_APPLIED = set()


def apply_arm_overrides(include=None, exclude=None):
    """Apply ARM CPU op overrides, optionally restricted to a subset.

    Args:
        include: iterable of override names to apply (None = all known).
        exclude: iterable of override names to skip (applied after include).

    Idempotent: each override is applied at most once per process. Names are the
    keys of _ARM_OVERRIDE_REGISTRY (the aten/flag_gems op each one overrides).
    """
    names = (
        set(_ARM_OVERRIDE_REGISTRY)
        if include is None
        else set(include) & set(_ARM_OVERRIDE_REGISTRY)
    )
    if exclude:
        names -= set(exclude)
    for name in names:
        if name in _ARM_OVERRIDES_APPLIED:
            continue
        try:
            _ARM_OVERRIDE_REGISTRY[name]()
            _ARM_OVERRIDES_APPLIED.add(name)
        except Exception as e:  # noqa: BLE001
            _logging.getLogger(__name__).warning(
                f"FlagGems ARM: failed to apply override '{name}': {e}"
            )


# NOTE: overrides are NOT applied on import. The caller selects which ones to
# engage via apply_arm_overrides(include=[...]) — mirroring FlagGems'
# only_enable() opt-in model. This avoids silently monkeypatching aten::mm /
# F.scaled_dot_product_attention / flag_gems.* process-wide just by importing,
# and lets a workload pick the net-positive subset (e.g. exclude "mm" on
# prefill-heavy runs where the decode-tuned mm regresses prefill).
#
#   from flag_gems.runtime.backend._arm.ops import apply_arm_overrides
#   apply_arm_overrides()                       # engage all curated overrides
#   apply_arm_overrides(include=["mm", "argmax"])  # only these
#   apply_arm_overrides(exclude=["mm"])            # all but mm
