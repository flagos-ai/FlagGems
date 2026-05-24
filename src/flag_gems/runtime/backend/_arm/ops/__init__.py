from .addmm import addmm, addmm_out
from .all import all
from .any import any
from .argmax import argmax
from .attention import scaled_dot_product_attention
from .bmm import bmm
from .cumsum import cumsum
from .div import (
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
]

# Register Triton-CPU INT8 GEMM for quantized::linear_dynamic (quantized:: namespace,
# not aten::, so handled separately from the main FlagGems aten_lib registrations).
from .quantized_linear_dynamic import register as _register_quantized_linear_dynamic

_register_quantized_linear_dynamic()

# Register Triton-CPU INT8 GEMM for aten::_int_mm (enables torchao INT8 paths).
from .int_mm import register as _register_int_mm

_register_int_mm()

# Register FlagGems argmax for aten::argmax (decode lm_head: 2.2x faster for [1,151936]).
# Auto-registered on import so INT8 users get the speedup without explicit only_enable().
import logging as _logging
import torch as _torch
from .argmax import argmax as _fg_argmax

_argmax_aten_lib = None


def _register_argmax():
    global _argmax_aten_lib
    if _argmax_aten_lib is not None:
        return
    try:
        _argmax_aten_lib = _torch.library.Library("aten", "IMPL")
        _argmax_aten_lib.impl("argmax", _fg_argmax, "CPU", allow_override=True)
        _logging.getLogger(__name__).debug(
            "FlagGems ARM: registered Triton-CPU argmax for aten::argmax"
        )
    except Exception as e:
        _logging.getLogger(__name__).warning(
            f"FlagGems ARM: failed to register argmax override: {e}"
        )


_register_argmax()

# Override flag_gems.fused_add_rms_norm with ARM CPU version.
# Standalone rms_norm override removed: no measurable benefit vs ATen native
# on Qwen3-1.7B INT8 decode (A/B 3 rounds: 9.93 vs 9.97 tok/s, within noise).
# fused_add_rms_norm kept: saves a residual-add memory roundtrip (not yet
# benchmarked separately but used by vLLM residual path).
def _override_fused_add_rms_norm_with_arm():
    try:
        import flag_gems as _fg
        from .rms_norm import fused_add_rms_norm as _arm_fused_add_rms_norm
        _fg.fused_add_rms_norm = _arm_fused_add_rms_norm
        _logging.getLogger(__name__).debug(
            "FlagGems ARM: overrode flag_gems.fused_add_rms_norm with ARM Triton kernel"
        )
    except Exception as e:
        _logging.getLogger(__name__).warning(
            f"FlagGems ARM: failed to override fused_add_rms_norm: {e}"
        )


_override_fused_add_rms_norm_with_arm()


# Override flag_gems.apply_rotary_pos_emb with ARM pure-PyTorch version.
# The generic fused/rotary_embedding.py uses @libentry() → DEVICE_COUNT crash on CPU.
def _override_rope_with_arm():
    try:
        import flag_gems as _fg
        from .rope import arm_apply_rotary_pos_emb as _arm_rope
        _fg.apply_rotary_pos_emb = _arm_rope
        import flag_gems.fused as _fg_fused
        _fg_fused.apply_rotary_pos_emb = _arm_rope
        _logging.getLogger(__name__).debug(
            "FlagGems ARM: overrode flag_gems.apply_rotary_pos_emb with pure-PyTorch"
        )
    except Exception as e:
        _logging.getLogger(__name__).warning(
            f"FlagGems ARM: failed to override apply_rotary_pos_emb: {e}"
        )


_override_rope_with_arm()


# Override flag_gems.silu_and_mul / silu_and_mul_out with ARM Triton version.
# The generic fused/silu_and_mul.py uses @pointwise_dynamic → @libentry() → crash.
def _override_silu_and_mul_with_arm():
    try:
        import flag_gems as _fg
        from .silu_and_mul import arm_silu_and_mul as _arm_sam
        from .silu_and_mul import arm_silu_and_mul_out as _arm_sam_out
        _fg.silu_and_mul = _arm_sam
        _fg.silu_and_mul_out = _arm_sam_out
        import flag_gems.fused as _fg_fused
        _fg_fused.silu_and_mul = _arm_sam
        _fg_fused.silu_and_mul_out = _arm_sam_out
        _logging.getLogger(__name__).debug(
            "FlagGems ARM: overrode flag_gems.silu_and_mul / silu_and_mul_out"
        )
    except Exception as e:
        _logging.getLogger(__name__).warning(
            f"FlagGems ARM: failed to override silu_and_mul: {e}"
        )


_override_silu_and_mul_with_arm()


# Register FlagGems ARM mm for aten::mm (BF16 decode speedup).
# M=1 decode: 2-5x faster than ATen (ATen GEMV unoptimized).
# M=64 prefill: 2-3x slower (ATen uses native BF16 BFMMLA) — acceptable for
# decode-heavy workloads where decode runs O(n_tokens) vs prefill once.
# _mm_aten_lib must stay alive (GC would revoke the registration).
try:
    from .mm import mm as _fg_mm
    _mm_aten_lib = _torch.library.Library("aten", "IMPL")
    _mm_aten_lib.impl("mm", _fg_mm, "CPU", allow_override=True)
    _logging.getLogger(__name__).debug("FlagGems ARM: registered Triton-CPU mm for aten::mm")
except Exception as _e:
    _mm_aten_lib = None
    _logging.getLogger(__name__).warning(f"FlagGems ARM: failed to register aten::mm override: {_e}")


# Register Triton-CPU Flash Attention for aten::scaled_dot_product_attention.
# Prefill (M >= 32, BF16, no attn_mask): 4-5x faster than ATen on ARM64.
# Decode (M=1) and other cases fall back to ATen automatically.
# Strategy: try torch.library first; fall back to F.sdpa monkey-patch if needed.
def _register_sdpa():
    # NOTE: We intentionally use monkey-patch instead of torch.library here.
    #
    # torch.library.Library("aten", "IMPL").impl("scaled_dot_product_attention", ...)
    # replaces the C++ dispatch for the op. This means ALL calls to the op —
    # including the "fallback to ATen" call inside our own wrapper (_aten_sdpa) —
    # would route back to our function, causing infinite recursion.
    #
    # With monkey-patch, _aten_sdpa in attention.py holds a reference to the
    # original Python function object (captured at import time before patching).
    # When our wrapper calls _aten_sdpa(...), it goes through the original C++
    # dispatch WITHOUT our override, breaking the recursion.
    try:
        from .attention import scaled_dot_product_attention as _fg_sdpa
        import torch.nn.functional as _F
        _F.scaled_dot_product_attention = _fg_sdpa
        _logging.getLogger(__name__).debug(
            "FlagGems ARM: monkey-patched F.scaled_dot_product_attention "
            "with Triton Flash Attention (prefill 4-5x speedup)"
        )
    except Exception as _e:
        _logging.getLogger(__name__).warning(
            f"FlagGems ARM: SDPA registration failed: {_e}"
        )


_register_sdpa()
