"""ARM CPU INT8 model utilities.

Drop-in Linear replacement with decode-optimized TLE SDOT GEMV + prefill
via torch._int_mm (SVE2 i8mm), plus a helper to replace all nn.Linear
layers in a transformers model from a pre-quantized state dict.

Usage:
    from safetensors.torch import load_file
    from flag_gems.runtime.backend._arm.int8 import replace_linears_with_tle_int8

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=bf16)
    state = load_file("Qwen3-1.7B-W8A8-INT8/model.safetensors")
    replace_linears_with_tle_int8(model, state)
"""

from .tle_int8_linear import TLEInt8Linear, pack_weights_sdot  # noqa: F401
from .replace import replace_linears_with_tle_int8  # noqa: F401
from .quantize_live import quantize_and_replace_linears  # noqa: F401

__all__ = [
    "TLEInt8Linear",
    "pack_weights_sdot",
    "replace_linears_with_tle_int8",
    "quantize_and_replace_linears",
]
