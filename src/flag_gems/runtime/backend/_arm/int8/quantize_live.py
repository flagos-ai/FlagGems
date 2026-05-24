"""Live (in-memory) W8 per-channel symmetric quantization of nn.Linear layers,
replacing each with a TLEInt8Linear.

Use case: take a BF16 model that has no pre-quantized state dict (e.g. the
public Qwen3.5-2B BF16 release) and turn it into a TLE INT8 stack on the fly.

Quantization scheme matches the llm-compressor / compressed-tensors W8A8
output format (per-channel symmetric INT8 weights, per-token symmetric INT8
activations). Activation quantization is already done inside
TLEInt8Linear.forward; this helper only handles the weight side.

Example:
    from transformers import AutoModelForCausalLM
    from flag_gems.runtime.backend._arm.int8 import quantize_and_replace_linears

    m = AutoModelForCausalLM.from_pretrained("...", dtype=torch.bfloat16)
    n = quantize_and_replace_linears(m, skip={"lm_head"})
"""
import logging
from typing import Iterable, Optional, Tuple

import torch

from .tle_int8_linear import TLEInt8Linear

logger = logging.getLogger(__name__)


def _quantize_weight_per_channel_sym(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric INT8 quant of an [N, K] weight.

    Returns:
        w_int8:   [N, K] int8
        w_scale:  [N]    fp32   (per-channel)
    """
    w_fp32 = w.detach().to(torch.float32)
    # max(|w|) along K (axis 1)
    absmax = w_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # [N, 1]
    scale = absmax / 127.0  # [N, 1]
    w_int8 = (w_fp32 / scale).round().clamp(-128, 127).to(torch.int8)  # [N, K]
    scale_flat = scale.squeeze(-1).contiguous().to(torch.float32)  # [N]
    return w_int8, scale_flat


def quantize_and_replace_linears(
    model: torch.nn.Module,
    skip: Optional[Iterable[str]] = None,
    require_divisible_by: int = 4,
    skip_with_bias: bool = True,
) -> int:
    """Walk model.named_modules(), in-memory quantize each nn.Linear weight to
    per-channel symmetric INT8, and replace it with a TLEInt8Linear.

    Args:
        model: any torch.nn.Module (typically a transformers model).
        skip:  iterable of module names to leave alone (e.g. {"lm_head"} when
               the head is tied or too large to benefit from INT8 GEMV).
        require_divisible_by: SDOT requires K%4==0 and N%4==0; smaller-aligned
               linears stay BF16 (e.g. tiny GDN scalar projections in_proj_a/b).
        skip_with_bias: TLEInt8Linear has no bias parameter; if True, leave any
               nn.Linear with a non-None bias as-is. Set False to assert no
               biases exist (matches Qwen3-style models).

    Returns: number of Linear modules replaced.
    """
    skip_set = set(skip) if skip else set()
    n_replaced = 0
    n_skipped_align = 0
    n_skipped_bias = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if name in skip_set:
            continue
        if module.bias is not None:
            if skip_with_bias:
                n_skipped_bias += 1
                continue
            raise ValueError(
                f"{name} has bias=True; TLEInt8Linear does not support bias")

        N, K = module.weight.shape
        if K % require_divisible_by != 0 or N % require_divisible_by != 0:
            n_skipped_align += 1
            logger.debug(
                "quantize_and_replace_linears: %s K=%d N=%d not divisible by %d",
                name, K, N, require_divisible_by,
            )
            continue

        w_int8, w_scale = _quantize_weight_per_channel_sym(module.weight.data)

        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], TLEInt8Linear(w_int8, w_scale))

        # Free the original BF16 weight memory.
        del module
        n_replaced += 1

    logger.info(
        "quantize_and_replace_linears: replaced %d Linears "
        "(skipped: %d alignment, %d bias, %d explicit)",
        n_replaced, n_skipped_align, n_skipped_bias, len(skip_set),
    )
    return n_replaced
