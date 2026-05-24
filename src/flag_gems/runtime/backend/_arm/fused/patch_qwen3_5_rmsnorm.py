"""Monkey-patch Qwen3_5RMSNorm.forward to use the existing
tle_ops.rms_norm @builtin (NEON RMSNorm in C runtime), eliminating the
~30 us ATen sequence (pow + mean + rsqrt + mul × 2) per call.

Per Qwen3.5-2B decode token, RMSNorm is invoked at:
  - input_layernorm (per decoder layer)            : 24 calls
  - post_attention_layernorm (per decoder layer)   : 24 calls
  - q_norm, k_norm (full_attention layers)         : 6 layers × 2 = 12 calls
Total: ~60 RMSNorm calls per token → ~1.8 ms/token saved if each call drops
from 30 us to 5 us.

Qwen3.5's RMSNorm uses `out = (x / rms(x)) * (1 + weight)` (note the +1).
We pre-compute `_weight_plus_one = weight + 1.0` at patch time so the
existing tle.rms_norm (which computes `out = (x / rms(x)) * w_in`) gives
the right result.

Decode (M=1, BF16) hits the fast path. Other shapes / dtypes fall back
to the original forward.
"""
import logging
import types

import torch
import triton
import triton.language as tl

from triton.language.extra.cpu.tle_ops import rms_norm as _tle_rms_norm

logger = logging.getLogger(__name__)

_PATCHED: set = set()


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, D: tl.constexpr, eps: tl.constexpr):
    _tle_rms_norm(x_ptr, w_ptr, out_ptr, D, eps)


def _patched_rmsnorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    # Fast path: bf16 input, last-dim contiguous, single row.
    if (x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.shape[-1] == self._tle_D):
        # Reshape to [M, D] flat
        shape = x.shape
        D = self._tle_D
        M = x.numel() // D
        if M == 1:
            xc = x.reshape(D).contiguous()
            out = torch.empty(D, dtype=torch.bfloat16)
            _rms_norm_kernel[(1,)](xc, self._weight_plus_one_bf16, out,
                                     D=D, eps=float(self.eps))
            return out.reshape(*shape)
        # Multi-row: call kernel M times. For decode M=1 we never hit this.
        out = torch.empty_like(x)
        x_2d = x.reshape(M, D)
        out_2d = out.reshape(M, D)
        for i in range(M):
            _rms_norm_kernel[(1,)](x_2d[i].contiguous(),
                                     self._weight_plus_one_bf16,
                                     out_2d[i],
                                     D=D, eps=float(self.eps))
        return out

    # Slow / fallback path: original forward
    return self._original_forward(x)


def _get_qwen3_5_rmsnorm_classes():
    classes = []
    for modname, clsname in [
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5RMSNorm"),
        ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5RMSNorm"),
        ("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextRMSNorm"),
    ]:
        try:
            mod = __import__(modname, fromlist=[clsname])
            classes.append(getattr(mod, clsname))
        except (ImportError, AttributeError):
            pass
    return tuple(classes)


def patch_qwen3_5_rmsnorm(model) -> int:
    rms_classes = _get_qwen3_5_rmsnorm_classes()
    if not rms_classes:
        return 0
    n = 0
    for _name, mod in list(model.named_modules()):
        if isinstance(mod, rms_classes) and id(mod) not in _PATCHED:
            D = mod.weight.shape[0]
            # Pre-compute weight + 1.0 in bf16 (matches Qwen3.5's RMSNorm formula)
            mod._weight_plus_one_bf16 = (1.0 + mod.weight).to(
                torch.bfloat16).contiguous()
            mod._tle_D = D
            mod._original_forward = mod.forward
            mod.forward = types.MethodType(_patched_rmsnorm_forward, mod)
            _PATCHED.add(id(mod))
            n += 1
    if n > 0:
        logger.info("Patched %d Qwen3.5 RMSNorm modules with TLE rms_norm", n)
    return n


def unpatch_qwen3_5_rmsnorm(model) -> int:
    rms_classes = _get_qwen3_5_rmsnorm_classes()
    if not rms_classes:
        return 0
    n = 0
    for _name, mod in list(model.named_modules()):
        if isinstance(mod, rms_classes) and id(mod) in _PATCHED:
            if hasattr(mod, "_original_forward"):
                mod.forward = mod._original_forward
                del mod._original_forward
                del mod._weight_plus_one_bf16
                del mod._tle_D
            _PATCHED.discard(id(mod))
            n += 1
    return n
