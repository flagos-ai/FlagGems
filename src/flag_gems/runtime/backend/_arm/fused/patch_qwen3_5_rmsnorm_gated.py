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

"""Monkey-patch Qwen3_5RMSNormGated.forward to use the fused multi-row
tle_ops.rms_norm_gated builtin (NEON RMSNormGated in C runtime), replacing
a 6-op ATen sequence (pow + mean + rsqrt + mul × 2 + silu*mul) and turning
M single-row calls into one OMP-parallel multi-row kernel.

Per Qwen3.5-2B GDN decode token, RMSNormGated is invoked at:
  - GDN per-head norm (each linear_attention layer): 1 call/layer × 6 GDN
    layers × 1 token = 6 calls/tok, each shape [M=num_v_heads, D=head_v_dim]
    typically [16, 128].

Reference formula:
    out = (x / rms(x)) * weight * silu(gate)

Decode (BF16, [M, D] last-dim contiguous, M aligned). Other shapes /
dtypes fall back to the original forward.
"""

import logging
import types

import torch
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import rms_norm_gated as _tle_rms_norm_gated

logger = logging.getLogger(__name__)

_PATCHED: set = set()


@triton.jit
def _rms_norm_gated_kernel(
    x_ptr, gate_ptr, w_ptr, out_ptr, M: tl.constexpr, D: tl.constexpr, eps: tl.constexpr
):
    _tle_rms_norm_gated(x_ptr, gate_ptr, w_ptr, out_ptr, M, D, eps)


def _patched_rmsnorm_gated_forward(self, hidden_states, gate=None):
    if (
        gate is not None
        and hidden_states.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and gate.is_contiguous()
        and hidden_states.shape == gate.shape
        and hidden_states.shape[-1] == self._tle_D
    ):
        shape = hidden_states.shape
        D = self._tle_D
        M = hidden_states.numel() // D
        x_flat = hidden_states.reshape(M, D).contiguous()
        g_flat = gate.reshape(M, D).contiguous()
        out = torch.empty_like(x_flat)
        _rms_norm_gated_kernel[(1,)](
            x_flat,
            g_flat,
            self.weight.to(torch.bfloat16),
            out,
            M=M,
            D=D,
            eps=float(self.variance_epsilon),
        )
        return out.reshape(*shape)

    # Fallback: original forward
    return self._original_forward(hidden_states, gate)


def _get_qwen3_5_rmsnorm_gated_classes():
    classes = []
    for modname, clsname in [
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5RMSNormGated"),
        ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5RMSNormGated"),
        ("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextRMSNormGated"),
    ]:
        try:
            mod = __import__(modname, fromlist=[clsname])
            classes.append(getattr(mod, clsname))
        except (ImportError, AttributeError):
            pass
    return tuple(classes)


def patch_qwen3_5_rmsnorm_gated(model) -> int:
    rms_classes = _get_qwen3_5_rmsnorm_gated_classes()
    if not rms_classes:
        return 0
    n = 0
    for _name, mod in list(model.named_modules()):
        if isinstance(mod, rms_classes) and id(mod) not in _PATCHED:
            D = mod.weight.shape[0]
            mod._tle_D = D
            mod._original_forward = mod.forward
            mod.forward = types.MethodType(_patched_rmsnorm_gated_forward, mod)
            _PATCHED.add(id(mod))
            n += 1
    if n > 0:
        logger.info(
            "GEMS_ARM Patched %d Qwen3.5 RMSNormGated modules with TLE rms_norm_gated",
            n,
        )
    return n


def unpatch_qwen3_5_rmsnorm_gated(model) -> int:
    rms_classes = _get_qwen3_5_rmsnorm_gated_classes()
    if not rms_classes:
        return 0
    n = 0
    for _name, mod in list(model.named_modules()):
        if isinstance(mod, rms_classes) and id(mod) in _PATCHED:
            if hasattr(mod, "_original_forward"):
                mod.forward = mod._original_forward
                del mod._original_forward
                del mod._tle_D
            _PATCHED.discard(id(mod))
            n += 1
    return n
