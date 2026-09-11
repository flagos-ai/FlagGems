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

"""Monkey-patch Qwen3MLP.forward to use triton-cpu's fused_mlp_bf16 kernel.

This patch replaces the 5-op ATen sequence
    gate_proj(x) → silu → up_proj(x) → mul → down_proj
with a single fused C kernel call when:
    - decode shape (M=1)
    - BF16 activation
    - gate_proj and up_proj are INT8 SDOT-packed Linears
      (expose attributes: _packed, _w_scale, K, N)

Measured benefit on Qwen3-1.7B W8A8-INT8 decode (3 rounds × 5 runs median,
CIX P1 CD8180, 8 big cores, OMP=8, performance governor):

  ENABLE_MLP_PATCH=1  ON  → 9.92 tok/s median (9.88, 10.04, 9.92)
  ENABLE_MLP_PATCH=0  OFF → 9.73 tok/s median (9.61, 9.73, 9.76)
  → +1.95% median (+2.5% mean) consistent across 3 rounds.

Usage:
    from flag_gems.runtime.backend._arm.fused.patch_qwen3_mlp import patch_qwen3_mlp
    patch_qwen3_mlp(model)
"""

import logging
import types

import torch
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import fused_mlp as _tle_fused_mlp

logger = logging.getLogger(__name__)

_PATCHED: set = set()


@triton.jit
def _fused_mlp_kernel(
    x_ptr,
    gate_packed_ptr,
    up_packed_ptr,
    gate_scale_ptr,
    up_scale_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    # Coarse TLE op: gate GEMV + up GEMV + SWIGLU fused in one kernel launch.
    _tle_fused_mlp(
        x_ptr,
        gate_packed_ptr,
        up_packed_ptr,
        gate_scale_ptr,
        up_scale_ptr,
        out_ptr,
        K,
        N,
    )


class FusedMLPWrapper:
    """Holds references to a Qwen3MLP's 3 projections + act_fn, exposes
    a forward that dispatches to the fused C kernel on M=1 BF16 decode.

    Falls back to composing gate/up/down via their own forward for:
      - M > 1 (prefill)
      - non-BF16 activation
      - gate/up not SDOT-packed INT8 Linears
    """

    def __init__(self, gate_linear, up_linear, down_linear, act_fn):
        self._gate_linear = gate_linear
        self._up_linear = up_linear
        self.down_proj = down_linear
        self.act_fn = act_fn

        self._fused = (
            hasattr(gate_linear, "_packed")
            and hasattr(up_linear, "_packed")
            and hasattr(gate_linear, "_w_scale")
            and hasattr(up_linear, "_w_scale")
            and hasattr(gate_linear, "K")
            and hasattr(gate_linear, "N")
        )
        if self._fused:
            self._gate_packed = gate_linear._packed
            self._up_packed = up_linear._packed
            self._gate_scale = gate_linear._w_scale
            self._up_scale = up_linear._w_scale
            self._K = gate_linear.K
            self._N = gate_linear.N

    def forward(self, x):
        shape = x.shape
        M = x.numel() // shape[-1]
        if self._fused and M == 1 and x.dtype == torch.bfloat16:
            xc = x.reshape(-1).contiguous()
            out = torch.empty(self._N, dtype=torch.bfloat16)
            _fused_mlp_kernel[(1,)](
                xc,
                self._gate_packed,
                self._up_packed,
                self._gate_scale,
                self._up_scale,
                out,
                K=self._K,
                N=self._N,
            )
            return self.down_proj(out.reshape(*shape[:-1], self._N))
        # ATen fallback: compose gate+up+silu+mul+down via each Linear's own forward
        gate = self._gate_linear(x)
        up = self._up_linear(x)
        return self.down_proj(self.act_fn(gate) * up)


def _get_qwen_mlp_classes() -> tuple:
    """Return a tuple of MLP classes to patch (Qwen3MLP + Qwen3_5MLP if available).

    Both classes share the same structure (gate_proj, up_proj, down_proj, act_fn),
    so the FusedMLPWrapper works on either.
    """
    classes = []
    for modname, clsname in [
        ("transformers.models.qwen3.modeling_qwen3", "Qwen3MLP"),
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5MLP"),
        ("transformers.models.llama.modeling_llama", "LlamaMLP"),  # MiniCPM5 etc.
    ]:
        try:
            mod = __import__(modname, fromlist=[clsname])
            classes.append(getattr(mod, clsname))
        except (ImportError, AttributeError):
            pass
    return tuple(classes)


def patch_qwen3_mlp(model) -> int:
    """Walk model, replace Qwen3MLP / Qwen3_5MLP forward with FusedMLPWrapper.

    Returns number of MLP instances patched. Safe to call multiple times —
    each instance is patched once (tracked via id).
    """
    mlp_classes = _get_qwen_mlp_classes()
    if not mlp_classes:
        logger.debug(
            "GEMS_ARM No Qwen MLP classes found in transformers, skipping patch"
        )
        return 0

    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, mlp_classes) and id(module) not in _PATCHED:
            wrapper = FusedMLPWrapper(
                module.gate_proj,
                module.up_proj,
                module.down_proj,
                module.act_fn,
            )
            module._original_forward = module.forward
            module._fused_mlp_wrapper = wrapper
            module.forward = types.MethodType(
                lambda self, x, _w=wrapper: _w.forward(x),
                module,
            )
            _PATCHED.add(id(module))
            n += 1
    if n > 0:
        cls_names = ", ".join(c.__name__ for c in mlp_classes)
        logger.info(
            "GEMS_ARM Patched %d MLP modules (classes: %s) with fused_mlp_bf16",
            n,
            cls_names,
        )
    return n


def unpatch_qwen3_mlp(model) -> int:
    """Restore original MLP forward (for testing / revert)."""
    mlp_classes = _get_qwen_mlp_classes()
    if not mlp_classes:
        return 0
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, mlp_classes) and id(module) in _PATCHED:
            if hasattr(module, "_original_forward"):
                module.forward = module._original_forward
                del module._original_forward
                del module._fused_mlp_wrapper
            _PATCHED.discard(id(module))
            n += 1
    return n
