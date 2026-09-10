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

"""Monkey-patch Qwen3_5GatedDeltaNet.causal_conv1d_update with a TLE-CPU
fused depthwise conv1d update kernel.

The HF fallback torch_causal_conv1d_update goes through aten::conv1d
(groups=conv_dim, kernel_size=4) which uses MKL-DNN with high dispatch
overhead (~700us/call on ARM ACL). Profile shows 7% of decode time spent
here on Qwen3.5-2B BF16.

Replaces the entire torch_causal_conv1d_update body with one C kernel call
that does cat-with-state + conv + (optional silu) + state-roll in a single
NEON OMP loop.

Decode (T=1, kernel_size=4, BF16) only — other shapes fall back to torch.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import (
    causal_conv1d_update as _tle_causal_conv1d_update,
)

logger = logging.getLogger(__name__)

_PATCHED: set = set()
_DUMMY_BIAS = torch.zeros(1, dtype=torch.bfloat16)


@triton.jit
def _causal_conv1d_update_kernel(
    hidden_ptr,
    state_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    kernel_size: tl.constexpr,
    silu: tl.constexpr,
    has_bias: tl.constexpr,
):
    _tle_causal_conv1d_update(
        hidden_ptr,
        state_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        B,
        C,
        kernel_size,
        silu,
        has_bias,
    )


def _make_patched_fn(torch_causal_fn):
    def fn(hidden_states, conv_state, weight, bias=None, activation=None):
        # hidden_states: [B, C, T] bf16; weight: [C, kernel_size]; bias: None or [C]
        # conv_state:    [B, C, kernel_size-1] bf16 IN-OUT
        # activation: 'silu' or None.
        if (
            hidden_states.shape[-1] != 1
            or weight.shape[-1] != 4
            or hidden_states.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or conv_state.dtype != torch.bfloat16
            or (activation not in ("silu", None))
        ):
            return torch_causal_fn(hidden_states, conv_state, weight, bias, activation)

        B, C, _T = hidden_states.shape
        # [B, C] contiguous
        h = hidden_states.squeeze(-1).contiguous()
        w = weight.contiguous()
        # conv_state must be contiguous so the kernel can update it in place.
        if not conv_state.is_contiguous():
            conv_state_c = conv_state.contiguous()
        else:
            conv_state_c = conv_state
        out = torch.empty(B, C, dtype=torch.bfloat16)
        if bias is None:
            b_t = _DUMMY_BIAS
            has_bias = 0
        else:
            b_t = bias.contiguous()
            has_bias = 1
        silu_flag = 1 if activation == "silu" else 0

        _causal_conv1d_update_kernel[(1,)](
            h,
            conv_state_c,
            w,
            b_t,
            out,
            B=B,
            C=C,
            kernel_size=4,
            silu=silu_flag,
            has_bias=has_bias,
        )

        # If we made a contiguous copy of conv_state, write back.
        if conv_state_c.data_ptr() != conv_state.data_ptr():
            conv_state.copy_(conv_state_c)

        return out.unsqueeze(-1)

    return fn


def _get_qwen_gdn_classes() -> tuple:
    classes = []
    for modname, clsname in [
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5GatedDeltaNet"),
        (
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
            "Qwen3_5MoeGatedDeltaNet",
        ),
        (
            "transformers.models.qwen3_next.modeling_qwen3_next",
            "Qwen3NextGatedDeltaNet",
        ),
    ]:
        try:
            mod = __import__(modname, fromlist=[clsname])
            classes.append(getattr(mod, clsname))
        except (ImportError, AttributeError):
            pass
    return tuple(classes)


def patch_qwen3_5_conv1d(model) -> int:
    gdn_classes = _get_qwen_gdn_classes()
    if not gdn_classes:
        return 0
    n = 0
    for _name, module in list(model.named_modules()):
        if isinstance(module, gdn_classes) and id(module) not in _PATCHED:
            torch_fn = module.causal_conv1d_update
            module._original_causal_conv1d_update = torch_fn
            module.causal_conv1d_update = _make_patched_fn(torch_fn)
            _PATCHED.add(id(module))
            n += 1
    if n > 0:
        logger.info("GEMS_ARM Patched %d GDN causal_conv1d_update with TLE kernel", n)
    return n


def unpatch_qwen3_5_conv1d(model) -> int:
    gdn_classes = _get_qwen_gdn_classes()
    if not gdn_classes:
        return 0
    n = 0
    for _name, module in list(model.named_modules()):
        if isinstance(module, gdn_classes) and id(module) in _PATCHED:
            if hasattr(module, "_original_causal_conv1d_update"):
                module.causal_conv1d_update = module._original_causal_conv1d_update
                del module._original_causal_conv1d_update
            _PATCHED.discard(id(module))
            n += 1
    return n
