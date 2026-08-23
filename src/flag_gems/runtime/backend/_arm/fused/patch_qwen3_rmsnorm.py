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

"""Patch Qwen3RMSNorm.forward to use a single Triton kernel call,
replacing the 5-6 ATen dispatches in the eager implementation.

Each Qwen3 decode token does ~113 RMSNorm calls:
  - 28 input_layernorm (hidden_size, e.g. 2048)
  - 1  final norm (hidden_size)
  - 28 post_attention_layernorm (already fused via patch_qwen3_layer_norm)
  - 28 q_norm (head_dim=128, batched across heads)
  - 28 k_norm (head_dim=128, batched across heads)

This patch targets the input/q/k/final norms (~85 calls/token).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_TILE = 128
_PATCHED: dict = {}


@triton.jit(do_not_specialize=["eps"])
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_r,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    x_row = x_ptr + pid * stride_r
    out_row = out_ptr + pid * stride_r

    sum_sq = tl.zeros([1], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / N + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        y = (x * rrms).to(out_ptr.dtype.element_ty) * w
        tl.store(out_row + cols, y, mask=mask)


_PREWARM_DONE = False


def _prewarm():
    global _PREWARM_DONE
    if _PREWARM_DONE:
        return
    try:
        for N in (128, 2048, 2560):
            x = torch.ones((1, N), dtype=torch.bfloat16)
            w = torch.ones(N, dtype=torch.bfloat16)
            o = torch.empty_like(x)
            _rms_norm_kernel[(1,)](
                x, w, o, N, N, 1e-6, BLOCK_SIZE=_TILE, num_warps=1, num_stages=1
            )
    except Exception:
        logger.debug("GEMS_ARM rmsnorm prewarm failed", exc_info=True)
    _PREWARM_DONE = True


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Standalone RMSNorm. Operates on last dim. Supports any leading dim shape."""
    _prewarm()
    N = weight.shape[0]
    assert x.shape[-1] == N
    M = x.numel() // N
    x_2d = x.reshape(M, N).contiguous()
    out = torch.empty_like(x_2d)
    _rms_norm_kernel[(M,)](
        x_2d,
        weight,
        out,
        N,  # stride_r
        N,
        eps,
        BLOCK_SIZE=_TILE,
        num_warps=1,
        num_stages=1,
    )
    return out.reshape(x.shape)


def _make_patched_forward(orig):
    def patched(self, hidden_states):
        # Fast path: BF16 only
        if hidden_states.dtype != torch.bfloat16:
            return orig(self, hidden_states)
        return _rms_norm(hidden_states, self.weight, self.variance_epsilon)

    return patched


def patch_qwen3_rmsnorm() -> int:
    # Targets regular Qwen3 only. Qwen3.5 has different math
    # (output * (1.0 + weight) and attr is `eps` not `variance_epsilon`);
    # use patch_qwen3_5_rmsnorm.py for that.
    targets = [
        "transformers.models.qwen3.modeling_qwen3",
    ]
    n = 0
    for modname in targets:
        try:
            mod = __import__(modname, fromlist=["Qwen3RMSNorm"])
        except (ImportError, AttributeError):
            continue
        cls_name = "Qwen3RMSNorm" if "qwen3_5" not in modname else "Qwen3_5RMSNorm"
        if not hasattr(mod, cls_name):
            cls_name = "Qwen3RMSNorm"
        if not hasattr(mod, cls_name):
            continue
        cls = getattr(mod, cls_name)
        key = (modname, cls_name)
        if key in _PATCHED:
            continue
        orig = cls.forward
        _PATCHED[key] = (cls, orig)
        cls.forward = _make_patched_forward(orig)
        n += 1
        logger.info(f"GEMS_ARM Patched {modname}.{cls_name}.forward")
    return n


def unpatch_qwen3_rmsnorm() -> int:
    n = 0
    for key, (cls, orig) in list(_PATCHED.items()):
        cls.forward = orig
        del _PATCHED[key]
        n += 1
    return n
