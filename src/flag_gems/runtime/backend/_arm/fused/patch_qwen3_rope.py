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

"""Monkey-patch transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb
to use a TLE @triton.jit RoPE kernel for the decode (T=1) BF16 hot path.

Replaces 4 ATen muls + 2 adds + slice/cat of rotate_half with two single
Triton kernel launches (one for q, one for k). Goes through the @triton.jit
TLE path (NOT ctypes) per project requirement.

Decode (B=1, T=1, BF16 q/k contiguous) hits the fast path. Other shapes fall
back to the original PyTorch implementation.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_PATCHED: dict = {}


@triton.jit
def _rope_qk_bf16_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    n_heads_q,
    head_dim,
    half,
    BLOCK_HALF: tl.constexpr,
):
    """Apply RoPE in-place to q (heads 0..n_heads_q) and k (heads n_heads_q..total)
    in a single kernel launch. Grid is (n_heads_q + n_heads_kv,).

    Layout: q_ptr / k_ptr point to flat memory [n_heads * head_dim] bf16 each.
    cos_ptr / sin_ptr point to [half] bf16 (interleaved RoPE convention:
    cos/sin first half repeated for the second half).
    """
    pid = tle.program_id(0)
    is_q = pid < n_heads_q
    # Branch on q vs k by selecting the right base + index
    row = tl.where(is_q, q_ptr + pid * head_dim, k_ptr + (pid - n_heads_q) * head_dim)
    for off in range(0, half, BLOCK_HALF):
        d = off + tl.arange(0, BLOCK_HALF)
        mask = d < half
        q0 = tl.load(row + d, mask=mask, other=0.0).to(tl.float32)
        q1 = tl.load(row + half + d, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos_ptr + d, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + d, mask=mask, other=0.0).to(tl.float32)
        r0 = q0 * c - q1 * s
        r1 = q0 * s + q1 * c
        tl.store(row + d, r0.to(q_ptr.dtype.element_ty), mask=mask)
        tl.store(row + half + d, r1.to(q_ptr.dtype.element_ty), mask=mask)


_PREWARM_DONE = False


def _prewarm():
    global _PREWARM_DONE
    if _PREWARM_DONE:
        return
    try:
        for hd in (128,):
            q = torch.zeros((2, hd), dtype=torch.bfloat16)
            k = torch.zeros((2, hd), dtype=torch.bfloat16)
            c = torch.zeros(hd // 2, dtype=torch.bfloat16)
            s = torch.zeros(hd // 2, dtype=torch.bfloat16)
            _rope_qk_bf16_kernel[(4,)](
                q,
                k,
                c,
                s,
                2,
                hd,
                hd // 2,
                BLOCK_HALF=64,
                num_warps=1,
                num_stages=1,
            )
    except Exception:
        logger.debug("GEMS_ARM rope prewarm failed", exc_info=True)
    _PREWARM_DONE = True


def _rope_bf16_jit(q, k, cos_half, sin_half, n_heads_q, n_heads_kv, head_dim):
    """Apply RoPE in-place via single @triton.jit kernel launch (q+k fused)."""
    _prewarm()
    half = head_dim // 2
    total = n_heads_q + n_heads_kv
    _rope_qk_bf16_kernel[(total,)](
        q,
        k,
        cos_half,
        sin_half,
        n_heads_q,
        head_dim,
        half,
        BLOCK_HALF=64,
        num_warps=1,
        num_stages=1,
    )


def _patched_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Fast path uses @triton.jit kernel for decode B=1 T=1 BF16; else fall back."""
    # Fast path conditions
    if (
        q.dim() == 4
        and k.dim() == 4
        and q.shape[0] == 1
        and k.shape[0] == 1
        and q.shape[2] == 1
        and k.shape[2] == 1
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and cos.dim() in (2, 3, 4)
        and sin.dim() in (2, 3, 4)
        and q.is_contiguous()
        and k.is_contiguous()
    ):
        n_heads = q.shape[1]
        n_kv_heads = k.shape[1]
        head_dim = q.shape[3]
        # Kernel uses interleaved convention: only reads cos/sin first half (assumes
        # cos[d/2:]==cos[:d/2] which is HF's standard repeat-half RoPE pattern).
        cos_half = cos.reshape(-1, head_dim)[0, : head_dim // 2].contiguous()
        sin_half = sin.reshape(-1, head_dim)[0, : head_dim // 2].contiguous()
        # Kernel writes in-place; clone q/k to preserve original (HF semantics).
        q_buf = q.clone()
        k_buf = k.clone()
        _rope_bf16_jit(q_buf, k_buf, cos_half, sin_half, n_heads, n_kv_heads, head_dim)
        return q_buf, k_buf

    # Fallback: original PyTorch implementation
    return _PATCHED["original"](q, k, cos, sin, unsqueeze_dim)


def patch_qwen3_rope() -> int:
    """Monkey-patch apply_rotary_pos_emb in transformers.models.qwen3.

    Returns count of patched modules.
    """
    # Targets regular Qwen3 only. Qwen3.5 supports partial rotary
    # (q_rot vs q_pass split); needs separate handling.
    targets = [
        "transformers.models.qwen3.modeling_qwen3",
    ]
    n = 0
    for modname in targets:
        try:
            mod = __import__(modname, fromlist=["apply_rotary_pos_emb"])
        except (ImportError, AttributeError):
            continue
        if not hasattr(mod, "apply_rotary_pos_emb"):
            continue
        if modname in _PATCHED:
            continue
        original = getattr(mod, "apply_rotary_pos_emb")
        _PATCHED["original"] = original
        _PATCHED[modname] = True
        setattr(mod, "apply_rotary_pos_emb", _patched_apply_rotary_pos_emb)
        n += 1
        logger.info(f"GEMS_ARM Patched {modname}.apply_rotary_pos_emb")
    return n


def unpatch_qwen3_rope() -> int:
    n = 0
    for modname in list(_PATCHED.keys()):
        if modname == "original":
            continue
        try:
            mod = __import__(modname, fromlist=["apply_rotary_pos_emb"])
        except (ImportError, AttributeError):
            continue
        if "original" in _PATCHED:
            setattr(mod, "apply_rotary_pos_emb", _PATCHED["original"])
        del _PATCHED[modname]
        n += 1
    return n
