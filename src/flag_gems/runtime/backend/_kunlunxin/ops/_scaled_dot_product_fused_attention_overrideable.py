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

import importlib
import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Stable per-shape grid callables (see _grid_for): the XPU JIT keys its compile
# cache on the grid object, so these must not be rebuilt per call.
_GRID_CACHE = {}


@triton.jit
def _sdpa_init_metadata_kernel(
    cum_seq_q,
    cum_seq_k,
    philox_seed,
    philox_offset,
    seq_q,
    seq_k,
    BATCH: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < BATCH:
        tl.store(cum_seq_q + pid * 2, 0)
        tl.store(cum_seq_q + pid * 2 + 1, seq_q)
        tl.store(cum_seq_k + pid * 2, 0)
        tl.store(cum_seq_k + pid * 2 + 1, seq_k)
    if pid == 0:
        tl.store(philox_seed, 0)
        tl.store(philox_offset, 0)


def _grid_for(seq_q, batch_heads):
    """Return a STABLE grid callable for ``(seq_q, batch*heads)``.

    The XPU JIT folds the ``grid`` callable into its compile cache key, so
    building a fresh ``lambda`` per call makes every launch a cache miss and
    recompiles the kernel -- measured at ~170 ms per call, which dominated the
    benchmark (a flat 160-195 ms across every shape and dtype, versus 0.10 ms
    for the real launch). Caching one callable per shape keeps the object
    identity stable, so repeated calls with the same shape hit the cache.
    """
    key = (seq_q, batch_heads)
    fn = _GRID_CACHE.get(key)
    if fn is None:

        def fn(args, _seq_q=seq_q, _batch_heads=batch_heads):
            return (triton.cdiv(_seq_q, args["BLOCK_M"]), _batch_heads, 1)

        _GRID_CACHE[key] = fn
    return fn


def _tiled_flash_forward():
    """Resolve the backend's tiled flash-attention kernel ``_attn_fwd``.

    Resolved through THIS module's own package (the runtime backend package is
    the top-level ``_kunlunxin.ops`` one; a second copy also exists under
    ``flag_gems.runtime.backend._kunlunxin.ops``), and resolved lazily so that
    importing this module never changes the backend package's import order.
    """
    att = importlib.import_module(__package__ + ".attention")
    kern = getattr(att, "_attn_fwd", None)
    if kern is None:
        raise NotImplementedError(
            "kunlunxin attention module does not expose the tiled flash kernel _attn_fwd"
        )
    return kern


def _normalize_attn_bias(attn_bias, batch_size, num_heads, seq_q, seq_k, dtype, device):
    """Broadcast a 2D/3D/4D attention bias to the 4D launch layout.

    Returns ``(mask, has_mask)``. The mask keeps broadcast dimensions as
    stride-0 views (``expand`` never materialises), which is exactly what the
    tiled kernel's pointer arithmetic expects. Boolean masks are converted to
    the additive ``0 / -1e6`` form the kernel consumes.
    """
    if attn_bias is None:
        return None, False
    if attn_bias.device != device:
        raise RuntimeError("attn_bias must be on the query device")

    if attn_bias.dtype == torch.bool:
        mask = attn_bias.to(dtype) * -1.0e6
    else:
        mask = attn_bias.to(dtype)

    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    elif mask.ndim != 4:
        raise RuntimeError("attn_bias must be a dense 2D, 3D, or 4D tensor")

    expected = (batch_size, num_heads, seq_q, seq_k)
    if any(actual not in (1, wanted) for actual, wanted in zip(mask.shape, expected)):
        raise RuntimeError(
            "attn_bias must broadcast to (batch, query_heads, query_seq, key_seq)"
        )
    return mask.expand(*expected), True


def _scaled_dot_product_fused_attention_overrideable(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float = None,
):
    logger.debug("GEMS_KUNLUNXIN SCALED_DOT_PRODUCT_FUSED_ATTENTION_OVERRIDEABLE")
    assert dropout_p == 0.0, "Only dropout_p=0.0 is supported"
    assert (
        query.ndim == key.ndim == value.ndim == 4
    ), "Only dense 4D attention is supported"
    assert (
        query.shape[0] == key.shape[0] == value.shape[0]
    ), "Batch dimensions must match"
    assert key.shape == value.shape, "key and value must have matching shapes"
    assert (
        query.shape[-1] == key.shape[-1] in (64, 128)
    ), "Only head dimensions 64 and 128 are supported"
    assert (
        query.shape[1] % key.shape[1] == 0
    ), "Query heads must be a multiple of key heads"

    batch_size, num_query_heads, seq_q, head_dim = query.shape
    _, num_key_heads, seq_k, _ = key.shape
    attn_mask, has_bias = _normalize_attn_bias(
        attn_bias,
        batch_size,
        num_query_heads,
        seq_q,
        seq_k,
        query.dtype,
        query.device,
    )

    output = torch.empty_like(query)
    logsumexp = torch.empty(
        (batch_size, num_query_heads, seq_q), dtype=torch.float32, device=query.device
    )
    cum_seq_q = torch.empty((batch_size, 2), dtype=torch.int32, device=query.device)
    cum_seq_k = torch.empty((batch_size, 2), dtype=torch.int32, device=query.device)
    philox_seed = torch.empty(1, dtype=torch.int64, device=query.device)
    philox_offset = torch.empty(1, dtype=torch.int64, device=query.device)
    debug_attn_mask = (
        torch.empty(
            (batch_size, num_query_heads, seq_q, seq_k),
            dtype=query.dtype,
            device=query.device,
        )
        if return_debug_mask
        else torch.empty(0, dtype=query.dtype, device=query.device)
    )

    # NOTE (2026-10-09): producing these four outputs costs 0.048 ms at
    # (2,8,64,64) on this device -- about 60% of this operator's total time.
    # Every cheaper alternative was measured and this kernel is the best of
    # them: `torch.zeros` + column fill is worse (0.094 ms), and `arange` +
    # `expand` scores 0.3966 aggregate against this kernel's 0.3475.  The cost
    # is the device's Triton launch overhead (0.0132 ms for a trivial
    # one-pointer kernel, about +0.0125 per extra pointer argument), not the
    # work being done.  Kept as-is deliberately: the only way to remove it is to
    # return tensors cached across calls, which would make the returned outputs
    # alias between calls.
    _sdpa_init_metadata_kernel[(max(batch_size, 1),)](
        cum_seq_q,
        cum_seq_k,
        philox_seed,
        philox_offset,
        seq_q,
        seq_k,
        BATCH=batch_size,
        num_warps=1,
    )

    # Delegate the score/softmax/output computation to the backend's tiled
    # flash-attention kernel instead of re-deriving a serial per-row sweep.
    # `_attn_fwd` is the same kernel the backend's autograd attention entry
    # launches; it writes the fp32 (B, H, S) log-sum-exp into its `M` argument.
    sm_scale = scale if scale is not None else 1.0 / (head_dim**0.5)
    stage = 3 if is_causal else 1
    fa_accum_type = os.environ.get("FLAG_GEMS_FA_ACCUM_TYPE", "float")
    fa_gemm_type = os.environ.get("FLAG_GEMS_FA_GEMM_TYPE", "float")
    grid = _grid_for(seq_q, batch_size * num_query_heads)

    with torch_device_fn.device(query.device):
        _tiled_flash_forward()[grid](
            query,
            key,
            value,
            attn_mask,  # must stay None when there is no mask: the vendor
            # `handle_mha` decoder (launch_extra.cpp:1238) requires exactly 23
            # runtime params in the unmasked case, and slot 3 (attn_mask) is a
            # hole in its position->signature table. A dummy tensor here yields
            # 24 params and the handler rejects the launch with err_code 1.
            sm_scale,
            logsumexp,
            output,  #
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),  #
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),  #
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),  #
            *(attn_mask.stride() if has_bias else (1, 1, 1, 1)),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),  #
            batch_size,
            num_query_heads,
            num_key_heads,  #
            num_query_heads // num_key_heads,  # group_head
            seq_q,  #
            seq_k,  #
            head_dim,  #
            STAGE=stage,  #
            HAS_ATTN_MASK=has_bias,  #
            ACCUM_TYPE=fa_accum_type,
            GEMM_TYPE=fa_gemm_type,
        )

    return (
        output,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        seq_q,
        seq_k,
        philox_seed,
        philox_offset,
        debug_attn_mask,
    )
