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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# The four metadata outputs are constants for a given (batch, seq_q, seq_k):
# one packed template per shape, copied into a fresh buffer on every call.
_METADATA_TEMPLATES = {}

# Zero-element debug masks, one per (dtype, device). A zero-element tensor holds
# no data, so reusing one is indistinguishable from allocating a fresh one.
_EMPTY_MASKS = {}

# The tiled flash kernel, its tile config and its launch grid, resolved once
# (see _tiled_flash_forward).
_FLASH_KERNEL = None
_FLASH_TILE = None

# Accumulation knobs for the tiled flash kernel, read once per process.
_FA_ACCUM_TYPE = os.environ.get("FLAG_GEMS_FA_ACCUM_TYPE", "float")
_FA_GEMM_TYPE = os.environ.get("FLAG_GEMS_FA_GEMM_TYPE", "float")


def _metadata_template(batch_size, seq_q, seq_k, device):
    """Packed metadata constants for one shape, as ``2 * batch_size + 2`` int64 lanes.

    Layout ``[cum_seq_q | cum_seq_k | philox_seed | philox_offset]``, padded up to
    a power of two. A cum_seq lane packs its ``(0, seq)`` int32 pair, so the low
    word is 0 and the high word is the sequence length.
    """
    key = (batch_size, seq_q, seq_k, device)
    tmpl = _METADATA_TEMPLATES.get(key)
    if tmpl is None:
        width = triton.next_power_of_2(2 * batch_size + 2)
        host = torch.zeros(width, dtype=torch.int64)
        for i in range(batch_size):
            host[i] = seq_q << 32
            host[batch_size + i] = seq_k << 32
        tmpl = host.to(device)
        _METADATA_TEMPLATES[key] = tmpl
    return tmpl


def _empty_debug_mask(query):
    """The empty debug mask returned when ``return_debug_mask`` is false."""
    key = (query.dtype, query.device)
    mask = _EMPTY_MASKS.get(key)
    if mask is None:
        mask = _EMPTY_MASKS[key] = torch.empty(
            0, dtype=query.dtype, device=query.device
        )
    return mask


def _tiled_flash_forward():
    """Resolve the backend's tiled flash-attention kernel ``_attn_fwd``.

    Resolved lazily, through THIS module's own package, and cached together with
    the tile config its launch needs: importing this module must not change the
    backend package's import order, and the hot path must not repeat the import
    lookup nor the per-call argument binding the tile-config wrapper performs.
    """
    global _FLASH_KERNEL, _FLASH_TILE
    if _FLASH_KERNEL is None:
        att = importlib.import_module(__package__ + ".attention")
        kern = getattr(att, "_attn_fwd", None)
        configs = getattr(att, "configs", None)
        if kern is None or not configs or len(configs) != 1:
            raise NotImplementedError(
                "kunlunxin attention module does not expose a single-config tiled "
                "flash kernel _attn_fwd"
            )
        cfg = configs[0]
        _FLASH_KERNEL = kern.fn
        _FLASH_TILE = dict(
            cfg.kwargs, num_warps=cfg.num_warps, num_stages=cfg.num_stages
        )
    return _FLASH_KERNEL


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
    if attn_bias is None:
        attn_mask, has_bias = None, False
    else:
        attn_mask, has_bias = _normalize_attn_bias(
            attn_bias,
            batch_size,
            num_query_heads,
            seq_q,
            seq_k,
            query.dtype,
            query.device,
        )

    # The four metadata outputs are shape constants: copy the packed template
    # into a fresh buffer and hand out one view per output.
    metadata = _metadata_template(batch_size, seq_q, seq_k, query.device).clone()
    cum_seq_q = metadata[:batch_size].view(torch.int32).view(batch_size, 2)
    cum_seq_k = (
        metadata[batch_size : 2 * batch_size].view(torch.int32).view(batch_size, 2)
    )
    philox_seed = metadata[2 * batch_size : 2 * batch_size + 1]
    philox_offset = metadata[2 * batch_size + 1 : 2 * batch_size + 2]

    output = torch.empty_like(query)
    logsumexp = torch.empty(
        (batch_size, num_query_heads, seq_q), dtype=torch.float32, device=query.device
    )
    debug_attn_mask = (
        torch.empty(
            (batch_size, num_query_heads, seq_q, seq_k),
            dtype=query.dtype,
            device=query.device,
        )
        if return_debug_mask
        else _empty_debug_mask(query)
    )

    # Delegate the score/softmax/output computation to the backend's tiled
    # flash-attention kernel instead of re-deriving a serial per-row sweep.
    # `_attn_fwd` is the same kernel the backend's autograd attention entry
    # launches; it writes the fp32 (B, H, S) log-sum-exp into its `M` argument.
    sm_scale = scale if scale is not None else 1.0 / (head_dim**0.5)
    stage = 3 if is_causal else 1
    fa_accum_type = _FA_ACCUM_TYPE
    fa_gemm_type = _FA_GEMM_TYPE
    kernel = _FLASH_KERNEL
    if kernel is None:
        kernel = _tiled_flash_forward()
    tile = _FLASH_TILE
    grid = (triton.cdiv(seq_q, tile["BLOCK_M"]), batch_size * num_query_heads, 1)

    # The launch goes to the current device, so switch to the tensors' device
    # first -- but only when it is not already the current one. The launch is
    # written out twice on purpose.
    device_index = query.device.index
    if device_index is None or device_index == torch.cuda.current_device():
        kernel[grid](
            query,
            key,
            value,
            attn_mask,  # must stay None when there is no mask
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
            **tile,
        )
    else:
        with torch_device_fn.device(query.device):
            kernel[grid](
                query,
                key,
                value,
                attn_mask,
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
                **tile,
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
