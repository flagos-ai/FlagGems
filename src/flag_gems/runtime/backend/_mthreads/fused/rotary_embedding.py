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

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.triton_version_utils import has_triton_tle

logger = logging.getLogger(__name__)

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle_gpu

        HAS_TLE_EXTRACT_TILE = hasattr(tle_gpu, "extract_tile")
    except ImportError:
        tle_gpu = None
        HAS_TLE_EXTRACT_TILE = False
else:
    tle_gpu = None
    HAS_TLE_EXTRACT_TILE = False

def _valid_rope_configs(configs, named_args, **kwargs):
    """Keep only PAIR_TILE values that tile PADDED_HALF_DIM evenly.

    The TLE path slices the pair vector with ``tl.static_range(0,
    PADDED_HALF_DIM // PAIR_TILE)``, so a PAIR_TILE larger than the padded
    half dim would run zero iterations and silently produce no output.
    """
    padded_half_dim = named_args["PADDED_HALF_DIM"]
    return [c for c in configs if padded_half_dim % c.kwargs["PAIR_TILE"] == 0]


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"PAIR_TILE": 8}, num_warps=4),
        triton.Config({"PAIR_TILE": 16}, num_warps=4),
        triton.Config({"PAIR_TILE": 32}, num_warps=4),
        triton.Config({"PAIR_TILE": 16}, num_warps=2),
    ],
    key=[
        "PADDED_HALF_DIM",
        "HALF_DIM",
        "NUM_Q_HEADS",
        "NUM_K_HEADS",
        "ROTARY_INTERLEAVED",
        "USE_TLE",
    ],
    prune_configs_by={"early_config_prune": _valid_rope_configs},
    # Autotuning benchmarks each config on the tensors passed at launch; for
    # the in-place update (oq_ptr == q_ptr) this would clobber q/k before the
    # final run. Restore them after every benchmark so inputs stay intact.
    restore_value=["q_ptr", "k_ptr"],
)
@triton.jit
def apply_rotary_pos_emb_kernel(
    oq_ptr,
    ok_ptr,
    q_ptr,  # (n_tokens, q_heads, head_dim)
    k_ptr,  # (n_tokens, k_heads, head_dim)
    cos_ptr,  # (max_seq_len, head_dim // 2)
    sin_ptr,  # (max_seq_len, head_dim // 2)
    pos_ptr,  # (n_tokens, )
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    oq_stride_s,
    oq_stride_h,
    oq_stride_d,
    ok_stride_s,
    ok_stride_h,
    ok_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HALF_DIM: tl.constexpr,
    PADDED_HALF_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    USE_TLE: tl.constexpr,
    PAIR_TILE: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    s_id = tle.program_id(0)

    if pos_ptr is None:
        pos_id = s_id % seq_len
    else:
        pos_ptr += s_id * p_stride_s
        pos_id = tl.load(pos_ptr)

    # note: set TRITON_DEBUG=1 to enable this check
    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    pair_id = tl.arange(0, PADDED_HALF_DIM)
    pair_mask = pair_id < HALF_DIM

    if USE_TLE:
        cos_full = tle_gpu.load(
            cos_ptr + pair_id, mask=pair_mask, other=0.0, is_async=False
        ).to(tl.float32)
        sin_full = tle_gpu.load(
            sin_ptr + pair_id, mask=pair_mask, other=0.0, is_async=False
        ).to(tl.float32)
    else:
        cos_full = tl.load(cos_ptr + pair_id, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        sin_full = tl.load(sin_ptr + pair_id, mask=pair_mask, other=0.0).to(
            tl.float32
        )

    if ROTARY_INTERLEAVED:
        first_dim = pair_id * 2
        second_dim = first_dim + 1
    else:
        first_dim = pair_id
        second_dim = pair_id + HALF_DIM

    oq_ptr += s_id * oq_stride_s
    q_ptr += s_id * q_stride_s

    for off_h in range(0, NUM_Q_HEADS):
        q_head_off = off_h * q_stride_h
        oq_head_off = off_h * oq_stride_h

        if USE_TLE:
            q0_full = tle_gpu.load(
                q_ptr + q_head_off + first_dim * q_stride_d,
                mask=pair_mask,
                other=0.0,
                is_async=False,
            )
            q1_full = tle_gpu.load(
                q_ptr + q_head_off + second_dim * q_stride_d,
                mask=pair_mask,
                other=0.0,
                is_async=False,
            )
        else:
            q0_full = tl.load(
                q_ptr + q_head_off + first_dim * q_stride_d,
                mask=pair_mask,
                other=0.0,
            )
            q1_full = tl.load(
                q_ptr + q_head_off + second_dim * q_stride_d,
                mask=pair_mask,
                other=0.0,
            )

        # A complete rotary pair is loaded into registers before either element
        # is stored, so the in-place update (oq_ptr == q_ptr) does not depend on
        # cross-warp load/store ordering. See flagos-ai/FlagGems#5142.
        if USE_TLE:
            for tile_idx in tl.static_range(0, PADDED_HALF_DIM // PAIR_TILE):
                base = tile_idx * PAIR_TILE
                pair = base + tl.arange(0, PAIR_TILE)
                pair_tile_mask = pair < HALF_DIM
                cos_tile = tle_gpu.extract_tile(cos_full, tile_idx, (PAIR_TILE,))
                sin_tile = tle_gpu.extract_tile(sin_full, tile_idx, (PAIR_TILE,))
                q0_tile = tle_gpu.extract_tile(q0_full, tile_idx, (PAIR_TILE,))
                q1_tile = tle_gpu.extract_tile(q1_full, tile_idx, (PAIR_TILE,))

                y0 = q0_tile * cos_tile - q1_tile * sin_tile
                y1 = q1_tile * cos_tile + q0_tile * sin_tile

                if ROTARY_INTERLEAVED:
                    first = pair * 2
                    second = first + 1
                else:
                    first = pair
                    second = pair + HALF_DIM

                tl.store(
                    oq_ptr + oq_head_off + first * oq_stride_d,
                    y0,
                    mask=pair_tile_mask,
                )
                tl.store(
                    oq_ptr + oq_head_off + second * oq_stride_d,
                    y1,
                    mask=pair_tile_mask,
                )
        else:
            y0 = q0_full * cos_full - q1_full * sin_full
            y1 = q1_full * cos_full + q0_full * sin_full

            tl.store(
                oq_ptr + oq_head_off + first_dim * oq_stride_d, y0, mask=pair_mask
            )
            tl.store(
                oq_ptr + oq_head_off + second_dim * oq_stride_d, y1, mask=pair_mask
            )

    ok_ptr += s_id * ok_stride_s
    k_ptr += s_id * k_stride_s

    for off_h in range(0, NUM_K_HEADS):
        k_head_off = off_h * k_stride_h
        ok_head_off = off_h * ok_stride_h

        if USE_TLE:
            k0_full = tle_gpu.load(
                k_ptr + k_head_off + first_dim * k_stride_d,
                mask=pair_mask,
                other=0.0,
                is_async=False,
            )
            k1_full = tle_gpu.load(
                k_ptr + k_head_off + second_dim * k_stride_d,
                mask=pair_mask,
                other=0.0,
                is_async=False,
            )
            for tile_idx in tl.static_range(0, PADDED_HALF_DIM // PAIR_TILE):
                base = tile_idx * PAIR_TILE
                pair = base + tl.arange(0, PAIR_TILE)
                pair_tile_mask = pair < HALF_DIM
                cos_tile = tle_gpu.extract_tile(cos_full, tile_idx, (PAIR_TILE,))
                sin_tile = tle_gpu.extract_tile(sin_full, tile_idx, (PAIR_TILE,))
                k0_tile = tle_gpu.extract_tile(k0_full, tile_idx, (PAIR_TILE,))
                k1_tile = tle_gpu.extract_tile(k1_full, tile_idx, (PAIR_TILE,))

                y0 = k0_tile * cos_tile - k1_tile * sin_tile
                y1 = k1_tile * cos_tile + k0_tile * sin_tile

                if ROTARY_INTERLEAVED:
                    first = pair * 2
                    second = first + 1
                else:
                    first = pair
                    second = pair + HALF_DIM

                tl.store(
                    ok_ptr + ok_head_off + first * ok_stride_d,
                    y0,
                    mask=pair_tile_mask,
                )
                tl.store(
                    ok_ptr + ok_head_off + second * ok_stride_d,
                    y1,
                    mask=pair_tile_mask,
                )
        else:
            k0_full = tl.load(
                k_ptr + k_head_off + first_dim * k_stride_d,
                mask=pair_mask,
                other=0.0,
            )
            k1_full = tl.load(
                k_ptr + k_head_off + second_dim * k_stride_d,
                mask=pair_mask,
                other=0.0,
            )

            y0 = k0_full * cos_full - k1_full * sin_full
            y1 = k1_full * cos_full + k0_full * sin_full

            tl.store(
                ok_ptr + ok_head_off + first_dim * ok_stride_d, y0, mask=pair_mask
            )
            tl.store(
                ok_ptr + ok_head_off + second_dim * ok_stride_d, y1, mask=pair_mask
            )


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
    inplace: bool = False,
):
    logger.debug("MTHREADS GEMS ROTARY_POS_EMBEDDING")
    assert (
        k.shape[-1] == q.shape[-1]
    ), f"q and k must have the same last dimension, got {q.shape} and {k.shape}"
    assert (
        q.shape[-1] % 2 == 0
    ), f"q/k head_dim must be even, got {q.shape[-1]}"
    assert (
        cos.shape[-1] == sin.shape[-1]
    ), f"cos and sin must have the same last dimension, got {cos.shape} and {sin.shape}"
    assert (
        cos.shape[-1] * 2 == q.shape[-1]
    ), f"cos/sin dim must be half of q/k dim, got {cos.shape} and {q.shape}"
    assert cos.stride(-1) == 1, "cos must be contiguous at the last dimension"
    assert sin.stride(-1) == 1, "sin must be contiguous at the last dimension"
    assert (
        q.shape[:-2] == k.shape[:-2]
    ), f"q and k must have the same length, got {q.shape[:-2]} and {k.shape[:-2]}"

    q_shape = q.shape
    k_shape = k.shape

    if position_ids is None:
        assert (
            len(q.shape) == 4
        ), f"q must have 4 dimensions if position_ids is not provided, got {q.shape}"
        seq_len = q.shape[-3]
    else:
        assert (
            position_ids.shape == q.shape[:-2]
        ), f"position_ids must have the same length as q, got {position_ids.shape} and {q.shape[:-2]}"
        position_ids = position_ids.view(-1)
        seq_len = None

    q = q.view(-1, q.shape[-2], q.shape[-1])
    k = k.view(-1, k.shape[-2], k.shape[-1])

    n_tokens, _, head_dim = q.shape
    half_dim = head_dim // 2
    padded_half_dim = max(triton.next_power_of_2(half_dim), 16)

    grid = (n_tokens,)

    if inplace:
        q_embed = q
        k_embed = k
    else:
        q_embed = torch.empty_like(q)
        k_embed = torch.empty_like(k)

    with torch_device_fn.device(q.device):
        apply_rotary_pos_emb_kernel[grid](
            q_embed,
            k_embed,
            q,
            k,
            cos,
            sin,
            position_ids,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            q_embed.stride(0),
            q_embed.stride(1),
            q_embed.stride(2),
            k_embed.stride(0),
            k_embed.stride(1),
            k_embed.stride(2),
            position_ids.stride(0) if position_ids is not None else 0,
            cos.stride(0),
            sin.stride(0),
            seq_len,
            q.shape[-2],
            k.shape[-2],
            half_dim,
            padded_half_dim,
            rotary_interleaved,
            USE_TLE=HAS_TLE_EXTRACT_TILE,
            MAX_POSITION_EMBEDDINGS=cos.shape[0],
        )

    if inplace:
        return q.view(q_shape), k.view(k_shape)
    return q_embed.view(q_shape), k_embed.view(k_shape)