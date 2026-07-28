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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


DECODE_GROUP = 8
PREFILL_GROUP = 8
DECODE_WARPS = 4
SMALL_WARPS = 2
LARGE_WARPS = 4
SCRATCH_MIN_TOKENS = 1 << 30


@triton.jit
def _axis_for_pair(
    pair,
    axis_map,
    st: tl.constexpr,
    sh: tl.constexpr,
    sw: tl.constexpr,
    interleaved: tl.constexpr,
    glm: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if glm:
        axis = tl.load(axis_map + pair, mask=True, other=0).to(tl.int32)
    elif interleaved:
        mod3 = pair % 3
        is_h = (mod3 == 1) & (pair < sh * 3)
        is_w = (mod3 == 2) & (pair < sw * 3)
        axis = tl.where(is_h, 1, tl.where(is_w, 2, 0)).to(tl.int32)
    else:
        axis = tl.where(pair < st, 0, tl.where(pair < st + sh, 1, 2)).to(tl.int32)
    return axis


@triton.jit
def _rotate_one_head(
    inp, base, rotary_dim: tl.constexpr, neox: tl.constexpr, valid, pair, cv, sv
):
    half = rotary_dim // 2
    if neox:
        d0 = pair
        d1 = pair + half
    else:
        d0 = pair * 2
        d1 = d0 + 1

    x0 = tl.load(inp + base + d0, mask=valid, other=0.0).to(tl.float32)
    x1 = tl.load(inp + base + d1, mask=valid, other=0.0).to(tl.float32)
    y0 = x0 * cv - x1 * sv
    y1 = x1 * cv + x0 * sv
    tl.store(inp + base + d0, y0, mask=valid)
    tl.store(inp + base + d1, y1, mask=valid)


@triton.jit
def _mrope_pair_kernel(
    q,
    k,
    cache,
    pos,
    axis_map,
    n_tokens: tl.constexpr,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    head_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    st: tl.constexpr,
    sh: tl.constexpr,
    sw: tl.constexpr,
    interleaved: tl.constexpr,
    glm: tl.constexpr,
    neox: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head_group = tl.program_id(1) * GROUP
    heads = head_group + tl.arange(0, GROUP)
    pair = tl.arange(0, BLOCK)
    half = rotary_dim // 2
    valid = pair < half

    axis = _axis_for_pair(pair, axis_map, st, sh, sw, interleaved, glm, BLOCK)
    p0 = tl.load(pos + row).to(tl.int32)
    p1 = tl.load(pos + n_tokens + row).to(tl.int32)
    p2 = tl.load(pos + n_tokens * 2 + row).to(tl.int32)
    p = tl.where(axis == 0, p0, tl.where(axis == 1, p1, p2))
    cv = tl.load(cache + p * rotary_dim + pair, mask=valid, other=1.0).to(tl.float32)
    sv = tl.load(cache + p * rotary_dim + half + pair, mask=valid, other=0.0).to(
        tl.float32
    )

    if neox:
        d0 = pair
        d1 = pair + half
    else:
        d0 = pair * 2
        d1 = d0 + 1

    h = heads[:, None]
    d0b = d0[None, :]
    d1b = d1[None, :]
    pair_valid = valid[None, :]
    cvb = cv[None, :]
    svb = sv[None, :]

    q_base = row * q_heads * head_size + h * head_size
    q_valid = pair_valid & (h < q_heads)
    q0 = tl.load(q + q_base + d0b, mask=q_valid, other=0.0).to(tl.float32)
    q1 = tl.load(q + q_base + d1b, mask=q_valid, other=0.0).to(tl.float32)
    tl.store(q + q_base + d0b, q0 * cvb - q1 * svb, mask=q_valid)
    tl.store(q + q_base + d1b, q1 * cvb + q0 * svb, mask=q_valid)

    k_base = row * k_heads * head_size + h * head_size
    k_valid = pair_valid & (h < k_heads)
    k0 = tl.load(k + k_base + d0b, mask=k_valid, other=0.0).to(tl.float32)
    k1 = tl.load(k + k_base + d1b, mask=k_valid, other=0.0).to(tl.float32)
    tl.store(k + k_base + d0b, k0 * cvb - k1 * svb, mask=k_valid)
    tl.store(k + k_base + d1b, k1 * cvb + k0 * svb, mask=k_valid)


@triton.jit
def _select_cache_kernel(
    cache,
    pos,
    axis_map,
    scratch,
    n_tokens: tl.constexpr,
    rotary_dim: tl.constexpr,
    st: tl.constexpr,
    sh: tl.constexpr,
    sw: tl.constexpr,
    interleaved: tl.constexpr,
    glm: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    half = rotary_dim // 2
    mask = offs < rotary_dim
    pair = offs % half
    axis = _axis_for_pair(pair, axis_map, st, sh, sw, interleaved, glm, BLOCK)
    p0 = tl.load(pos + row).to(tl.int32)
    p1 = tl.load(pos + n_tokens + row).to(tl.int32)
    p2 = tl.load(pos + n_tokens * 2 + row).to(tl.int32)
    p = tl.where(axis == 0, p0, tl.where(axis == 1, p1, p2))
    v = tl.load(cache + p * rotary_dim + offs, mask=mask, other=0.0)
    tl.store(scratch + row * rotary_dim + offs, v, mask=mask)


@triton.jit
def _mrope_pair_cached_kernel(
    q,
    k,
    scratch,
    n_tokens: tl.constexpr,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    head_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    neox: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head_group = tl.program_id(1) * GROUP
    heads = head_group + tl.arange(0, GROUP)
    pair = tl.arange(0, BLOCK)
    half = rotary_dim // 2
    valid = pair < half
    cv = tl.load(scratch + row * rotary_dim + pair, mask=valid, other=1.0).to(
        tl.float32
    )
    sv = tl.load(scratch + row * rotary_dim + half + pair, mask=valid, other=0.0).to(
        tl.float32
    )

    if neox:
        d0 = pair
        d1 = pair + half
    else:
        d0 = pair * 2
        d1 = d0 + 1

    h = heads[:, None]
    d0b = d0[None, :]
    d1b = d1[None, :]
    pair_valid = valid[None, :]
    cvb = cv[None, :]
    svb = sv[None, :]

    q_base = row * q_heads * head_size + h * head_size
    q_valid = pair_valid & (h < q_heads)
    q0 = tl.load(q + q_base + d0b, mask=q_valid, other=0.0).to(tl.float32)
    q1 = tl.load(q + q_base + d1b, mask=q_valid, other=0.0).to(tl.float32)
    tl.store(q + q_base + d0b, q0 * cvb - q1 * svb, mask=q_valid)
    tl.store(q + q_base + d1b, q1 * cvb + q0 * svb, mask=q_valid)

    k_base = row * k_heads * head_size + h * head_size
    k_valid = pair_valid & (h < k_heads)
    k0 = tl.load(k + k_base + d0b, mask=k_valid, other=0.0).to(tl.float32)
    k1 = tl.load(k + k_base + d1b, mask=k_valid, other=0.0).to(tl.float32)
    tl.store(k + k_base + d0b, k0 * cvb - k1 * svb, mask=k_valid)
    tl.store(k + k_base + d1b, k1 * cvb + k0 * svb, mask=k_valid)


@triton.jit
def _mrope_key_tail_kernel(
    k,
    cache,
    pos,
    axis_map,
    n_tokens: tl.constexpr,
    k_heads: tl.constexpr,
    head_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    st: tl.constexpr,
    sh: tl.constexpr,
    sw: tl.constexpr,
    interleaved: tl.constexpr,
    glm: tl.constexpr,
    neox: tl.constexpr,
    q_heads_done: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1) + q_heads_done
    pair = tl.arange(0, BLOCK)
    half = rotary_dim // 2
    valid = pair < half

    axis = _axis_for_pair(pair, axis_map, st, sh, sw, interleaved, glm, BLOCK)
    p0 = tl.load(pos + row).to(tl.int32)
    p1 = tl.load(pos + n_tokens + row).to(tl.int32)
    p2 = tl.load(pos + n_tokens * 2 + row).to(tl.int32)
    p = tl.where(axis == 0, p0, tl.where(axis == 1, p1, p2))
    cv = tl.load(cache + p * rotary_dim + pair, mask=valid, other=1.0).to(tl.float32)
    sv = tl.load(cache + p * rotary_dim + half + pair, mask=valid, other=0.0).to(
        tl.float32
    )

    base = row * k_heads * head_size + head * head_size
    _rotate_one_head(k, base, rotary_dim, neox, valid, pair, cv, sv)


def mrope(
    query,
    key,
    cos_sin_cache,
    positions,
    section_t,
    section_h,
    section_w,
    head_size,
    rotary_dim,
    is_interleaved=False,
    is_interleaved_glm=False,
    is_neox_style=True,
    axis_map=None,
):
    logger.debug("GEMS MROPE")
    n_tokens = query.shape[0]
    q_heads = query.shape[1] // head_size
    k_heads = key.shape[1] // head_size
    block = rotary_dim // 2
    group = DECODE_GROUP if n_tokens == 1 else PREFILL_GROUP
    warps = (
        DECODE_WARPS
        if n_tokens == 1
        else (SMALL_WARPS if rotary_dim <= 64 or n_tokens <= 512 else LARGE_WARPS)
    )
    axis_arg = axis_map if axis_map is not None else positions

    with torch_device_fn.device(query.device):
        if n_tokens >= SCRATCH_MIN_TOKENS:
            scratch = torch.empty(
                (n_tokens, rotary_dim), device=query.device, dtype=query.dtype
            )
            _select_cache_kernel[(n_tokens,)](
                cos_sin_cache,
                positions,
                axis_arg,
                scratch,
                n_tokens,
                rotary_dim,
                section_t,
                section_h,
                section_w,
                bool(is_interleaved),
                bool(is_interleaved_glm),
                BLOCK=rotary_dim,
                num_warps=4,
            )
            _mrope_pair_cached_kernel[(n_tokens, triton.cdiv(q_heads, group))](
                query,
                key,
                scratch,
                n_tokens,
                q_heads,
                k_heads,
                head_size,
                rotary_dim,
                bool(is_neox_style),
                GROUP=group,
                BLOCK=block,
                num_warps=warps,
            )
        else:
            _mrope_pair_kernel[(n_tokens, triton.cdiv(q_heads, group))](
                query,
                key,
                cos_sin_cache,
                positions,
                axis_arg,
                n_tokens,
                q_heads,
                k_heads,
                head_size,
                rotary_dim,
                section_t,
                section_h,
                section_w,
                bool(is_interleaved),
                bool(is_interleaved_glm),
                bool(is_neox_style),
                GROUP=group,
                BLOCK=block,
                num_warps=warps,
            )
        if k_heads > q_heads:
            _mrope_key_tail_kernel[(n_tokens, k_heads - q_heads)](
                key,
                cos_sin_cache,
                positions,
                axis_arg,
                n_tokens,
                k_heads,
                head_size,
                rotary_dim,
                section_t,
                section_h,
                section_w,
                bool(is_interleaved),
                bool(is_interleaved_glm),
                bool(is_neox_style),
                q_heads,
                BLOCK=block,
                num_warps=warps,
            )
    return query, key
