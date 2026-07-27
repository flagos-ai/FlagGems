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

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_split_pre_post_kernel(
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_ptr,
    post_ptr,
    eps,
    HC: tl.constexpr,
    MIX_HC: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    mix_base = token * MIX_HC
    stream = tl.arange(0, BLOCK_C)
    stream_mask = stream < HC
    scale_pre = tl.load(hc_scale_ptr)
    scale_post = tl.load(hc_scale_ptr + 1)

    pre = tl.sigmoid(
        tl.load(mixes_ptr + mix_base + stream, mask=stream_mask, other=0.0)
        * scale_pre
        + tl.load(hc_base_ptr + stream, mask=stream_mask, other=0.0)
    ) + eps
    post = 2.0 * tl.sigmoid(
        tl.load(
            mixes_ptr + mix_base + HC + stream, mask=stream_mask, other=0.0
        )
        * scale_post
        + tl.load(hc_base_ptr + HC + stream, mask=stream_mask, other=0.0)
    )
    tl.store(pre_ptr + token * HC + stream, pre, mask=stream_mask)
    tl.store(post_ptr + token * HC + stream, post, mask=stream_mask)


@triton.jit
def _hc_split_comb_kernel(
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    comb_ptr,
    HC: tl.constexpr,
    MIX_HC: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    output_offset = tl.program_id(0)
    token = output_offset // BLOCK_C
    elem = output_offset - token * BLOCK_C
    mix_offset = token * MIX_HC + 2 * HC + elem
    base_offset = 2 * HC + elem
    scale_comb = tl.load(hc_scale_ptr + 2)

    comb = tl.load(mixes_ptr + mix_offset) * scale_comb + tl.load(
        hc_base_ptr + base_offset
    )
    tl.store(comb_ptr + output_offset, comb)


@triton.jit
def _sinkhorn_row_kernel(
    comb_ptr,
    num_tokens,
    eps,
    HC: tl.constexpr,
    ROW: tl.constexpr,
    SOFTMAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    token_mask = tokens < num_tokens
    row_base = tokens * HC * HC + ROW * HC
    value_0 = tl.load(comb_ptr + row_base, mask=token_mask, other=0.0)
    value_1 = tl.load(comb_ptr + row_base + 1, mask=token_mask, other=0.0)
    if HC == 4:
        value_2 = tl.load(comb_ptr + row_base + 2, mask=token_mask, other=0.0)
        value_3 = tl.load(comb_ptr + row_base + 3, mask=token_mask, other=0.0)

    if SOFTMAX:
        row_max = tl.maximum(value_0, value_1)
        if HC == 4:
            row_max = tl.maximum(row_max, tl.maximum(value_2, value_3))
        value_0 = tl.exp(value_0 - row_max)
        value_1 = tl.exp(value_1 - row_max)
        row_sum = value_0 + value_1
        if HC == 4:
            value_2 = tl.exp(value_2 - row_max)
            value_3 = tl.exp(value_3 - row_max)
            row_sum += value_2 + value_3
        inv_sum = 1.0 / row_sum
        value_0 = value_0 * inv_sum + eps
        value_1 = value_1 * inv_sum + eps
        if HC == 4:
            value_2 = value_2 * inv_sum + eps
            value_3 = value_3 * inv_sum + eps
    else:
        row_sum = value_0 + value_1
        if HC == 4:
            row_sum += value_2 + value_3
        inv_sum = 1.0 / (row_sum + eps)
        value_0 *= inv_sum
        value_1 *= inv_sum
        if HC == 4:
            value_2 *= inv_sum
            value_3 *= inv_sum

    tl.store(comb_ptr + row_base, value_0, mask=token_mask)
    tl.store(comb_ptr + row_base + 1, value_1, mask=token_mask)
    if HC == 4:
        tl.store(comb_ptr + row_base + 2, value_2, mask=token_mask)
        tl.store(comb_ptr + row_base + 3, value_3, mask=token_mask)


@triton.jit
def _sinkhorn_col_kernel(
    comb_ptr,
    num_tokens,
    eps,
    HC: tl.constexpr,
    COL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    token_mask = tokens < num_tokens
    col_base = tokens * HC * HC + COL
    value_0 = tl.load(comb_ptr + col_base, mask=token_mask, other=0.0)
    value_1 = tl.load(comb_ptr + col_base + HC, mask=token_mask, other=0.0)
    if HC == 4:
        value_2 = tl.load(comb_ptr + col_base + 2 * HC, mask=token_mask, other=0.0)
        value_3 = tl.load(comb_ptr + col_base + 3 * HC, mask=token_mask, other=0.0)

    col_sum = value_0 + value_1
    if HC == 4:
        col_sum += value_2 + value_3
    inv_sum = 1.0 / (col_sum + eps)
    tl.store(comb_ptr + col_base, value_0 * inv_sum, mask=token_mask)
    tl.store(comb_ptr + col_base + HC, value_1 * inv_sum, mask=token_mask)
    if HC == 4:
        tl.store(comb_ptr + col_base + 2 * HC, value_2 * inv_sum, mask=token_mask)
        tl.store(comb_ptr + col_base + 3 * HC, value_3 * inv_sum, mask=token_mask)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mix_hc = (2 + hc_mult) * hc_mult
    assert hc_mult in (2, 4)
    assert mixes.shape[-1] == mix_hc
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (mix_hc,)

    outer_shape = mixes.shape[:-1]
    num_tokens = mixes.numel() // mix_hc
    pre = torch.empty((*outer_shape, hc_mult), dtype=mixes.dtype, device=mixes.device)
    post = torch.empty_like(pre)
    comb = torch.empty(
        (*outer_shape, hc_mult, hc_mult), dtype=mixes.dtype, device=mixes.device
    )
    if num_tokens == 0:
        return pre, post, comb

    launch_options = {
        "num_warps": 4,
        "num_stages": 1,
        "isCloseVectorization": True,
        "buffer_size_limit": 2048,
        "unroll_num": 8,
    }
    _hc_split_pre_post_kernel[(num_tokens,)](
        mixes,
        hc_scale,
        hc_base,
        pre,
        post,
        eps,
        HC=hc_mult,
        MIX_HC=mix_hc,
        BLOCK_C=hc_mult * hc_mult,
        **launch_options,
    )
    _hc_split_comb_kernel[(num_tokens * hc_mult * hc_mult,)](
        mixes,
        hc_scale,
        hc_base,
        comb,
        HC=hc_mult,
        MIX_HC=mix_hc,
        BLOCK_C=hc_mult * hc_mult,
        **launch_options,
    )

    block_n = 256
    sinkhorn_grid = (triton.cdiv(num_tokens, block_n),)
    for row in range(hc_mult):
        _sinkhorn_row_kernel[sinkhorn_grid](
            comb,
            num_tokens,
            eps,
            HC=hc_mult,
            ROW=row,
            SOFTMAX=True,
            BLOCK_N=block_n,
            **launch_options,
        )
    for col in range(hc_mult):
        _sinkhorn_col_kernel[sinkhorn_grid](
            comb,
            num_tokens,
            eps,
            HC=hc_mult,
            COL=col,
            BLOCK_N=block_n,
            **launch_options,
        )
    for _ in range(sinkhorn_iters - 1):
        for row in range(hc_mult):
            _sinkhorn_row_kernel[sinkhorn_grid](
                comb,
                num_tokens,
                eps,
                HC=hc_mult,
                ROW=row,
                SOFTMAX=False,
                BLOCK_N=block_n,
                **launch_options,
            )
        for col in range(hc_mult):
            _sinkhorn_col_kernel[sinkhorn_grid](
                comb,
                num_tokens,
                eps,
                HC=hc_mult,
                COL=col,
                BLOCK_N=block_n,
                **launch_options,
            )
    return pre, post, comb
