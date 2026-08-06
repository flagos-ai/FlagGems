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

logger = logging.getLogger(__name__)


@triton.jit
def _hc_head_partial_kernel(
    hs_ptr,
    fn_ptr,
    partial_ptr,
    hidden_size: tl.constexpr,
    num_chunks: tl.constexpr,
    hs_stride_t,
    hs_stride_m,
    hs_stride_h,
    fn_stride_m,
    fn_stride_k,
    HC: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    flat = pid_c * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = flat < HC * hidden_size
    stream = flat // hidden_size
    hidden = flat - stream * hidden_size
    x = tl.load(
        hs_ptr
        + pid_t * hs_stride_t
        + stream * hs_stride_m
        + hidden * hs_stride_h,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    out_base = (pid_t * num_chunks + pid_c) * (HC + 1)
    tl.store(partial_ptr + out_base, tl.sum(x * x))
    for mix in tl.static_range(HC):
        weight = tl.load(
            fn_ptr + mix * fn_stride_m + flat * fn_stride_k,
            mask=mask,
            other=0.0,
        )
        tl.store(partial_ptr + out_base + mix + 1, tl.sum(x * weight))


@triton.jit
def _hc_head_finalize_kernel(
    partial_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_mix_ptr,
    num_chunks: tl.constexpr,
    rms_eps,
    hc_eps,
    hidden_size: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_t = tl.program_id(0)
    chunk = tl.arange(0, BLOCK_R)
    mask = chunk < num_chunks
    base = pid_t * num_chunks * (HC + 1) + chunk * (HC + 1)
    sqrsum = tl.sum(tl.load(partial_ptr + base, mask=mask, other=0.0))
    inv_rms = tl.rsqrt(sqrsum / (HC * hidden_size) + rms_eps)
    scale = tl.load(hc_scale_ptr)

    for mix in tl.static_range(HC):
        dot = tl.sum(tl.load(partial_ptr + base + mix + 1, mask=mask, other=0.0))
        bias = tl.load(hc_base_ptr + mix)
        value = tl.sigmoid(dot * inv_rms * scale + bias) + hc_eps
        tl.store(pre_mix_ptr + pid_t * HC + mix, value)


@triton.jit
def _hc_head_apply_kernel(
    hs_ptr,
    pre_mix_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hs_stride_t,
    hs_stride_m,
    hs_stride_h,
    out_stride_t,
    out_stride_h,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    hidden = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = hidden < hidden_size
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for stream in tl.static_range(HC):
        x = tl.load(
            hs_ptr
            + pid_t * hs_stride_t
            + stream * hs_stride_m
            + hidden * hs_stride_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        mix = tl.load(pre_mix_ptr + pid_t * HC + stream)
        acc += x * mix
    tl.store(
        out_ptr + pid_t * out_stride_t + hidden * out_stride_h,
        acc,
        mask=mask,
    )


def hc_head_fused_kernel(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> torch.Tensor:
    logger.debug("GEMS HC_HEAD_FUSED")
    assert hs_flat.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32
    assert hc_mult in (2, 4)
    assert hs_flat.shape == (hs_flat.shape[0], hc_mult, hidden_size)
    assert fn.shape == (hc_mult, hc_mult * hidden_size)

    num_tokens = hs_flat.shape[0]
    if num_tokens == 0:
        return out

    block_k = 256
    num_chunks = triton.cdiv(hc_mult * hidden_size, block_k)
    partial = torch.empty(
        (num_tokens, num_chunks, hc_mult + 1),
        dtype=torch.float32,
        device=hs_flat.device,
    )
    pre_mix = torch.empty(
        (num_tokens, hc_mult), dtype=torch.float32, device=hs_flat.device
    )
    launch_options = {
        "num_warps": 4,
        "num_stages": 1,
        "isCloseVectorization": True,
        "buffer_size_limit": 2048,
        "unroll_num": 8,
    }

    _hc_head_partial_kernel[(num_tokens, num_chunks)](
        hs_flat,
        fn,
        partial,
        hidden_size=hidden_size,
        num_chunks=num_chunks,
        hs_stride_t=hs_flat.stride(0),
        hs_stride_m=hs_flat.stride(1),
        hs_stride_h=hs_flat.stride(2),
        fn_stride_m=fn.stride(0),
        fn_stride_k=fn.stride(1),
        HC=hc_mult,
        BLOCK_K=block_k,
        **launch_options,
    )
    _hc_head_finalize_kernel[(num_tokens,)](
        partial,
        hc_scale,
        hc_base,
        pre_mix,
        num_chunks=num_chunks,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        hidden_size=hidden_size,
        HC=hc_mult,
        BLOCK_R=triton.next_power_of_2(num_chunks),
        **launch_options,
    )
    block_h = 256
    _hc_head_apply_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        hs_flat,
        pre_mix,
        out,
        hidden_size=hidden_size,
        hs_stride_t=hs_flat.stride(0),
        hs_stride_m=hs_flat.stride(1),
        hs_stride_h=hs_flat.stride(2),
        out_stride_t=out.stride(0),
        out_stride_h=out.stride(1),
        HC=hc_mult,
        BLOCK_H=block_h,
        **launch_options,
    )
    return out
