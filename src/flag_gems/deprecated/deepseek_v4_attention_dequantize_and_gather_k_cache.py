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

from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


def _default_scale_slots(nope_dim: int) -> int:
    return triton.cdiv(nope_dim, 64) + (1 if nope_dim % 64 == 0 else 0)


def _as_cache_2d(k_cache: torch.Tensor) -> torch.Tensor:
    if k_cache.ndim == 2:
        return k_cache
    if k_cache.ndim == 3:
        if k_cache.is_contiguous():
            return k_cache.view(k_cache.shape[0], -1)
        return k_cache.contiguous().view(k_cache.shape[0], -1)
    raise ValueError(f"k_cache must be 2D or 3D, got shape={tuple(k_cache.shape)}")


@triton.jit
def _dequantize_and_gather_k_cache_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    scale_slots: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    cache_block_stride: tl.constexpr,
    output_dim: tl.constexpr,
    num_workers: tl.constexpr,
    n_quant_blocks: tl.constexpr,
    HAVE_GATHER_LENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    worker_idx = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + req_idx)
    if HAVE_GATHER_LENS:
        gather_len = tl.load(gather_lens_ptr + req_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for local_i in range(worker_idx, gather_len, num_workers):
        pos = start_pos + local_i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos - block_in_seq * cache_block_size
        physical_block = tl.load(
            block_table_ptr + req_idx * max_blocks_per_seq + block_in_seq
        )
        cache_block = k_cache_ptr + physical_block.to(tl.int64) * cache_block_stride
        token_data = cache_block + pos_in_block * token_data_size
        scale_base = (
            cache_block
            + cache_block_size * token_data_size
            + pos_in_block * scale_slots
        )
        out_row = out_ptr + req_idx * out_stride0 + (offset + local_i) * out_stride1

        if nope_dim % quant_block == 0:
            for qblock in tl.static_range(0, n_quant_blocks):
                qoffs = qblock * quant_block + tl.arange(0, quant_block)
                x_u8 = tl.load(token_data + qoffs)
                x_fp8 = x_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                encoded = tl.load(scale_base + qblock)
                scale = tl.exp2(encoded.to(tl.float32) - 127.0)
                x = x_fp8 * scale
                tl.store(out_row + qoffs, x.to(tl.bfloat16))
        else:
            for qblock in tl.static_range(0, n_quant_blocks - 1):
                qoffs = qblock * quant_block + tl.arange(0, quant_block)
                x_u8 = tl.load(token_data + qoffs)
                x_fp8 = x_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                encoded = tl.load(scale_base + qblock)
                scale = tl.exp2(encoded.to(tl.float32) - 127.0)
                x = x_fp8 * scale
                tl.store(out_row + qoffs, x.to(tl.bfloat16))

            qblock = n_quant_blocks - 1
            qoffs = qblock * quant_block + tl.arange(0, quant_block)
            qmask = qoffs < nope_dim
            x_u8 = tl.load(token_data + qoffs, mask=qmask, other=0)
            x_fp8 = x_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            encoded = tl.load(scale_base + qblock)
            scale = tl.exp2(encoded.to(tl.float32) - 127.0)
            x = x_fp8 * scale
            tl.store(out_row + qoffs, x.to(tl.bfloat16), mask=qmask)

        bf16_ptr = (token_data + nope_dim).to(tl.pointer_type(tl.bfloat16))
        for rblock in tl.static_range(0, rope_dim, 64):
            roffs = rblock + tl.arange(0, 64)
            rmask = roffs < rope_dim
            vals = tl.load(bf16_ptr + roffs, mask=rmask, other=0.0)
            tl.store(out_row + nope_dim + roffs, vals, mask=rmask)


def dequantize_and_gather_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    block_table: torch.Tensor,
    block_size: int,
    offset: int = 0,
    rope_dim: int = 64,
    nope_dim: Optional[int] = None,
    scale_slots: Optional[int] = None,
) -> None:
    assert out.ndim == 3 and out.dtype == torch.bfloat16
    assert seq_lens.ndim == 1 and block_table.ndim == 2
    assert seq_lens.shape[0] == block_table.shape[0] <= out.shape[0]
    output_dim = out.shape[-1]
    if nope_dim is None:
        nope_dim = output_dim - rope_dim
    if scale_slots is None:
        scale_slots = _default_scale_slots(nope_dim)

    n_quant_blocks = triton.cdiv(nope_dim, 64)
    assert nope_dim + rope_dim <= output_dim
    k_cache_2d = _as_cache_2d(k_cache)
    token_data_size = nope_dim + rope_dim * 2
    num_reqs = seq_lens.shape[0]
    num_workers = 128
    with torch_device_fn.device(out.device):
        _dequantize_and_gather_k_cache_kernel[(num_reqs, num_workers)](
            out,
            out.stride(0),
            out.stride(1),
            k_cache_2d,
            seq_lens,
            block_table,
            offset,
            gather_lens,
            block_table.shape[-1],
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            scale_slots=scale_slots,
            quant_block=64,
            cache_block_size=block_size,
            token_data_size=token_data_size,
            cache_block_stride=k_cache_2d.stride(0),
            output_dim=output_dim,
            num_workers=num_workers,
            n_quant_blocks=n_quant_blocks,
            HAVE_GATHER_LENS=gather_lens is not None,
        )


__all__ = ["dequantize_and_gather_k_cache"]
