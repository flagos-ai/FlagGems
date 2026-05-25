from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _compute_global_topk_indices_and_lens_kernel(
    global_indices_ptr,
    global_stride,
    lens_ptr,
    local_indices_ptr,
    local_stride,
    topk,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    is_valid_token_ptr,
    BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    count = tl.zeros((), dtype=tl.int32)

    for start in range(0, topk, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < topk
        local_idx = tl.load(
            local_indices_ptr + token_idx * local_stride + offs, mask=mask, other=-1
        )
        valid = local_idx >= 0
        block_idx = local_idx // block_size
        block_off = local_idx - block_idx * block_size
        block_no = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_idx,
            mask=mask & valid,
            other=0,
        )
        slot = block_no * block_size + block_off
        slot = tl.where(valid, slot, -1)
        tl.store(global_indices_ptr + token_idx * global_stride + offs, slot, mask=mask)
        count += tl.sum(valid.to(tl.int32), axis=0)

    tl.store(lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


def compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert topk_indices.ndim == 2
    if is_valid_token is None:
        is_valid_token = torch.ones(
            (topk_indices.shape[0],), device=topk_indices.device, dtype=torch.int32
        )
    num_tokens, topk = topk_indices.shape
    global_indices = torch.empty_like(topk_indices, dtype=torch.int32)
    lens = torch.empty((num_tokens,), device=topk_indices.device, dtype=torch.int32)
    with torch_device_fn.device(topk_indices.device):
        _compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
            global_indices,
            global_indices.stride(0),
            lens,
            topk_indices,
            topk_indices.stride(0),
            topk,
            token_to_req_indices,
            block_table,
            block_table.stride(0),
            block_size,
            is_valid_token,
            BLOCK=1024,
        )
    return global_indices, lens


dsv4_compute_global_topk_indices_and_lens = compute_global_topk_indices_and_lens

__all__ = [
    "compute_global_topk_indices_and_lens",
    "dsv4_compute_global_topk_indices_and_lens",
]
