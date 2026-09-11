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

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner


def _prune_configs(configs, named_args, **kwargs):
    topk = named_args["topk"]
    num_tokens = named_args["num_tokens"]
    pruned = []
    for cfg in configs:
        BLOCK = cfg.kwargs.get("BLOCK", 1024)
        TPP = cfg.kwargs.get("TPP", 1)
        if BLOCK * TPP > 1024:
            continue
        if BLOCK > topk * 4:
            continue
        if num_tokens <= 64 and TPP >= 8:
            continue
        if num_tokens <= 32 and TPP >= 4:
            continue
        if num_tokens <= 16 and TPP >= 2:
            continue
        if TPP > triton.cdiv(num_tokens, 2):
            continue
        pruned.append(cfg)
    return pruned


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("compute_global_topk_indices_and_lens"),
    key=["topk", "num_tokens"],
    strategy=["log", "log"],
    prune_configs_by={"early_config_prune": _prune_configs},
    reset_to_zero=["lens_ptr"],
    flagtune_op_name="compute_global_topk_indices_and_lens",
    flagtune_expand_op_name="compute_global_topk_indices_and_lens",
)
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
    num_tokens,
    BLOCK: tl.constexpr,
    TPP: tl.constexpr,
):
    pid = tl.program_id(0)
    token_start = pid * TPP
    token_offs = token_start + tl.arange(0, TPP)
    token_mask = token_offs < num_tokens

    is_valid = tl.load(is_valid_token_ptr + token_offs, mask=token_mask, other=0)
    req_idx = tl.load(token_to_req_indices_ptr + token_offs, mask=token_mask, other=0)

    local_base = token_offs[:, None] * local_stride
    global_base = token_offs[:, None] * global_stride
    block_table_base = req_idx[:, None] * block_table_stride

    counts = tl.zeros((TPP,), dtype=tl.int32)

    for start in range(0, topk, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        topk_mask = offs < topk
        mask_2d = token_mask[:, None] & topk_mask[None, :]

        local_idx = tl.load(
            local_indices_ptr + local_base + offs[None, :],
            mask=mask_2d,
            other=-1,
        )
        valid = local_idx >= 0

        block_idx = local_idx // block_size
        block_off = local_idx - block_idx * block_size

        block_no = tl.load(
            block_table_ptr + block_table_base + block_idx,
            mask=mask_2d & valid,
            other=0,
        )
        slot = block_no * block_size + block_off
        slot = tl.where(valid, slot, -1)

        tl.store(
            global_indices_ptr + global_base + offs[None, :],
            slot,
            mask=mask_2d,
        )
        counts += tl.sum(valid.to(tl.int32), axis=1)

    lens = tl.where(is_valid != 0, counts, 0)
    tl.store(lens_ptr + token_offs, lens, mask=token_mask)


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
        grid = lambda meta: (triton.cdiv(num_tokens, meta["TPP"]),)
        _compute_global_topk_indices_and_lens_kernel[grid](
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
            num_tokens,
        )
    return global_indices, lens


__all__ = [
    "compute_global_topk_indices_and_lens",
]
